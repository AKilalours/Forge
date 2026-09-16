"""Load a trained arm and score one document, on CPU.

This exists so the UI and the API score text the same way the evaluation did. The window
size, stride, aggregation and threshold all come from the arm's own config and its committed
`summary.json`, not from constants retyped here. If evaluation and serving disagree about
any of those, the number a user sees is not the number the results tables describe.

**Mean over windows is the score, max is reported alongside.** That matches
`scripts/eval_ood.py`. Max pooling raises the score of any long document that contains one
unusual passage, which inflates FPR on exactly the human writing this project exists to
protect.

**The threshold is the DEPLOYED one**, fit on the arm's validation split at the 0.1% FPR
budget and read from `summary.json`. It is the only threshold available outside an
experiment. The optimistic re-fit thresholds used in the OOD comparison are not usable here
and are not offered.

**Abstention is off unless a validation score file exists.** `decision.band_from_validation`
derives the uncertain band from validation scores, and those are not committed (they come
from the corpus, which this project does not redistribute). Picking a band by eye would
produce either an abstention rate nobody accepts or a band that never fires, so the policy
reports `abstains=False` with a stated reason rather than inventing one.

**Out-of-distribution behaviour, stated because the UI cannot detect it.** These checkpoints
were trained on four generator families at 1.7B to 3.8B parameters. Against unseen
generators they miss 63% to 96% of AI text at this threshold and their ECE rises from 0.004
to between 0.18 and 0.44. A confident score here is not evidence of a confident model. See
docs/evaluation.md.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from functools import lru_cache

from forge.common.config import load
from forge.inference.decision import DecisionPolicy

ARMS = ("baseline", "mirror")
ARM_LABEL = {"baseline": "A: random synthetic", "mirror": "B: matched mirrors"}


class ArmUnavailable(RuntimeError):
    """Raised with a reason a user can act on, never swallowed into a fake score."""


@dataclass
class ArmScore:
    arm: str
    label: str
    mean: float
    maximum: float
    n_windows: int
    window_probabilities: list[float]


@dataclass
class Arm:
    name: str
    experiment: str
    model: object
    tokenizer: object
    mcfg: dict
    policy: DecisionPolicy
    summary: dict

    def score(self, text: str) -> ArmScore:
        import torch
        from torch.utils.data import DataLoader

        from forge.training.data import Collator, RawExample, build_dataset

        if not text or not text.strip():
            raise ArmUnavailable("empty text")

        example = RawExample(
            doc_id="query", source_group_id="query", split="test",
            text=text, label=0, spans=None, domain="unknown",
            generator_family="unknown",
        )
        feats = build_dataset(
            [example], self.tokenizer,
            max_length=self.mcfg["max_length"], stride=self.mcfg["window"]["stride"],
        )
        if not feats:
            raise ArmUnavailable(
                "windowing produced no features; the text is shorter than one token"
            )

        loader = DataLoader(
            feats, batch_size=32, shuffle=False,
            collate_fn=Collator(self.tokenizer, max_length=self.mcfg["max_length"]),
        )
        self.model.eval()
        probs: list[float] = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch["input_ids"], batch["attention_mask"])
                p = torch.softmax(out["doc_logits"].float(), dim=-1)[:, 1]
                probs.extend(p.numpy().tolist())

        return ArmScore(
            arm=self.name, label=ARM_LABEL[self.name],
            mean=float(sum(probs) / len(probs)), maximum=float(max(probs)),
            n_windows=len(probs), window_probabilities=[float(p) for p in probs],
        )


def _paths(arm: str) -> tuple[pathlib.Path, pathlib.Path, dict, dict]:
    config_path = f"configs/training/{arm}_minimal.yaml"
    cfg = load(config_path)
    mcfg = load(cfg["model_config"])
    out = pathlib.Path(cfg["paths"].get("out", "outputs")) / cfg["experiment"]["id"]
    return out / "best.pt", out / "summary.json", cfg, mcfg


@lru_cache(maxsize=len(ARMS))
def load_arm(arm: str) -> Arm:
    """Load one arm onto CPU. Cached, because a 735 MB load per request is not serving.

    Raises ArmUnavailable with the missing path rather than returning a degraded object.
    A caller that gets an Arm back can score with it; there is no half-loaded state.
    """
    if arm not in ARMS:
        raise ArmUnavailable(f"unknown arm {arm!r}, expected one of {ARMS}")

    checkpoint, summary_path, cfg, mcfg = _paths(arm)
    if not checkpoint.exists():
        raise ArmUnavailable(
            f"no checkpoint at {checkpoint}. Train this arm, or copy the inference weights "
            "there. Serving a score without weights would be fabricating a result."
        )
    if not summary_path.exists():
        raise ArmUnavailable(
            f"no {summary_path}. The deployed threshold lives there; without it a score "
            "cannot be turned into a verdict, and guessing a threshold would silently "
            "change the false-positive rate this project is built around."
        )

    from transformers import AutoTokenizer

    from forge.modeling.encoder import ForgeConfig, build_model
    from forge.training.train import load_checkpoint

    summary = json.loads(summary_path.read_text())
    model = build_model(ForgeConfig(
        backbone=mcfg["backbone"], max_length=mcfg["max_length"],
        stride=mcfg["window"]["stride"],
    ))
    load_checkpoint(checkpoint, model)

    # FLOAT32, ALWAYS, ON CPU. Training ran in bf16, so parts of the checkpoint come back
    # half precision. CPU matrix multiply refuses to mix: "mat1 and mat2 must have the same
    # dtype, but got Half and Float". The failure is at the first linear layer, so nothing
    # scores at all, and it only appears on a machine without a GPU, which is exactly where
    # this code is meant to run. .float() is a no-op on an already-float model.
    model.float()
    model.eval()

    policy = DecisionPolicy(
        threshold=float(summary["val"]["threshold"]),
        fpr_budget=float(cfg.get("fpr_budget", 0.001)),
        model_version=f"{cfg['experiment']['id']}@{summary.get('code_commit', 'unknown')[:8]}",
        calibrated=True,
        # No abstention band: see the module docstring. Validation scores are not committed.
    )
    return Arm(
        name=arm, experiment=cfg["experiment"]["id"], model=model,
        tokenizer=AutoTokenizer.from_pretrained(mcfg["backbone"]),
        mcfg=mcfg, policy=policy, summary=summary,
    )


def available() -> dict[str, str]:
    """Which arms can serve, and for those that cannot, why. Never raises."""
    state: dict[str, str] = {}
    for arm in ARMS:
        try:
            load_arm(arm)
            state[arm] = "ready"
        except Exception as error:                     # noqa: BLE001 - reported, not raised
            state[arm] = str(error)
    return state
