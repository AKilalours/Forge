"""Phase 3 training entrypoint. The fit loop.

Decisions here that change results, stated because they are the ones worth arguing about:

**Validation reports FPR at the budget, not just loss.** A run whose val loss improved but
whose FPR-at-budget got worse has not improved for this product.

**Checkpoint selection uses FPR-at-budget as a CONSTRAINT and FNR as the objective.** It did
not always. The first version selected on FPR-at-budget itself, which cannot work: that
number is measured at a threshold that has just been moved until the budget is met, so it is
pinned to the nearest achievable value and is the same at every epoch. In the Tier 1 runs it
was 1/1993 = 0.000502 at every evaluation of both arms, and with a `<=` comparison the last
epoch therefore always won. `best.pt` was byte-identical to `last.pt` in both runs, and the
selection step was decorative. The metric being flat is what made it invisible: nothing ever
looked wrong, because a constant never regresses. See tests/unit/test_checkpoint_selection.py.

**Checkpoint resume is mandatory, not a nicety.** Spot GPU instances are the affordable
way to run this, and a run that cannot resume turns every preemption into a lost day.
Resume restores model, optimizer, scheduler, scaler, epoch and step.

**Grouped batching is not needed; grouped SPLITTING already happened.** Windows from one
document can share a batch safely because the split boundary is at the group level.

**--smoke runs 20 steps on 200 examples.** Find the bugs in two minutes, not two hours
into a paid run. Every real run should be preceded by a smoke run on the same config.

torch and transformers are imported lazily so the package imports without them.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from forge.common.config import REPO_ROOT, load
from forge.evaluation import metrics as M
from forge.evaluation.calibration import fit_temperature


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")     # FPR at budget of the selected checkpoint
    best_fnr: float = float("inf")        # FNR at that budget; THIS is the objective
    best_in_budget: bool = False          # whether the selected checkpoint met the budget
    best_step: int = 0
    dataset_version: str = ""
    code_commit: str = ""
    temperature: float = 1.0
    threshold: float = 0.5


def _code_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def enable_gradient_checkpointing(encoder) -> bool:
    """Turn on gradient checkpointing, NON-REENTRANT. Returns whether it was enabled.

    THE BUG THIS CAME FROM. The first minimal smoke run died on its first backward pass:

        RuntimeError: Trying to backward through the graph a second time (or directly
        access saved tensors after they have already been freed)

    raised from inside torch/utils/checkpoint.py's own backward, which is the tell.

    WHY, isolated to a single config field rather than guessed at.
    `gradient_checkpointing_enable()` defaults to REENTRANT checkpointing, which recomputes
    each block during backward and assumes the block is a closed function of its inputs.
    DeBERTa-v2's disentangled attention needs the relative position embeddings, and the
    encoder computes those ONCE, outside the per-layer blocks, then hands the same tensor to
    every layer. deberta-v3-base sets norm_rel_ebd="layer_norm", so that tensor is the OUTPUT
    OF A LAYERNORM and carries an autograd graph living outside the checkpointed blocks. The
    first layer's backward frees it; the next layer's recomputation walks it again. Hence
    "a second time".

    Set norm_rel_ebd="none" and the same tensor is a leaf parameter with no graph behind it,
    and reentrant checkpointing runs cleanly. That one field is the entire difference, which
    is why the crash appeared on the real backbone and on no smaller stand-in.
    tests/unit/test_gradient_checkpointing.py reproduces both sides.

    Non-reentrant checkpointing tracks tensors that come from outside the block, so the
    shared embedding is handled correctly.

    This could not have been caught by any test that avoids a GPU-shaped model, and it cost
    a model download plus a few minutes rather than a training run, which is the entire
    argument for --smoke existing.

    Returns False rather than raising when the encoder does not support checkpointing at
    all, because a missing memory optimisation is not a reason to refuse to train. It is
    logged by the caller.
    """
    if not hasattr(encoder, "gradient_checkpointing_enable"):
        return False
    try:
        encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        # transformers older than 4.35 has no kwargs passthrough. Enabling reentrant
        # checkpointing on DeBERTa-v2 would fail on the first backward, so refuse loudly
        # instead of enabling something known to be broken.
        raise RuntimeError(
            "this transformers version cannot select non-reentrant gradient checkpointing, "
            "and reentrant checkpointing crashes on DeBERTa-v2's shared relative position "
            "embeddings. Upgrade transformers, or set gradient_checkpointing: false."
        ) from None
    return True


def save_checkpoint(path, model, optimizer, scheduler, scaler, state: TrainState) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "state": asdict(state),
    }, path)
    # Sidecar so a checkpoint can be identified without loading it.
    Path(str(path) + ".json").write_text(json.dumps(asdict(state), indent=2) + "\n")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None,
                    mmap: bool = False) -> TrainState:
    """Restore a checkpoint. `mmap=True` maps the tensors instead of reading them.

    Serving passes it. Without it, torch.load holds all 735 MB of weights in memory while
    the model already holds its own 736 MB copy, so the peak is roughly twice the model
    at the one moment a 2.7 GB host can least afford it. Mapped pages are read on demand
    and are evictable, and load_state_dict copies into the model either way, so the
    resident result is identical.

    Training keeps the default: a resume also restores optimizer state, which is not a
    parameter of the model and is not what this helps with.
    """
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
    model.load_state_dict(ck["model"])
    if optimizer and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return TrainState(**ck["state"])


@dataclass
class ValResult:
    loss: float
    fpr_at_budget: float
    threshold: float
    fnr: float
    auroc: float
    ece: float
    temperature: float
    n_human: int
    n_ai: int

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def is_better_checkpoint(val: ValResult, state: TrainState, fpr_budget: float) -> bool:
    """Should this evaluation replace the saved best checkpoint?

    The rule: **meeting the false-positive budget is a constraint, missing less AI text is
    the objective.** A checkpoint inside the budget always beats one outside it, however
    good the outside one looks. Among checkpoints inside the budget, lower FNR wins. Among
    checkpoints outside it, the one closest to the budget wins, so that a run which never
    reaches the budget still saves something rather than nothing.

    Why this replaces `val.fpr_at_budget <= state.best_metric`:

    `fpr_at_budget` is measured at a threshold that was just chosen to hit the budget. The
    threshold moves until the FPR lands at the closest achievable value, so the number is
    pinned by construction. With 1,993 human validation documents the only reachable values
    near a 0.1% budget are 0 and 1/1993 = 0.000502, and every epoch of both Tier 1 arms
    reported exactly 0.000502. Selecting on a constant is not selection, and `<=` then handed
    the win to whichever epoch came last. `best.pt` and `last.pt` were identical files.

    FNR at that same threshold is the quantity that actually varies (0.0072 against 0.0043
    between the two arms) and is the one the product cares about. Comparison is strict, so a
    tie keeps the EARLIER checkpoint rather than silently drifting to the last epoch again.
    """
    in_budget = val.fpr_at_budget <= fpr_budget
    if in_budget != state.best_in_budget:
        return in_budget                      # inside the budget always beats outside it
    if in_budget:
        return val.fnr < state.best_fnr       # objective
    return val.fpr_at_budget < state.best_metric   # nothing qualifies yet; get closer


def evaluate_split(model, loader, device, fpr_budget: float, calibrate: bool = True) -> ValResult:
    import torch
    from torch import nn

    model.eval()
    logits_all, labels_all, losses = [], [], []
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            dl = out["doc_logits"].float()
            losses.append(ce(dl, batch["doc_labels"]).item())
            logits_all.append(dl.cpu().numpy())
            labels_all.append(batch["doc_labels"].cpu().numpy())

    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)

    temp = fit_temperature(logits, labels) if calibrate and len(labels) > 50 else 1.0
    z = logits / temp
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    scores = (p / p.sum(axis=1, keepdims=True))[:, 1]

    n_human = int((labels == 0).sum())
    # A threshold cannot be fitted without human examples in the split.
    thr = M.threshold_at_fpr(labels, scores, fpr_budget) if n_human else 0.5
    return ValResult(
        loss=float(np.mean(losses)),
        fpr_at_budget=M.false_positive_rate(labels, scores, thr),
        threshold=float(thr),
        fnr=M.false_negative_rate(labels, scores, thr),
        auroc=M.auroc(labels, scores),
        ece=M.expected_calibration_error(labels, scores),
        temperature=float(temp),
        n_human=n_human,
        n_ai=int((labels == 1).sum()),
    )


def validate_config(cfg: dict) -> str:
    """Config checks that must not require torch.

    Config errors should surface in a second on a laptop, not after a GPU stack has
    loaded on a paid pod.
    """
    arm = cfg.get("data", {}).get("arm")
    if arm is None:
        raise RuntimeError(
            "training config must declare data.arm (random | mirror | hard_negative). "
            "Without it the loader cannot verify the arm is reading its own data, and "
            "two arms pointed at the same directory produce a false finding silently."
        )
    paths = cfg.get("paths", {})
    if not (paths.get("ai") or paths.get("mirror")):
        raise RuntimeError("training config must declare paths.ai, this arm's AI source")
    return arm


# data.ai_reference accepts this instead of a directory: match the AI pool's length
# distribution to the HUMAN corpus rather than to the other arm.
REFERENCE_HUMAN = "human"


def _reference_arm(dcfg: dict, human_rows=None):
    """Load the pool whose length distribution this arm's AI documents should match.

    WHY THIS CHANGED, and it is the difference between a result and an artifact.

    The first version pointed both arms at the MIRROR arm's directory: the mirror arm
    matched itself (a no-op) and the control arm was reshaped to match it. That makes the
    two arms comparable to each other, which is what it was written for, and does nothing
    about the gap between AI and HUMAN text. Both arms were left with AI documents running
    about 20% longer than the human corpus, and a detector reached AUROC 0.70 on the mirror
    arm and 0.72 on the control arm WITHOUT READING A WORD.

    Setting ai_reference to "human" reshapes each arm's AI pool to the human length
    distribution instead. Length then carries almost no signal, and because both arms are
    matched to the SAME reference they stay comparable to each other as well. It is strictly
    better than the old behaviour on both axes.

    Measured on the real corpora: 27,744 documents per arm survive the match, against an
    ai_cap of 20,000, so this costs nothing the experiment was going to use.
    """
    root = dcfg.get("ai_reference")
    if not root:
        return None
    if root == REFERENCE_HUMAN:
        if human_rows is None:
            raise RuntimeError(
                'data.ai_reference is "human" but no human corpus was passed to match '
                "against. This is a wiring bug, not a config error."
            )
        return human_rows
    from forge.training.data import load_examples

    return load_examples(ai_root=root)


def run(config: dict | str, smoke: bool = False, resume: str | None = None) -> dict:
    cfg_early = load(config) if isinstance(config, str) else config
    validate_config(cfg_early)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    from forge.modeling.encoder import ForgeConfig, build_model
    from forge.training.data import Collator, build_dataset, load_examples

    cfg = load(config) if isinstance(config, str) else config
    mcfg = load(cfg["model_config"])
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    paths = cfg.get("paths", {})

    exp_id = cfg["experiment"]["id"]
    out_dir = Path(paths.get("out", REPO_ROOT / "outputs")) / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(tcfg.get("seed", 42))

    fpr_budget = cfg.get("fpr_budget", 0.001)

    tok = AutoTokenizer.from_pretrained(mcfg["backbone"])
    limit = 200 if smoke else dcfg.get("limit")
    arm = validate_config(cfg)
    # ai_cap holds the AI budget equal across arms. Each arm's validator rejects at its
    # own rate, so generation yields different counts; without the cap the comparison
    # measures data volume as much as data strategy. Set the SAME value in every arm's
    # config, equal to the smallest arm's accepted count.
    examples = load_examples(
        human_root=paths.get("human"), ai_root=paths.get("ai") or paths.get("mirror"),
        mixed_root=paths.get("mixed"), limit=limit, expect_arm=arm,
        ai_cap=dcfg.get("ai_cap"),
        ai_reference=_reference_arm(
            dcfg,
            human_rows=(
                load_examples(human_root=paths.get("human"))
                if dcfg.get("ai_reference") == REFERENCE_HUMAN
                else None
            ),
        ),
        human_cap=dcfg.get("human_cap"),
    )
    if not examples:
        raise RuntimeError(
            f"no examples loaded from {paths}. Run `forge ingest` and `forge mirror` first."
        )

    feats = build_dataset(examples, tok, max_length=mcfg["max_length"],
                          stride=mcfg["window"]["stride"])
    by_split = {s: [f for f in feats if f["split"] == s] for s in ("train", "val", "test")}

    # Hard invariant, re-checked here rather than trusted: no group may span two splits.
    seen: dict[str, str] = {}
    for f in feats:
        prior = seen.setdefault(f["source_group_id"], f["split"])
        if prior != f["split"]:
            raise RuntimeError(
                f"group {f['source_group_id']} appears in both {prior} and {f['split']}. "
                "Training on this data would produce inflated metrics."
            )

    if not by_split["train"]:
        raise RuntimeError("no training windows after windowing")

    coll = Collator(tok, max_length=mcfg["max_length"])
    bs = tcfg["batch_size"]
    dl_train = DataLoader(by_split["train"], batch_size=bs, shuffle=True, collate_fn=coll,
                          num_workers=tcfg.get("num_workers", 2), drop_last=False)
    dl_val = DataLoader(by_split["val"] or by_split["train"], batch_size=bs * 2,
                        shuffle=False, collate_fn=coll)

    model = build_model(ForgeConfig(
        backbone=mcfg["backbone"], max_length=mcfg["max_length"],
        stride=mcfg["window"]["stride"],
        token_loss_weight=tcfg.get("token_loss_weight", 0.5),
    )).to(device)

    accum = tcfg.get("grad_accum", 1)
    steps_per_epoch = max(1, math.ceil(len(dl_train) / accum))
    # A SMOKE RUN THAT TAKES THREE STEPS TESTS ALMOST NOTHING. --smoke declared a budget of
    # 20 optimizer steps but ran a single epoch, and one epoch over 200 examples at batch 32
    # with grad_accum 2 is about seven batches, so THREE optimizer steps. The run reported a
    # val block whose AUROC matched a completely untrained model to four decimal places,
    # because that is what it was measuring. A smoke run exists to exercise the optimizer,
    # the scheduler, the checkpoint write and the eval path; three steps at the front of a
    # warmup barely exercises the first two.
    #
    # So take enough epochs to actually reach the step budget. The existing `stop` flag ends
    # the run the moment the budget is hit, so this cannot overshoot.
    epochs = max(1, math.ceil(20 / steps_per_epoch)) if smoke else tcfg["epochs"]
    total_steps = 20 if smoke else steps_per_epoch * epochs

    decay = [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n and "LayerNorm" not in n]
    nodecay = [p for n, p in model.named_parameters() if p.requires_grad and ("bias" in n or "LayerNorm" in n)]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.get("weight_decay", 0.01)},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=float(tcfg["learning_rate"]),
    )
    warmup = int(total_steps * tcfg.get("warmup_ratio", 0.06))
    sched = get_linear_schedule_with_warmup(optim, warmup, total_steps)

    precision = tcfg.get("precision", "bf16")
    use_amp = device == "cuda" and precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if precision == "bf16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "config requests bf16 but this GPU does not support it (T4 and older). "
            "DeBERTa-v3 in fp16 is known to produce NaN losses, so switch to fp32 or use "
            "an Ampere-or-newer GPU rather than silently downgrading to fp16."
        )
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and device == "cuda"))

    if tcfg.get("gradient_checkpointing"):
        enable_gradient_checkpointing(model.encoder)

    state = TrainState(dataset_version=dcfg.get("dataset_version", ""), code_commit=_code_commit())
    if resume:
        state = load_checkpoint(resume, model, optim, sched, scaler)

    # TRACKING, AND WHY IT RECORDS ITS OWN ABSENCE. This used to be a silent three-way
    # AND: wrong backend, smoke run, or no API key, and tracking simply did not happen with
    # nothing said about it. A run that was not tracked then looks identical to a run that
    # was, and "there is no W&B link for this experiment" becomes unanswerable after the
    # fact. The reason is now recorded in the run record, and the run URL with it, so the
    # README can link an artifact instead of a URL somebody retyped.
    wb = None
    tracking: dict[str, object] = {"backend": cfg.get("tracking", {}).get("backend")}
    if tracking["backend"] != "wandb":
        tracking["enabled"], tracking["reason"] = False, "tracking.backend is not wandb"
    elif smoke:
        tracking["enabled"], tracking["reason"] = False, "smoke run; not worth a tracked experiment"
    elif not os.getenv("WANDB_API_KEY"):
        tracking["enabled"], tracking["reason"] = False, "WANDB_API_KEY is not set"
    else:
        import wandb

        wb = wandb.init(project=cfg["tracking"].get("project", "forge"), name=exp_id,
                        config={"model": mcfg, "training": tcfg, "data": dcfg})
        tracking.update({
            "enabled": True,
            "project": cfg["tracking"].get("project", "forge"),
            "run_name": exp_id,
            "run_id": getattr(wb, "id", None),
            "run_url": getattr(wb, "url", None),
        })
    if not tracking["enabled"]:
        print(f"[tracking] disabled: {tracking['reason']}", flush=True)
    else:
        print(f"[tracking] wandb run: {tracking.get('run_url')}", flush=True)

    history = []
    t0 = time.time()
    stop = False
    for epoch in range(state.epoch, epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(dl_train):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(batch["input_ids"], batch["attention_mask"],
                            batch["doc_labels"], batch["token_labels"])
                loss = out["loss"] / accum
            if not torch.isfinite(loss):
                # THE MESSAGE USED TO NAME ONE CAUSE AND IT WAS THE WRONG ONE. It said
                # "usually fp16 overflow in disentangled attention; use bf16 or fp32" on
                # a run that was already bf16, which has fp32's exponent range and does
                # not overflow that way. The real cause was a batch whose token labels
                # were all ignore_index, reducing over zero elements. A confident wrong
                # diagnosis is worse than none: it sends the reader to the setting that
                # is already correct. The precision actually in use is reported, and the
                # fp16 hypothesis is offered only when fp16 is actually running.
                hint = (
                    "fp16 overflow in DeBERTa-v3's disentangled attention is the usual "
                    "cause at this precision; try bf16 or fp32."
                    if precision == "fp16"
                    else "precision is not the usual cause at this setting. Check the "
                         "batch: an all-ignore_index token-label tensor reduces over zero "
                         "elements and yields NaN, and a corrupt or empty example does "
                         "the same."
                )
                raise RuntimeError(
                    f"non-finite loss at step {state.global_step} (precision={precision}, "
                    f"amp={'on' if use_amp else 'off'}). {hint}"
                )
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()

            if (i + 1) % accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                state.global_step += 1

                if state.global_step % tcfg.get("log_every", 50) == 0:
                    rec = {"step": state.global_step, "loss": float(loss.item() * accum),
                           "lr": sched.get_last_lr()[0],
                           "examples_per_second": state.global_step * bs * accum / (time.time() - t0)}
                    history.append(rec)
                    print(f"step {rec['step']:>6}  loss {rec['loss']:.4f}  "
                          f"lr {rec['lr']:.2e}  {rec['examples_per_second']:.0f} ex/s", flush=True)
                    if wb:
                        wb.log(rec, step=state.global_step)

                if smoke and state.global_step >= total_steps:
                    stop = True
                    break
        state.epoch = epoch + 1

        val = evaluate_split(model, dl_val, device, fpr_budget)
        print(f"[epoch {state.epoch}] {json.dumps(val.as_dict())}", flush=True)
        if wb:
            wb.log({f"val/{k}": v for k, v in val.as_dict().items()}, step=state.global_step)

        # Budget is the constraint, FNR is the objective. See is_better_checkpoint.
        if is_better_checkpoint(val, state, fpr_budget):
            state.best_metric = val.fpr_at_budget
            state.best_fnr = val.fnr
            state.best_in_budget = val.fpr_at_budget <= fpr_budget
            state.best_step = state.global_step
            state.temperature = val.temperature
            state.threshold = val.threshold
            save_checkpoint(out_dir / "best.pt", model, optim, sched, scaler, state)
        save_checkpoint(out_dir / "last.pt", model, optim, sched, scaler, state)
        if stop:
            break

    val = evaluate_split(model, dl_val, device, fpr_budget)
    summary = {
        "experiment": exp_id,
        "smoke": smoke,
        "code_commit": state.code_commit,
        "dataset_version": state.dataset_version,
        "backbone": mcfg["backbone"],
        "device": device,
        "precision": precision,
        "arm": arm,
        "ai_source": paths.get("ai") or paths.get("mirror"),
        "n_windows": {k: len(v) for k, v in by_split.items()},
        "steps": state.global_step,
        "wall_seconds": round(time.time() - t0, 1),
        "fpr_budget": fpr_budget,
        "val": val.as_dict(),
        "history": history[-20:],
        "tracking": tracking,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wb:
        wb.finish()
    return summary
