"""Score a trained arm against a held-out benchmark it has never seen.

WHY THIS IS THE RUN THAT MATTERS. The in-distribution comparison came back saturated and
underpowered: AUROC 0.99997 on both arms, and a two-proportion test on the FNR gap gave
p = 0.217. That is what happens when a detector is asked about text from the same four
generator families it trained on. RAID, MAGE and HC3 carry generators the model has never
seen, which is the question the project actually asks.

TWO DECISIONS THAT DECIDE WHETHER THE NUMBERS MEAN ANYTHING.

1. THE THRESHOLD COMES FROM VALIDATION, NOT FROM THE BENCHMARK. Re-fitting the operating
   point on the test set is the standard way to make an out-of-distribution number look
   good: you are then reporting the best case the benchmark allows rather than what the
   deployed detector would actually do. The primary numbers here use the threshold each arm
   fit on its own validation split, which is the threshold a user would get. The re-fit
   value is reported too, clearly labelled, because the gap between them is itself
   informative: a large gap means calibration did not transfer.

2. WINDOW SCORES ARE AVERAGED, NOT MAXED. A long document becomes several windows. Taking
   the maximum answers "does any window look generated", which finds more AI and raises the
   false-positive rate on long human documents, because more windows means more chances to
   trip. For a detector whose entire product constraint is not accusing human writers, that
   is the wrong trade. The mean is reported as primary and the max alongside it.

Deterministic subsampling by document id, so two runs of the same command score the same
documents and a difference between arms is not a difference in sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

from forge.common.config import load
from forge.evaluation import metrics as M
from forge.evaluation.benchmarks import BenchmarkDoc
from forge.training.data import Collator, RawExample, build_dataset

BENCHMARKS = ("hc3", "mage", "raid")


def _rank(doc_id: str) -> str:
    return hashlib.sha256(f"ood-sample:{doc_id}".encode()).hexdigest()


def subsample(docs: list[BenchmarkDoc], limit: int) -> list[BenchmarkDoc]:
    """Deterministic, and stratified by label so a cap cannot skew the class balance."""
    if limit <= 0 or limit >= len(docs):
        return docs
    human = [d for d in docs if d.label == 0]
    ai = [d for d in docs if d.label == 1]
    share = limit // 2
    keep = (
        sorted(human, key=lambda d: _rank(d.doc_id))[:share]
        + sorted(ai, key=lambda d: _rank(d.doc_id))[: limit - share]
    )
    chosen = {d.doc_id for d in keep}
    return [d for d in docs if d.doc_id in chosen]


def load_benchmark(name: str, limit: int) -> list[BenchmarkDoc]:
    from forge.evaluation import benchmarks as B

    loader = {"hc3": B.load_hc3, "mage": B.load_mage, "raid": B.load_raid}[name]
    docs = loader()
    if not docs:
        raise RuntimeError(f"{name} returned no documents")
    return subsample(docs, limit)


def score_documents(model, tokenizer, docs: list[BenchmarkDoc], mcfg: dict, device: str):
    """Return per-document mean and max window probabilities, plus labels."""
    import torch
    from torch.utils.data import DataLoader

    examples = [
        RawExample(
            doc_id=d.doc_id, source_group_id=d.doc_id, split="test",
            text=d.text, label=d.label, spans=None,
            domain=getattr(d, "domain", "unknown"),
            generator_family=getattr(d, "generator", "unknown"),
        )
        for d in docs
        if d.text and d.text.strip()
    ]
    feats = build_dataset(
        examples, tokenizer,
        max_length=mcfg["max_length"], stride=mcfg["window"]["stride"],
    )
    if not feats:
        raise RuntimeError("windowing produced no features")

    loader = DataLoader(
        feats, batch_size=64, shuffle=False,
        collate_fn=Collator(tokenizer, max_length=mcfg["max_length"]),
    )
    model.eval()
    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            p = torch.softmax(out["doc_logits"].float(), dim=-1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())

    # Windows come back in the order build_dataset produced them; group by document.
    by_doc: dict[str, list[float]] = {}
    for feature, probability in zip(feats, probs, strict=True):
        by_doc.setdefault(feature["doc_id"], []).append(probability)

    label_of = {e.doc_id: e.label for e in examples}
    doc_ids = [d for d in by_doc if d in label_of]
    mean = np.array([float(np.mean(by_doc[d])) for d in doc_ids])
    maximum = np.array([float(np.max(by_doc[d])) for d in doc_ids])
    labels = np.array([label_of[d] for d in doc_ids])
    return labels, mean, maximum, len(feats)


def evaluate(arm: str, benchmark: str, limit: int, out_dir: pathlib.Path) -> dict:
    import torch

    from forge.modeling.encoder import ForgeConfig, build_model
    from forge.training.train import load_checkpoint

    config_path = f"configs/training/{arm}_minimal.yaml"
    cfg = load(config_path)
    mcfg = load(cfg["model_config"])
    experiment = cfg["experiment"]["id"]
    checkpoint = pathlib.Path(cfg["paths"].get("out", "outputs")) / experiment / "best.pt"
    summary_path = pathlib.Path(cfg["paths"].get("out", "outputs")) / experiment / "summary.json"
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}; train this arm first")

    summary = json.loads(summary_path.read_text())
    val_threshold = float(summary["val"]["threshold"])
    fpr_budget = float(cfg.get("fpr_budget", 0.001))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(ForgeConfig(
        backbone=mcfg["backbone"], max_length=mcfg["max_length"],
        stride=mcfg["window"]["stride"],
    )).to(device)
    load_checkpoint(checkpoint, model)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(mcfg["backbone"])

    started = time.perf_counter()
    docs = load_benchmark(benchmark, limit)
    labels, mean, maximum, n_windows = score_documents(model, tokenizer, docs, mcfg, device)

    if benchmark == "mage":
        # The dataset card and the original paper disagree on label polarity. Assert it
        # rather than discovering an inverted result and rationalising it afterwards.
        from forge.evaluation.benchmarks import assert_mage_polarity

        assert_mage_polarity(
            [d for d in docs if d.text and d.text.strip()][: len(mean)], list(mean)
        )

    refit = M.threshold_at_fpr(labels, mean, fpr_budget)
    result = {
        "arm": arm,
        "benchmark": benchmark,
        "experiment": experiment,
        "code_commit": summary.get("code_commit"),
        "dataset_version": summary.get("dataset_version"),
        "n_documents": int(len(labels)),
        "n_human": int((labels == 0).sum()),
        "n_ai": int((labels == 1).sum()),
        "n_windows": int(n_windows),
        "fpr_budget": fpr_budget,
        "deployed": {
            "threshold": val_threshold,
            "source": "fit on this arm's validation split; the operating point a user gets",
            "fpr": float(M.false_positive_rate(labels, mean, val_threshold)),
            "fnr": float(M.false_negative_rate(labels, mean, val_threshold)),
        },
        "refit_on_benchmark": {
            "threshold": float(refit),
            "source": "re-fit on this benchmark. OPTIMISTIC: not available at deployment",
            "fpr": float(M.false_positive_rate(labels, mean, refit)),
            "fnr": float(M.false_negative_rate(labels, mean, refit)),
        },
        "auroc_mean_pooled": float(M.auroc(labels, mean)),
        "auroc_max_pooled": float(M.auroc(labels, maximum)),
        "ece": float(M.expected_calibration_error(labels, mean)),
        "aggregation": "mean over windows; max reported alongside because it inflates FPR",
        "seconds": round(time.perf_counter() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"ood_{arm}_{benchmark}.json").write_text(json.dumps(result, indent=2) + "\n")

    # Per-document scores, so the two arms can be compared with a PAIRED test.
    #
    # Comparing two AUROCs from summary numbers alone forces an unpaired approximation,
    # which is wrong here: both arms score the SAME documents, so their errors are
    # correlated, and ignoring that inflates the standard error. A paired bootstrap over
    # these arrays is the honest test and costs nothing once the scores are on disk.
    #
    # Written as .npz rather than into the JSON because it is thousands of floats, and a
    # results file a human reads should stay readable.
    np.savez_compressed(
        out_dir / f"ood_scores_{arm}_{benchmark}.npz",
        labels=labels, mean=mean, max=maximum,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["baseline", "mirror"])
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--limit", type=int, default=4000)
    parser.add_argument("--out", default="reports/experiments")
    args = parser.parse_args()

    result = evaluate(args.arm, args.benchmark, args.limit, pathlib.Path(args.out))
    d, r = result["deployed"], result["refit_on_benchmark"]
    print(
        f"{args.arm:>9} on {args.benchmark:<5} "
        f"n={result['n_documents']:>5} "
        f"AUROC={result['auroc_mean_pooled']:.4f}  "
        f"deployed FPR={d['fpr']:.4f} FNR={d['fnr']:.4f}  "
        f"refit FPR={r['fpr']:.4f} FNR={r['fnr']:.4f}  "
        f"({result['seconds']}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
