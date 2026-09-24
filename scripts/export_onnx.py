"""Export an arm to ONNX, prove it scores the same, and measure what that bought.

    python scripts/export_onnx.py --arm mirror
    python scripts/export_onnx.py --arm mirror --quantize --threads 1 2 4 8 10

WHY ONNX RUNTIME AND NOT vLLM. FORGE-Base is a bidirectional encoder over a fixed
512-token window with no KV cache, so continuous batching and paged attention have nothing
to work with. What this serving path actually pays for is per-window encoder compute on
CPU, which is what ONNX Runtime's graph fusions and int8 kernels target. Choosing the
framework that fits the model rather than the one in the job description is the point.

PARITY BEFORE SPEED, AND THE PARITY THAT MATTERS IS THE VERDICT. Two models can agree to
four decimal places on average and still disagree about a document sitting near the
threshold, and the deployed threshold here is 0.998252 on one arm: a 1e-3 shift is the
difference between "AI" and "human" for anything in that neighbourhood. So this script
reports both the numeric difference and the number of documents whose VERDICT changes,
and it refuses to call an export good on the numeric agreement alone.

INT8 IS MEASURED, NOT ASSUMED. Dynamic quantisation is where the real CPU speedup is, and
it genuinely changes the scores. A quantised arm is only usable if the verdict flips are
zero on a real sample, and this script tells you rather than deciding for you. If they are
not zero, the honest options are to keep float32 or to refit the threshold on the
quantised model and re-run the whole evaluation, which is a bigger job than it sounds.

WHAT THE BENCHMARK COMPARES AGAINST. reports/experiments/inference/cpu_latency_b8_t*.json,
produced by scripts/benchmark_inference.py at batch 8 over 1, 2, 4, 8 and 10 threads. This
script writes onnx_latency_b8_t*.json in the same shape, over the same windows, so the two
are comparable rather than merely adjacent.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports" / "experiments" / "inference"
OPSET = 17


def _windows(arm, texts: list[str], batch: int):
    """Real tokenised windows, through the same code path serving uses."""
    from torch.utils.data import DataLoader

    from forge.training.data import Collator, RawExample, build_dataset

    examples = [
        RawExample(doc_id=f"d{i}", source_group_id=f"d{i}", split="test", text=text,
                   label=0, spans=None, domain="unknown", generator_family="unknown")
        for i, text in enumerate(texts)
    ]
    feats = build_dataset(examples, arm.tokenizer,
                          max_length=arm.mcfg["max_length"],
                          stride=arm.mcfg["window"]["stride"])
    loader = DataLoader(feats, batch_size=batch, shuffle=False,
                        collate_fn=Collator(arm.tokenizer, max_length=arm.mcfg["max_length"]))
    return list(loader)


def _human_texts(root: Path, limit: int) -> list[str]:
    """Documents from the training corpus: real inputs, not lorem ipsum.

    ParquetFile rather than read_table, for the hive-discovery collision documented in
    training/data.py.
    """
    import pyarrow.parquet as pq

    from forge.ingestion.writer import PARTITION_GLOB

    texts: list[str] = []
    for path in sorted(root.glob(PARTITION_GLOB)):
        for row in pq.ParquetFile(path).read(columns=["text"]).to_pylist():
            texts.append(row["text"])
            if len(texts) >= limit:
                return texts
    if not texts:
        raise SystemExit(f"no documents under {root}; parity needs real text to score")
    return texts


class _DocOnly:
    """Wraps the arm's model so the exported graph has tensor inputs and outputs.

    The training forward takes optional label tensors and returns a dict including a loss.
    ONNX export needs positional tensors in and tensors out, and serving only reads
    doc_logits. Exporting the training signature would put a loss branch in the graph that
    can never run and two inputs that are always None.
    """

    def __new__(cls, model):
        import torch

        class Wrapped(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = model

            def forward(self, input_ids, attention_mask):
                return self.inner(input_ids=input_ids, attention_mask=attention_mask)["doc_logits"]

        return Wrapped().eval()


def export(arm, path: Path) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    batch = _windows(arm, ["The quick brown fox jumps over the lazy dog. " * 80], 1)[0]
    with torch.no_grad():
        torch.onnx.export(
            _DocOnly(arm.model),
            (batch["input_ids"], batch["attention_mask"]),
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=["doc_logits"],
            # Batch AND sequence are dynamic. A graph frozen at 512 would silently refuse
            # the short final window of every document.
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "doc_logits": {0: "batch"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
        )
    # A 184M-parameter model is not a 145 KB file. The dynamo exporter writes weights to
    # a sibling .data file, so the graph alone is tiny and an export that produced NO
    # weights at all looks exactly the same from the graph's size. Both are checked.
    weights = path.with_suffix(".onnx.data")
    total = path.stat().st_size + (weights.stat().st_size if weights.exists() else 0)
    if total < 100_000_000:
        raise SystemExit(
            f"the export produced {total / 1e6:.1f} MB across {path.name} and its external "
            f"data. A DeBERTa-v3-base arm is roughly 700 MB, so this graph is missing its "
            f"weights and anything measured on it would be meaningless."
        )
    return path


def quantize(src: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = src.with_suffix(".int8.onnx")
    # Weights only, per channel. Activations stay float, which is what keeps the accuracy
    # cost small enough to be worth measuring at all on a transformer encoder.
    quantize_dynamic(str(src), str(out), weight_type=QuantType.QInt8)
    return out


def session(path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _probabilities_torch(arm, batches) -> list[float]:
    import torch

    probs: list[float] = []
    with torch.no_grad():
        for batch in batches:
            out = arm.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs += torch.softmax(out["doc_logits"].float(), dim=-1)[:, 1].tolist()
    return probs


def _probabilities_onnx(sess, batches) -> list[float]:
    import numpy as np

    probs: list[float] = []
    for batch in batches:
        logits = sess.run(["doc_logits"], {
            "input_ids": batch["input_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
        })[0].astype("float64")
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs += (exp[:, 1] / exp.sum(axis=-1)).tolist()
    return probs


def compare(reference: list[float], candidate: list[float], threshold: float) -> dict:
    """Numeric agreement AND verdict agreement. Pure, so it can be tested without torch.

    `verdict_flips` is the number that decides whether an export is usable.
    `near_threshold_windows` is how many windows were close enough to flip at all: zero
    flips out of a sample with nothing near the threshold is not evidence of anything,
    and reporting the two together is what stops it being read as if it were.
    """
    if len(reference) != len(candidate):
        raise SystemExit(
            f"the two runtimes returned different window counts, {len(reference)} and "
            f"{len(candidate)}. Comparing them elementwise would invent an agreement."
        )
    deltas = [abs(a - b) for a, b in zip(reference, candidate)]
    return {
        "windows": len(deltas),
        "max_abs_delta": max(deltas) if deltas else 0.0,
        "mean_abs_delta": statistics.fmean(deltas) if deltas else 0.0,
        "threshold": threshold,
        "verdict_flips": sum(
            (a >= threshold) != (b >= threshold) for a, b in zip(reference, candidate)
        ),
        "near_threshold_windows": sum(abs(p - threshold) < 1e-3 for p in reference),
    }


def parity(arm, sess, batches, threshold: float) -> dict:
    return compare(_probabilities_torch(arm, batches),
                   _probabilities_onnx(sess, batches), threshold)


def benchmark(sess, batches, repeats: int) -> dict:
    windows = sum(len(b["input_ids"]) for b in batches)
    timings = []
    _probabilities_onnx(sess, batches[:1])          # warm up the graph, not the measurement
    for _ in range(repeats):
        started = time.perf_counter()
        _probabilities_onnx(sess, batches)
        timings.append((time.perf_counter() - started) * 1000)
    median = statistics.median(timings)
    return {
        "batch_windows": len(batches[0]["input_ids"]),
        "median_ms": round(median, 2),
        "p95_ms": round(sorted(timings)[max(0, int(0.95 * len(timings)) - 1)], 2),
        "windows_per_second": round(windows / (median / 1000), 2),
        "ms_per_window": round(median / windows, 2),
        "samples": repeats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", default="mirror")
    parser.add_argument("--docs", type=int, default=8, help="documents for parity and timing")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 10])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--corpus", type=Path, default=REPO / "data" / "silver")
    args = parser.parse_args(argv)

    from forge.inference.scorer import ARMS, load_arm

    if args.arm not in ARMS:
        raise SystemExit(f"unknown arm {args.arm!r}, expected one of {ARMS}")

    arm = load_arm(args.arm)
    threshold = arm.policy.threshold
    batches = _windows(arm, _human_texts(args.corpus, args.docs), args.batch)
    print(f"{len(batches)} batches, {sum(len(b['input_ids']) for b in batches)} windows, "
          f"threshold {threshold}", flush=True)

    onnx_path = REPO / "outputs" / arm.experiment / "model.onnx"
    export(arm, onnx_path)
    weights = onnx_path.with_suffix(".onnx.data")
    print(f"exported {onnx_path} (graph {onnx_path.stat().st_size / 1e6:.1f} MB"
          + (f" + weights {weights.stat().st_size / 1e6:.0f} MB)" if weights.exists() else ")"),
          flush=True)

    variants = {"float32": onnx_path}
    if args.quantize:
        # NOT FATAL. The float32 export is the deliverable; int8 is the optimisation on
        # top of it. The first run died here, inside onnxruntime's shape inference, and
        # took the float32 benchmark down with it: a run that had already produced a
        # valid model reported nothing at all.
        try:
            variants["int8"] = quantize(onnx_path)
        except Exception as error:                  # noqa: BLE001 - reported, not hidden
            print(f"int8 quantisation failed and is being skipped: "
                  f"{type(error).__name__}: {error}", flush=True)
            print("  float32 results below are unaffected.", flush=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "arm": args.arm,
        "experiment": arm.experiment,
        "opset": OPSET,
        "device_name": f"{platform.machine()} ({platform.system()})",
        "backbone": arm.mcfg["backbone"],
        "seq_len": arm.mcfg["max_length"],
        "scope": "Model forward over pre-tokenised windows only. Tokenisation, windowing "
                 "and HTTP overhead are NOT included, so a real request costs this plus "
                 "those. Same scope as cpu_latency_b8_t*.json, so the two compare.",
        "variants": {},
    }

    for name, path in variants.items():
        sess = session(path, max(args.threads))
        checks = parity(arm, sess, batches, threshold)
        print(f"{name}: max delta {checks['max_abs_delta']:.2e}, "
              f"verdict flips {checks['verdict_flips']}", flush=True)
        sweep = {}
        for threads in args.threads:
            result = benchmark(session(path, threads), batches, args.repeats)
            sweep[str(threads)] = result
            path_out = REPORTS / f"onnx_{name}_b{args.batch}_t{threads}.json"
            path_out.write_text(json.dumps(
                {**{k: v for k, v in report.items() if k != "variants"},
                 "precision": name, "torch_threads": threads,
                 "batch_sweep": [result]}, indent=2) + "\n")
            print(f"  {threads} threads: {result['windows_per_second']} windows/s", flush=True)
        report["variants"][name] = {
            "file": str(path.relative_to(REPO)),
            "bytes": path.stat().st_size,
            "parity": checks,
            "threads": sweep,
        }

    summary = REPORTS / f"onnx_summary_{args.arm}.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {summary}")

    for name, variant in report["variants"].items():
        if variant["parity"]["verdict_flips"]:
            print(f"WARNING: {name} changes {variant['parity']['verdict_flips']} verdicts. "
                  f"It must not be deployed against a threshold fitted on float32 torch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
