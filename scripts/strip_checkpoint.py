"""Strip the optimizer state out of a training checkpoint so it can be served.

WHY THIS EXISTS. `forge.training.train.save_checkpoint` writes five keys:

    {"model", "optimizer", "scheduler", "scaler", "state"}

That is correct for resuming a run and wrong for serving one. AdamW keeps two float32
moment buffers per parameter, so a 184M-parameter DeBERTa-v3-base costs roughly 735 MB of
weights and roughly 1.4 GB of optimizer moments, and each `best.pt` on disk is 2.1 GB. The
Streamlit Community Cloud page has to download that file and then hold torch, the arm and
the page in about 2.7 GB of RAM, and it does not survive it: the app reports

    Loading the arms and scoring. The first run fetches 1.5 GB of weights.  ->  CONNECTING
    ->  Oh no. Error running app.

Inference reads `ck["model"]` and nothing else (see `forge.inference.scorer`). So two
thirds of every byte served is state no reader ever uses.

WHAT THIS DOES NOT DO. It does not cast, quantise or prune. The tensors written are the
same float32 tensors bit for bit, and the script proves that before it reports success by
re-reading what it wrote and comparing every tensor against the source by SHA-256 of its
raw bytes. A checkpoint that serves different numbers than the one the evaluation measured
would invalidate every published figure, so a mismatch here is a hard failure, not a
warning. `state` is carried across unchanged so the sidecar and the checkpoint keep
agreeing about which run produced the weights.

The source file is never modified and never deleted. Deleting it would throw away the
ability to resume training, which is the one thing the optimizer state is for.

    python scripts/strip_checkpoint.py outputs/forge_min_baseline/best.pt
    python scripts/strip_checkpoint.py outputs/*/best.pt --suffix -model-only

Requires torch, so it runs on the machine that has the checkpoints, not in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Keys that inference needs. Everything else in the checkpoint exists to resume a run.
SERVE_KEYS = ("model", "state")
DROPPED_KEYS = ("optimizer", "scheduler", "scaler")


def _digest(tensor) -> str:
    """SHA-256 over a tensor's raw bytes, on CPU and contiguous.

    Comparing with `torch.equal` would answer a weaker question: it is true for two
    tensors that hold equal values in different dtypes on different devices. What has to
    be true here is that the served bytes ARE the measured bytes, so the comparison is
    over bytes, and dtype and shape are compared separately and named in the digest so a
    reshape cannot collide with itself.
    """
    import torch

    # .view(torch.uint8) rather than .numpy(), because numpy has no bfloat16 and a
    # checkpoint saved in it would make this crash instead of comparing.
    raw = tensor.detach().to("cpu").contiguous().flatten().view(torch.uint8)
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(header + raw.numpy().tobytes()).hexdigest()


def fingerprint(state_dict: dict) -> dict[str, str]:
    return {key: _digest(value) for key, value in sorted(state_dict.items())}


def strip(source: Path, destination: Path) -> dict:
    """Write a serve-only copy of `source` at `destination` and verify it.

    Returns a record of what happened, suitable for printing or for a test to assert on.
    Raises rather than returning a partial result, because a half-written checkpoint that
    looks finished is worse than no checkpoint.
    """
    import torch

    if destination.resolve() == source.resolve():
        raise SystemExit(f"refusing to overwrite the source checkpoint: {source}")

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    missing = [key for key in SERVE_KEYS if key not in checkpoint]
    if missing:
        raise SystemExit(
            f"{source} is not a FORGE training checkpoint: missing {missing}. "
            f"It has {sorted(checkpoint)}."
        )

    served = {key: checkpoint[key] for key in SERVE_KEYS}
    before = fingerprint(checkpoint["model"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(served, destination)

    # VERIFY BY RE-READING. Trusting torch.save to have written what it was handed is the
    # assumption that would make a silent corruption look like a success.
    reloaded = torch.load(destination, map_location="cpu", weights_only=False)
    after = fingerprint(reloaded["model"])
    if after != before:
        differing = sorted(set(before) ^ set(after)) or [
            key for key in before if before[key] != after.get(key)
        ]
        destination.unlink(missing_ok=True)
        raise SystemExit(
            f"the stripped checkpoint does not match the source on {len(differing)} "
            f"tensor(s), first {differing[:5]}. Nothing was kept."
        )
    if reloaded["state"] != checkpoint["state"]:
        destination.unlink(missing_ok=True)
        raise SystemExit("the stripped checkpoint lost its training state record.")

    # The sidecar identifies a checkpoint without loading it, so the copy gets one too.
    sidecar = Path(str(source) + ".json")
    if sidecar.exists():
        Path(str(destination) + ".json").write_text(sidecar.read_text())
    else:
        Path(str(destination) + ".json").write_text(
            json.dumps(checkpoint["state"], indent=2, default=str) + "\n"
        )

    dropped = [key for key in DROPPED_KEYS if checkpoint.get(key) is not None]
    return {
        "source": str(source),
        "destination": str(destination),
        "source_bytes": source.stat().st_size,
        "destination_bytes": destination.stat().st_size,
        "tensors": len(before),
        "dropped": dropped,
        "verified": True,
    }


def _human(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "--suffix",
        default="-model-only",
        help="appended to the stem of each output file (default: -model-only)",
    )
    args = parser.parse_args(argv)

    for path in args.checkpoints:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        destination = path.with_name(f"{path.stem}{args.suffix}{path.suffix}")
        record = strip(path, destination)
        saved = record["source_bytes"] - record["destination_bytes"]
        print(
            f"{record['source']}\n"
            f"  -> {record['destination']}\n"
            f"     {_human(record['source_bytes'])} -> "
            f"{_human(record['destination_bytes'])} "
            f"(saved {_human(saved)}, "
            f"{saved / record['source_bytes']:.0%})\n"
            f"     {record['tensors']} tensors verified byte-identical; "
            f"dropped {', '.join(record['dropped']) or 'nothing'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
