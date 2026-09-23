"""What the stripping script must not get wrong.

The script exists to make a checkpoint small enough to serve. The dangerous failure is not
that it fails, it is that it SUCCEEDS on a file whose weights are no longer the weights the
evaluation measured, because every published number then describes a model nobody is
running. So the tests here are about identity first and size second:

  * the weights come back bit for bit,
  * the optimizer state is actually gone (a copy that kept it would still "work"),
  * the training state record survives, so the checkpoint can still be identified,
  * the source is never touched and never overwritten.

These build a two-tensor checkpoint rather than a real one. A 184M-parameter arm would make
the suite slow for no extra coverage: the property being tested does not depend on size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

torch = pytest.importorskip("torch")

from strip_checkpoint import SERVE_KEYS, fingerprint, strip  # noqa: E402

STATE = {"step": 1200, "epoch": 2, "best": 0.9791, "commit": "8e06099f"}


def _checkpoint(tmp_path: Path) -> Path:
    """A checkpoint shaped like save_checkpoint's, including the optimizer moments."""
    torch.manual_seed(0)
    weights = {
        # Big enough that pickle's framing is negligible. At 16x8 the container
        # overhead was a third of the file and the ratio below failed on a copy that
        # had in fact dropped everything it was supposed to.
        "encoder.weight": torch.randn(256, 128),
        "head.bias": torch.randn(64),
    }
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model": weights,
            # Two moment buffers per parameter, which is what makes the file big.
            "optimizer": {
                "state": {
                    index: {"exp_avg": torch.randn_like(t), "exp_avg_sq": torch.randn_like(t)}
                    for index, t in enumerate(weights.values())
                }
            },
            "scheduler": {"last_epoch": 2},
            "scaler": {"scale": 65536.0},
            "state": STATE,
        },
        path,
    )
    (tmp_path / "best.pt.json").write_text('{"step": 1200}\n')
    return path


def test_weights_survive_byte_for_byte(tmp_path):
    source = _checkpoint(tmp_path)
    before = fingerprint(torch.load(source, map_location="cpu", weights_only=False)["model"])

    strip(source, tmp_path / "best-model-only.pt")

    after = fingerprint(
        torch.load(tmp_path / "best-model-only.pt", map_location="cpu", weights_only=False)["model"]
    )
    assert after == before


def test_optimizer_state_is_gone_and_only_serve_keys_remain(tmp_path):
    source = _checkpoint(tmp_path)

    strip(source, tmp_path / "small.pt")

    served = torch.load(tmp_path / "small.pt", map_location="cpu", weights_only=False)
    assert sorted(served) == sorted(SERVE_KEYS)
    assert served["state"] == STATE


def test_the_stripped_file_is_smaller(tmp_path):
    source = _checkpoint(tmp_path)

    record = strip(source, tmp_path / "small.pt")

    # Not an arbitrary ratio: AdamW holds two moments per parameter, so dropping them
    # should remove roughly two thirds. Asserting merely "smaller" would pass on a copy
    # that dropped only the scaler.
    assert record["destination_bytes"] < record["source_bytes"] * 0.5
    assert record["dropped"] == ["optimizer", "scheduler", "scaler"]


def test_the_source_is_left_alone(tmp_path):
    source = _checkpoint(tmp_path)
    before = source.read_bytes()

    strip(source, tmp_path / "small.pt")

    assert source.read_bytes() == before


def test_refuses_to_overwrite_the_source(tmp_path):
    source = _checkpoint(tmp_path)

    with pytest.raises(SystemExit, match="refusing to overwrite"):
        strip(source, source)


def test_refuses_a_file_that_is_not_a_training_checkpoint(tmp_path):
    path = tmp_path / "weights.pt"
    torch.save({"encoder.weight": torch.zeros(2)}, path)

    with pytest.raises(SystemExit, match="not a FORGE training checkpoint"):
        strip(path, tmp_path / "small.pt")


def test_the_sidecar_follows_the_copy(tmp_path):
    source = _checkpoint(tmp_path)

    strip(source, tmp_path / "small.pt")

    # Without this, a served checkpoint cannot be identified without loading 700 MB.
    assert (tmp_path / "small.pt.json").read_text() == '{"step": 1200}\n'
