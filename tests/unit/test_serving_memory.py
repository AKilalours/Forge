"""The two choices that keep the deployed page inside 2.7 GB.

Neither is testable end to end without torch and a 735 MB checkpoint, and neither fails
loudly when it regresses: the page still works locally, on a laptop with 16 GB, and dies
only on the host. So what is pinned here is that the serving path asks for them at all.

The failure being prevented is specific. Streamlit Community Cloud kills the container
mid-request when it runs out of memory, and the visitor sees "Oh no. Error running app."
with no reason given. That is what this app did for weeks.
"""

from __future__ import annotations

import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORER = (ROOT / "src" / "forge" / "inference" / "scorer.py").read_text()


def test_serving_builds_the_model_without_fetching_pretrained_weights():
    # from_pretrained downloads 371 MB and allocates a full encoder, every tensor of
    # which load_checkpoint overwrites on the next line.
    assert "pretrained=False" in SCORER


def test_serving_memory_maps_the_checkpoint():
    # Otherwise torch.load holds 735 MB of weights while the model holds its own copy.
    assert "mmap=True" in SCORER


def test_both_options_exist_where_the_serving_path_calls_them():
    from forge.modeling.encoder import build_model
    from forge.training.train import load_checkpoint

    assert inspect.signature(build_model).parameters["pretrained"].default is True
    assert inspect.signature(load_checkpoint).parameters["mmap"].default is False


def test_training_is_unaffected():
    """Defaults stay put: fine-tuning from random init would be a different experiment,
    and a resume restores optimizer state that mapping does not help with."""
    from forge.modeling.encoder import build_model

    source = inspect.getsource(build_model)
    assert "if pretrained" in source
