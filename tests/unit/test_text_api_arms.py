"""Scoring a subset of arms, which is what keeps the deployed page inside its memory.

The Streamlit host allows about 2.7 GB. Two float32 arms at 701 MB each plus torch plus
the page does not fit, and the failure is not an exception: the container is killed
mid-request and the visitor sees "Oh no. Error running app." with no reason given. So the
page asks for one arm. These tests pin the two things that could undo that quietly: the
default must stay both arms, so the FastAPI reference page is unaffected, and a typo in an
arm name must raise rather than silently score nothing and render an empty page.
"""

from __future__ import annotations

import inspect

import pytest

from forge.inference.text_api import analyse


def test_the_default_is_still_every_arm():
    assert inspect.signature(analyse).parameters["arms"].default is None


def test_an_unknown_arm_is_refused_before_anything_is_loaded():
    # Without this, a typo yields zero scored arms and a page that renders as though every
    # arm were merely unavailable, which reads like a memory problem rather than a typo.
    with pytest.raises(ValueError, match="unknown arm"):
        analyse("some text", arms=("mirrorr",))


def test_the_refusal_names_the_arms_that_exist():
    from forge.inference.scorer import ARMS

    with pytest.raises(ValueError) as error:
        analyse("some text", arms=("nope",))
    assert all(arm in str(error.value) for arm in ARMS)
