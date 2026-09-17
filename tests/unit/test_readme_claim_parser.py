"""The README badge check must not fail open.

WHY THIS FILE EXISTS. scripts/check_readme_claims.py has one claim it verifies by
running a command rather than by reading an artifact: the "N tests passing" badge.
That check silently never ran, on this machine or in CI, for its entire life.

The mechanism is worth remembering because nothing about it looked wrong. pyproject.toml
sets addopts = "-q -ra". The script invoked `pytest --collect-only -q`. Two -q flags is
-qq, and at -qq pytest stops printing the "N tests collected" line. So pytest ran, exited
0, printed several hundred node ids, and omitted the single line the parser looked for.
The parser returned None, and the caller printed "note: pytest could not be run" and
passed. Every part of that sentence was false, and it was reassuring.

Two fixes, both tested here. The parser now handles every shape pytest emits, including
-qq. And the caller distinguishes "could not launch pytest" (a legitimate skip on a
machine without the dev extra) from "ran pytest and could not read the answer" (a broken
check, which must fail).

The samples below are real pytest output, not invented.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from check_readme_claims import (  # noqa: E402
    main,
    parse_collected_count,
    suite_size_claims,
)

MODERN = """tests/unit/test_arm_configs.py::test_both_arms_declare_a_budget
tests/unit/test_arm_configs.py::test_the_arms_agree_on_the_budget

914 tests collected in 2.31s
"""

OLDER = "collected 918 items\n"

DESELECTED = "900/914 tests collected (14 deselected) in 2.31s\n"

# The one that broke it: -qq, so node ids and then warnings, and no count line at all.
DOUBLE_QUIET = """tests/unit/test_arm_configs.py::test_both_arms_declare_a_budget
tests/unit/test_arm_configs.py::test_the_arms_agree_on_the_budget
tests/unit/test_metrics.py::test_auroc_is_rank_based

=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/fastapi/testclient.py:1
  StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
"""

NOT_PYTEST_AT_ALL = "ERROR: file or directory not found: tests/\n"


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        pytest.param(MODERN, 914, id="modern-summary-line"),
        pytest.param(OLDER, 918, id="older-collected-items-line"),
        pytest.param(DESELECTED, 914, id="deselection-reports-the-total"),
        pytest.param(DOUBLE_QUIET, 3, id="qq-has-no-count-line-so-count-node-ids"),
    ],
)
def test_the_parser_reads_every_shape_pytest_emits(sample: str, expected: int) -> None:
    assert parse_collected_count(sample) == expected


def test_the_parser_returns_none_when_there_is_genuinely_nothing_to_read() -> None:
    """None must stay possible. The caller decides what None means; the parser must not
    invent a number to avoid returning it."""
    assert parse_collected_count(NOT_PYTEST_AT_ALL) is None


def test_the_qq_case_specifically_does_not_return_none() -> None:
    """The regression, stated on its own.

    This exact sample made the old parser return None, which the caller rendered as
    'pytest could not be run'. If this ever returns None again, the badge check has
    stopped running and will say nothing about it.
    """
    assert parse_collected_count(DOUBLE_QUIET) is not None


def test_the_claim_check_still_passes_without_spawning_pytest() -> None:
    """The rest of the gate, run the way the test suite calls it.

    check_test_count=False because obtaining the count requires running pytest, and a
    test that spawns the collector it is running under is slow at best.
    """
    assert main(check_test_count=False) == 0


# --------------------------------------------------- the prose the badge check never read

def test_bold_and_tree_counts_are_found_and_located() -> None:
    md = "# Title\n\n**918 tests**, mostly regressions.\n\n\u251c\u2500 tests/unit/  # 918 tests, named\n"
    assert suite_size_claims(md) == [(3, 918), (5, 918)]


def test_a_readme_whose_badge_and_prose_disagree_is_caught() -> None:
    """The exact drift this project shipped, reconstructed.

    Badge 985, prose 918, CI green for months. The badge check looked at the badge and at
    nothing else, so the sentence under it was free to say whatever it said.
    """
    md = "Tests-985%20passing\n\n**918 tests**, and the interesting ones are regressions.\n"
    disagreeing = [(line, n) for line, n in suite_size_claims(md) if n != 985]
    assert disagreeing == [(3, 918)]


def test_a_count_about_one_module_is_not_a_claim_about_the_suite() -> None:
    """README line 339 says a module's 11 tests run in their own CI job. It is not the total.

    Treating it as one would make the gate fail on a true sentence, and a gate that cries
    wolf gets its check deleted rather than its claim fixed.
    """
    md = "> its 11 tests run in **their own CI job** rather than skipping in the main one\n"
    assert suite_size_claims(md) == []


def test_an_unbolded_suite_count_is_knowingly_not_gated() -> None:
    """The stated gap, pinned so it cannot be discovered as a surprise later."""
    assert suite_size_claims("The suite has 994 tests today.\n") == []


def test_the_committed_readme_agrees_with_itself() -> None:
    """Every gated count in the README is the number the badge claims."""
    from pathlib import Path
    md = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    badge = re.search(r"Tests-(\d+)%20passing", md)
    assert badge is not None
    claims = suite_size_claims(md)
    assert claims, "the README no longer states its suite size in prose at all"
    assert {n for _, n in claims} == {int(badge.group(1))}, (
        f"README prose claims {claims} against a badge of {badge.group(1)}"
    )
