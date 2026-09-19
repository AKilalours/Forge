"""What the engineering panel is allowed to show.

The panel exists because the alternative was typing measured numbers into a page, and a
typed number is a copy of a measurement rather than the measurement. These tests hold the
three properties that make the panel safer than the alternative:

  it reads the artifacts rather than carrying values of its own,
  it never shows a table without the condition recorded alongside it,
  and a missing artifact renders as an absence rather than as a blank or a stale value.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re

import pytest

from forge.ui import evidence
from forge.ui.evidence import MEASURED, MISSING, UNREADABLE, Panel, build_panels

MODULE = pathlib.Path(evidence.__file__)


def test_the_module_contains_no_measurement_of_its_own():
    """The defect this panel was written to avoid: a number living in the page.

    A decimal literal in this file would be a value that no artifact produced, which is
    exactly the shape of the badge that said 918 when the suite had 914 tests. Version
    numbers in the docstring are not decimals with digits on both sides of the point, and
    a rounding precision like `digits: int = 2` is an integer, so neither trips this.
    """
    body = MODULE.read_text()
    code = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )
    # Strip the module docstring, which is prose about the file and not executable.
    code = code.split('"""', 2)[-1]
    offenders = re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", code)
    assert not offenders, f"the panel carries its own numbers: {offenders}"


def test_every_panel_with_a_table_carries_a_caveat_from_its_artifact():
    for panel in build_panels():
        if panel.rows:
            assert panel.caveats, f"{panel.key} renders a table with no condition attached"
            for caveat in panel.caveats:
                assert caveat.strip(), f"{panel.key} has an empty caveat"


def test_every_caveat_appears_verbatim_in_the_artifact_it_came_from():
    """A caveat this module composed itself would be an opinion, not a record.

    The one exception is the reproducibility sentence on the thread panel, which is
    assembled from a count in the artifact; it is allowed because the count is read.
    """
    root = evidence.EXPERIMENTS
    for panel in build_panels():
        if panel.status != MEASURED:
            continue
        blob = " ".join(
            (root.parent.parent / src).read_text() for src in panel.sources
        )
        for caveat in panel.caveats:
            if caveat.startswith("Taken over "):
                continue
            needle = json.dumps(caveat)[1:-1]
            assert needle in blob, f"{panel.key}: caveat not found in {panel.sources}"


def test_a_missing_artifact_is_an_absence_and_not_a_blank(tmp_path):
    panel = evidence.spark_panel(root=tmp_path)
    assert panel.status == MISSING
    assert panel.rows == ()
    assert "Not measured" in panel.headline
    assert panel.sources, "an absent panel must still say which file it looked for"


def test_a_corrupt_artifact_is_distinguished_from_a_missing_one(tmp_path):
    (tmp_path / "spark").mkdir()
    (tmp_path / "spark" / "spark_summary.json").write_text("{not json")
    panel = evidence.spark_panel(root=tmp_path)
    assert panel.status == UNREADABLE
    assert panel.rows == ()
    assert "could not be parsed" in panel.headline


def test_the_guard_fires_when_an_artifact_loses_its_condition(tmp_path, monkeypatch):
    """Delete the invariant from a real artifact and the page must refuse to render it.

    This is the regression that matters. A scaling table without "global batch held fixed"
    is the table that reported 108.6% efficiency across two different GPUs, and the fix
    belongs in the artifact, not in a page that shows the number anyway.
    """
    src = evidence.EXPERIMENTS / "spark" / "spark_summary.json"
    data = json.loads(src.read_text())
    # Every key that can supply a condition, not just the two this artifact had when the
    # test was written. speedup_semantics was added later and is a legitimate condition, so
    # leaving it in would let the artifact lose its invariant while the guard stayed quiet,
    # which is the exact failure this test exists to prevent.
    for condition_key in ("invariant", "caveat", "speedup_semantics"):
        data.pop(condition_key, None)
    target = tmp_path / "spark"
    target.mkdir()
    (target / "spark_summary.json").write_text(json.dumps(data))
    panel = evidence.spark_panel(root=tmp_path)
    assert panel.rows and not panel.caveats
    monkeypatch.setattr(evidence, "spark_panel", lambda **kw: panel)
    with pytest.raises(AssertionError, match="no caveat"):
        build_panels(root=tmp_path)


def test_the_committed_artifacts_all_render():
    panels = build_panels()
    assert len(evidence.measured(panels)) == len(panels), (
        "an artifact this page depends on is missing from the checkout: "
        + ", ".join(p.key for p in panels if p.status != MEASURED)
    )


def test_the_page_wires_the_panel_in():
    page = pathlib.Path(evidence.REPO_ROOT / "streamlit_app.py").read_text()
    assert "engineering_tab" in page
    assert "build_panels" in page, "the tab must read the artifacts, not restate them"
    tab = page[page.index("def engineering_tab"):page.index("# ------", page.index("def engineering_tab"))]
    assert not re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", tab), (
        "the tab body carries a number of its own; it must render only what it read"
    )


def test_panel_is_immutable():
    """Panels are handed to a renderer; a renderer that edits one has changed a record."""
    panel = Panel(key="k", title="t", sources=("s",), status=MISSING, headline="h")
    with pytest.raises(dataclasses.FrozenInstanceError):
        panel.headline = "something else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# THE COLUMN THAT OUTLIVED ITS FIELD.
#
# The partition sweeps published an `efficiency` field. It was removed from the artifacts
# when it turned out to be meaningless under a constant thread budget, and nothing told
# this module: _num and _pct turn a missing key into "-", so the Spark panel kept rendering
# an efficiency column of three dashes. No error, no failing test, just a column that looked
# like a measurement which happened to be unavailable.
#
# The caveat guard catches a table that overstates its result. This catches one that
# quietly stopped having a result at all.
# ---------------------------------------------------------------------------


def test_a_column_that_is_empty_in_every_row_is_refused() -> None:
    from forge.ui.evidence import Panel, _refuse_dead_columns

    panel = Panel(
        key="example", title="t", sources=("a.json",), status="measured", headline="h",
        columns=("partitions", "gone"),
        rows=(("1", "-"), ("2", "-")),
        caveats=("a condition",),
    )
    with pytest.raises(AssertionError, match="no longer carries that field"):
        _refuse_dead_columns(panel)


def test_a_column_with_one_real_value_is_allowed() -> None:
    """A genuinely missing cell is normal. A column that is missing in EVERY row is the
    signal, because that is what a removed field looks like."""
    from forge.ui.evidence import Panel, _refuse_dead_columns

    panel = Panel(
        key="example", title="t", sources=("a.json",), status="measured", headline="h",
        columns=("partitions", "sometimes"),
        rows=(("1", "-"), ("2", "0.5")),
        caveats=("a condition",),
    )
    _refuse_dead_columns(panel)


def test_the_spark_panel_no_longer_advertises_efficiency() -> None:
    """Under a constant thread budget the ideal speedup is 1.0, so speedup divided by
    partitions is not parallel efficiency. The artifact dropped it; the panel must not
    still promise it."""
    from forge.ui.evidence import spark_panel

    assert "efficiency" not in spark_panel().columns


def test_both_runners_have_a_panel() -> None:
    """The Beam port is a committed result. A page that shows only the Spark sweep claims
    less than the repository has measured."""
    from forge.ui.evidence import build_panels

    keys = {p.key for p in build_panels()}
    assert "spark" in keys
    assert "beam-multi_processing" in keys
    assert "beam-multi_threading" in keys


def test_the_adversarial_panel_reads_an_artifact_from_either_generation(tmp_path) -> None:
    """THE STALE-SCHEMA FALLBACK.

    fnr_raw and fnr_preprocessed were deliberately kept when the per-condition block was
    added. An artifact written before that change has no `conditions` key, and a reader
    that only knows the new shape finds nothing, renders an empty table, and still reports
    the panel as measured. A stale artifact must degrade to fewer columns, never to a
    confident blank.
    """
    import json

    from forge.ui.evidence import adversarial_panel

    (tmp_path / "adversarial_old.json").write_text(json.dumps({
        "n_ai_documents": 500, "threshold": 0.9,
        "note": "a condition from the artifact",
        "results": [{"attack": "case_perturb", "severity": 0.1,
                     "fnr_raw": 0.9, "fnr_preprocessed": 0.9}],
    }))
    panel = adversarial_panel("old", root=tmp_path)
    assert panel.columns == ("attack", "severity", "raw", "normalised")
    assert panel.rows, "an older artifact must still render its rows"
    assert panel.rows[0][0] == "case_perturb"


def test_the_adversarial_panel_takes_its_caveat_from_the_artifact() -> None:
    """A miss rate without the human cost reads as a free defence, so the table must carry
    that condition. It must come from the run's own note: a caveat composed by the page can
    say anything, including something the run did not do."""

    from forge.ui.evidence import EXPERIMENTS, adversarial_panel

    panel = adversarial_panel("forge_min_baseline")
    if not panel.has_rows:
        pytest.skip("no adversarial artifact in this checkout")
    blob = (EXPERIMENTS / "adversarial_forge_min_baseline.json").read_text()
    assert panel.caveats
    for caveat in panel.caveats:
        assert caveat in blob, "the page must not compose a caveat of its own"


def test_the_page_builder_carries_no_measurement_of_its_own() -> None:
    """Same rule as forge.ui.evidence, for the same reason.

    docs/index.html is generated, so a number can only reach it from an artifact. That
    holds exactly as long as the builder never contains one. A decimal literal in that
    file is a figure nobody can trace, sitting on the page a visitor reads first.
    """
    import re
    from pathlib import Path

    builder = Path(__file__).resolve().parents[2] / "scripts" / "build_evidence_page.py"
    source = builder.read_text()
    # Format precision (".2f", digits: int = 3) and the chart geometry are not
    # measurements; scale ticks are the values the axis is drawn to.
    stripped = re.sub(r'(?m)^\s*#.*$', '', source)
    stripped = re.sub(r'"""(?:.|\n)*?"""', '', stripped)
    stripped = re.sub(r"y_ticks=\[[^\]]*\]", "", stripped)
    stripped = re.sub(r"for t in \([^)]*\)", "", stripped)
    offenders = [m for m in re.findall(r"\d+\.\d+", stripped)
                 if m not in {"0.62", "1.0", "0.0"}]
    assert not offenders, f"the builder carries its own numbers: {offenders}"
