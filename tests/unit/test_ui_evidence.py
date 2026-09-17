"""What the engineering panel is allowed to show.

The panel exists because the alternative was typing measured numbers into a page, and a
typed number is a copy of a measurement rather than the measurement. These tests hold the
three properties that make the panel safer than the alternative:

  it reads the artifacts rather than carrying values of its own,
  it never shows a table without the condition recorded alongside it,
  and a missing artifact renders as an absence rather than as a blank or a stale value.
"""

from __future__ import annotations

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
    data.pop("invariant", None)
    data.pop("caveat", None)
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
    with pytest.raises(Exception):
        panel.headline = "something else"  # type: ignore[misc]
