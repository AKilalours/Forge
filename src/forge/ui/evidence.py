"""The engineering panel: measured numbers read from the artifacts that produced them.

WHY THIS FILE EXISTS. Every performance number this project publishes lives in a JSON file
under `reports/experiments/`, written by the script that measured it. The README quotes
those files and `scripts/check_readme_claims.py` fails the build when a quoted number and
its artifact disagree. The interface had no such link: it showed a verdict and nothing
about the system behind it, and the obvious way to fix that would have been to type the
numbers into the page.

THAT WOULD HAVE BEEN THE SAME BUG THE GATE EXISTS TO CATCH. A number typed into a page is a
copy, and copies drift; this project has already shipped a badge that said 918 when the
real count was 914, and a profiling figure of 1.17% that was not in the artifact at all.
So this module reads the artifacts at render time and renders nothing it did not read.
There are no literal measurements in this file. Grep it for a decimal point.

A PANEL IS A NUMBER PLUS THE REASON IT MIGHT MISLEAD. Each artifact carries its own
`invariant`, `scope`, `caveat` or `what_this_does_not_show` text, written when the
measurement was taken. `Panel.caveats` is populated from those fields and never from this
file, and `build_panels()` refuses to emit a panel that has rows and no caveat, because a
throughput table without its conditions is how 108.6% scaling efficiency got published
once already.

A MISSING ARTIFACT IS A STATE, NOT A BLANK. If a file is absent the panel still renders,
with status "missing" and no rows. The page then says the measurement does not exist, which
is true, instead of quietly showing the last thing somebody remembered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = REPO_ROOT / "reports" / "experiments"

MEASURED = "measured"
MISSING = "missing"
UNREADABLE = "unreadable"


@dataclass(frozen=True)
class Panel:
    """One measurement, its table, and the conditions under which it holds."""

    key: str
    title: str
    sources: tuple[str, ...]
    status: str
    headline: str
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    caveats: tuple[str, ...] = ()
    facts: dict = field(default_factory=dict)

    @property
    def has_rows(self) -> bool:
        return bool(self.rows)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load(path: Path) -> tuple[dict | None, str]:
    """Read one artifact. Returns (payload, status); payload is None unless MEASURED."""
    if not path.exists():
        return None, MISSING
    try:
        return json.loads(path.read_text()), MEASURED
    except (json.JSONDecodeError, OSError):
        # A corrupt artifact is not a missing one, and the page should not present it as
        # "never measured" when the truth is "measured and the file is damaged".
        return None, UNREADABLE


def _absent(key: str, title: str, sources: tuple[str, ...], status: str) -> Panel:
    word = {
        MISSING: "Not measured. The artifact this panel reads does not exist in this checkout.",
        UNREADABLE: "The artifact exists but could not be parsed, so nothing is shown.",
    }[status]
    return Panel(key=key, title=title, sources=sources, status=status, headline=word)


def _num(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _pct(value, digits: int = 1) -> str:
    return "-" if value is None else f"{value * 100:.{digits}f}%"


# --------------------------------------------------------------------------- panels

def scaling_panel(device_dir: str, strategy: str = "fsdp", *, root: Path | None = None) -> Panel:
    base = (root or EXPERIMENTS) / "scaling" / device_dir
    name = "scaling_summary.json" if strategy == "fsdp" else f"scaling_summary_{strategy}.json"
    path = base / name
    key = f"scaling:{device_dir}:{strategy}"
    title = f"Multi-GPU training throughput ({strategy.upper()})"
    data, status = load(path)
    if data is None:
        return _absent(key, title, (_rel(path),), status)

    rows = data.get("rows", [])
    top = rows[-1] if rows else {}
    headline = (
        f"{data.get('device_name', 'unknown device')}, global batch "
        f"{data.get('global_batch_size')} held fixed across world sizes. "
        f"At {top.get('world_size')} GPUs: {_num(top.get('speedup'))}x speedup, "
        f"{_pct(top.get('scaling_efficiency'))} of linear."
    )
    return Panel(
        key=key,
        title=f"{title} on {data.get('device_name', 'unknown device')}",
        sources=(_rel(path),),
        status=MEASURED,
        headline=headline,
        columns=("GPUs", "examples/s", "tokens/s", "s/step", "peak mem GiB", "speedup", "efficiency"),
        rows=tuple(
            (
                _num(r.get("world_size")),
                _num(r.get("examples_per_second")),
                _num(r.get("tokens_per_second")),
                _num(r.get("seconds_per_step"), 4),
                _num(r.get("peak_memory_gib")),
                f"{_num(r.get('speedup'))}x",
                _pct(r.get("scaling_efficiency")),
            )
            for r in rows
        ),
        caveats=tuple(t for t in (data.get("invariant"), data.get("caveat")) if t),
        facts={"device_name": data.get("device_name"), "strategy": data.get("strategy")},
    )


def checkpointing_panel(*, root: Path | None = None) -> Panel:
    path = (root or EXPERIMENTS) / "profile" / "train_step_comparison.json"
    key, title = "checkpointing", "Gradient checkpointing: what it costs and what it buys"
    data, status = load(path)
    if data is None:
        return _absent(key, title, (_rel(path),), status)

    cost = data.get("gradient_checkpointing_cost", {})
    headline = (
        f"Trading {_pct(cost.get('step_time_overhead'), 2)} step time for "
        f"{_num(cost.get('memory_saved_gib'))} GiB of activation memory, "
        f"measured on {data.get('device_name', 'unknown device')}."
    )
    arms = data.get("arms", {})
    return Panel(
        key=key,
        title=title,
        sources=(_rel(path),),
        status=MEASURED,
        headline=headline,
        columns=("arm", "median step ms", "p90 step ms", "peak mem GiB", "steps timed"),
        rows=tuple(
            (
                name,
                _num(arm.get("median_step_ms")),
                _num(arm.get("p90_step_ms")),
                _num(arm.get("peak_memory_gib")),
                _num(arm.get("steps_timed")),
            )
            for name, arm in arms.items()
        ),
        caveats=tuple(t for t in (data.get("scope"),) if t),
        facts={"device_name": data.get("device_name")},
    )


def serving_panel(device: str, *, root: Path | None = None) -> Panel:
    path = (root or EXPERIMENTS) / "inference" / f"{device}_latency.json"
    key, title = f"serving:{device}", f"Serving latency, batch sweep ({device.upper()})"
    data, status = load(path)
    if data is None:
        return _absent(key, title, (_rel(path),), status)

    peak = data.get("peak_throughput", {})
    headline = (
        f"{data.get('device_name', 'unknown device')}, {data.get('precision')}, "
        f"seq {data.get('seq_len')}. Best throughput at batch "
        f"{peak.get('batch_windows')}: {_num(peak.get('windows_per_second'))} windows/s, "
        f"{_num(peak.get('speedup_over_batch_1'))}x over batch 1."
    )
    caveats = [data.get("scope")] if data.get("scope") else []
    caveats += [str(v) for v in (data.get("caveats") or {}).values()]
    return Panel(
        key=key,
        title=title,
        sources=(_rel(path),),
        status=MEASURED,
        headline=headline,
        columns=("batch windows", "median ms", "p95 ms", "windows/s", "ms per window"),
        rows=tuple(
            (
                _num(r.get("batch_windows")),
                _num(r.get("median_ms")),
                _num(r.get("p95_ms")),
                _num(r.get("windows_per_second")),
                _num(r.get("ms_per_window")),
            )
            for r in data.get("batch_sweep", [])
        ),
        caveats=tuple(caveats),
        facts={"device_name": data.get("device_name")},
    )


def threads_panel(*, root: Path | None = None) -> Panel:
    path = (root or EXPERIMENTS) / "inference" / "cpu_threads.json"
    key, title = "threads", "Intra-op threads on the serving path"
    data, status = load(path)
    if data is None:
        return _absent(key, title, (_rel(path),), status)

    rows = data.get("rows", [])
    top = max(rows, key=lambda r: r.get("speedup_over_one_thread") or 0, default={})
    repro = data.get("reproducibility", {})
    headline = (
        f"{data.get('host_cpu_count')} cores available. Best observed "
        f"{_num(top.get('speedup_over_one_thread'))}x at {_num(top.get('threads'))} threads, "
        f"which is sublinear: threads do not buy their own number back."
    )
    caveats = [t for t in (data.get("note"),) if t]
    if repro.get("runs_taken"):
        caveats.append(
            f"Taken over {repro['runs_taken']} runs; the run-to-run spread is why this is "
            "reported as a range rather than a single figure."
        )
    return Panel(
        key=key,
        title=title,
        sources=(_rel(path),),
        status=MEASURED,
        headline=headline,
        columns=("threads", "peak batch", "windows/s", "speedup", "efficiency"),
        rows=tuple(
            (
                _num(r.get("threads")),
                _num(r.get("peak_batch")),
                _num(r.get("windows_per_second")),
                f"{_num(r.get('speedup_over_one_thread'))}x",
                _pct(r.get("efficiency")),
            )
            for r in rows
        ),
        caveats=tuple(caveats),
    )


def spark_panel(*, root: Path | None = None) -> Panel:
    path = (root or EXPERIMENTS) / "spark" / "spark_summary.json"
    key, title = "spark", "Hard-negative mining, partitioned scan"
    data, status = load(path)
    if data is None:
        return _absent(key, title, (_rel(path),), status)

    rows = data.get("rows", [])
    top = max(rows, key=lambda r: r.get("speedup") or 0, default={})
    headline = (
        f"{data.get('documents_scanned')} documents per configuration, held constant. "
        f"Best compute speedup {_num(top.get('speedup'))}x at "
        f"{_num(top.get('partitions'))} partitions."
    )
    return Panel(
        key=key,
        title=title,
        sources=(_rel(path),),
        status=MEASURED,
        headline=headline,
        columns=("partitions", "docs/s compute", "docs/s wall", "wall s", "startup s",
                 "speedup"),
        rows=tuple(
            (
                _num(r.get("partitions")),
                _num(r.get("documents_per_second_compute")),
                _num(r.get("documents_per_second_wall")),
                _num(r.get("wall_seconds")),
                _num(r.get("startup_seconds")),
                f"{_num(r.get('speedup'))}x",
            )
            for r in rows
        ),
        caveats=tuple(t for t in (data.get("invariant"), data.get("caveat"),
                                  data.get("speedup_semantics")) if t),
    )


def beam_panel(mode: str, *, root: Path | None = None) -> Panel:
    """The same scan on a second runner. One panel per execution mode, because threads in
    one process and separate worker processes pay different startup costs for identical
    throughput, and averaging them would hide the only thing the port measured."""
    path = (root or EXPERIMENTS) / "beam" / mode / "beam_summary.json"
    key, title = f"beam-{mode}", f"The same scan on Beam ({mode})"
    data, status = load(path)
    if data is None:
        return _absent(key, title, (_rel(path),), status)

    rows = data.get("rows", [])
    pool = data.get("pool") or {}
    headline = (
        f"{data.get('documents_scanned')} documents per configuration from the same "
        f"{pool.get('n_files')} shards (pool {pool.get('fingerprint')}), so this table and "
        f"the Spark one are timed over identical work."
    )
    return Panel(
        key=key,
        title=title,
        sources=(_rel(path),),
        status=MEASURED,
        headline=headline,
        columns=("partitions", "docs/s compute", "docs/s wall", "startup s",
                 "model loads", "worker pids", "speedup"),
        rows=tuple(
            (
                _num(r.get("partitions")),
                _num(r.get("documents_per_second_compute")),
                _num(r.get("documents_per_second_wall")),
                _num(r.get("startup_seconds")),
                _num(r.get("model_loads")),
                _num(r.get("distinct_worker_pids")),
                f"{_num(r.get('speedup'))}x",
            )
            for r in rows
        ),
        caveats=tuple(t for t in (data.get("invariant"), data.get("caveat"),
                                  data.get("speedup_semantics")) if t),
    )


def adversarial_panel(experiment: str, *, root: Path | None = None) -> Panel:
    """Where the detector fails, and what preprocessing does about it.

    Only the attacks that move the needle are rendered. Most of the cells are zero in every
    condition, and a table of zeros buries the handful of rows that matter.
    """
    path = (root or EXPERIMENTS) / f"adversarial_{experiment}.json"
    key, title = f"adversarial-{experiment}", f"Adversarial evasion, {experiment}"
    data, status = load(path)
    if data is None:
        return _absent(key, title, (_rel(path),), status)

    conditions = data.get("conditions_scored") or ["raw", "normalised"]

    def fnr(result: dict, condition: str) -> float | None:
        """Read one condition's miss rate, from either artifact generation.

        THE FALLBACK, and the reason fnr_raw and fnr_preprocessed were kept when the
        conditions block was added. An artifact written before that change has no
        `conditions` key, so a reader that only knows the new shape finds nothing, renders
        an empty table, and still reports the panel as measured. A stale artifact must
        degrade to fewer columns, never to a confident blank.
        """
        block = (result.get("conditions") or {}).get(condition)
        if block is not None:
            return block.get("fnr")
        legacy = {"raw": "fnr_raw", "normalised": "fnr_preprocessed"}.get(condition)
        return result.get(legacy) if legacy else None

    if not any("conditions" in r for r in data.get("results", [])):
        conditions = ["raw", "normalised"]

    live = [r for r in data.get("results", [])
            if any(fnr(r, c) for c in conditions)]
    live.sort(key=lambda r: -max((fnr(r, c) or 0) for c in conditions))

    worst = live[0] if live else {}
    headline = (
        f"{data.get('n_ai_documents')} AI test documents at the deployed threshold "
        f"{_num(data.get('threshold'), 6)}. Worst evasion: {worst.get('attack')} at "
        f"severity {worst.get('severity')} reaching "
        f"{_num(fnr(worst, 'normalised'), 3)} "
        f"miss rate after production normalisation."
    ) if live else "No attack moved the miss rate in any condition."

    return Panel(
        key=key,
        title=title,
        sources=(_rel(path),),
        status=MEASURED,
        headline=headline,
        columns=("attack", "severity", *conditions),
        rows=tuple(
            (
                str(r.get("attack")),
                _num(r.get("severity")),
                *(_num(fnr(r, c), 3) for c in conditions),
            )
            for r in live
        ),
        # FROM THE ARTIFACT ONLY. The first version of this panel appended a sentence
        # written here, about the miss rate being AI-only. It was true and it was the wrong
        # place: a caveat composed by the page can say anything, including something the
        # run did not do. test_every_caveat_appears_verbatim_in_the_artifact_it_came_from
        # caught it. The sentence moved into the note the CLI writes, where it is part of
        # the record rather than part of the presentation.
        caveats=tuple(t for t in (data.get("note"),) if t),
    )


def ray_panel(*, root: Path | None = None) -> Panel:
    path = (root or EXPERIMENTS) / "ray" / "ray_ws2.json"
    key, title = "ray", "Launcher swap: Ray Train in place of torchrun"
    data, status = load(path)
    if data is None:
        return _absent(key, title, (_rel(path),), status)

    headline = data.get("what_this_shows") or "No claim was recorded for this run."
    return Panel(
        key=key,
        title=title,
        sources=(_rel(path),),
        status=MEASURED,
        headline=headline,
        columns=("launcher", "workers", "backend", "GPU", "global batch", "grad accum", "steps"),
        rows=(
            (
                str(data.get("launcher")),
                _num(data.get("workers")),
                str(data.get("backend")),
                "yes" if data.get("use_gpu") else "no",
                _num(data.get("global_batch_size")),
                _num(data.get("grad_accum")),
                _num(data.get("steps")),
            ),
        ),
        caveats=tuple(t for t in (data.get("what_this_does_not_show"),) if t),
    )


# ------------------------------------------------------------------------- assembly

def build_panels(*, root: Path | None = None) -> list[Panel]:
    """Every panel, in the order the page shows them, present or absent.

    THE GUARD. A panel that carries rows must carry at least one caveat taken from its own
    artifact. This is not stylistic. Each of these tables has a condition that changes what
    the number means: a fixed global batch, a single-host Spark job, a toy model under Ray.
    Published without it, every one of them overstates the result.
    """
    panels = [
        scaling_panel("nvidia-a100-sxm4-80gb", "fsdp", root=root),
        scaling_panel("nvidia-l40s", "fsdp", root=root),
        scaling_panel("nvidia-l40s", "deepspeed", root=root),
        checkpointing_panel(root=root),
        serving_panel("cpu", root=root),
        serving_panel("cuda", root=root),
        threads_panel(root=root),
        spark_panel(root=root),
        beam_panel("multi_processing", root=root),
        beam_panel("multi_threading", root=root),
        adversarial_panel("forge_min_baseline", root=root),
        adversarial_panel("forge_min_mirror", root=root),
        ray_panel(root=root),
    ]
    for panel in panels:
        if panel.has_rows and not panel.caveats:
            raise AssertionError(
                f"panel {panel.key!r} renders a table with no caveat from its artifact; "
                "add the condition to the artifact rather than removing this check"
            )
        _refuse_dead_columns(panel)
    return panels


def _refuse_dead_columns(panel: Panel) -> None:
    """A column whose every cell is a placeholder is a column the artifact no longer has.

    THE SILENT REGRESSION THIS EXISTS TO CATCH, and it had already happened. The partition
    sweeps used to publish an `efficiency` field. It was removed from the artifacts when it
    turned out to be meaningless under a constant thread budget, and nothing told this
    module: _num and _pct turn a missing key into "-", so the panel kept rendering an
    efficiency column of three dashes. No error, no failing test, just a column that looks
    like a measurement which happened to be unavailable.

    The caveat guard above catches a table that overstates its result. This catches a table
    that quietly stopped having one.
    """
    if not panel.has_rows:
        return
    for index, column in enumerate(panel.columns):
        values = {row[index] for row in panel.rows}
        if values <= {"-", "", "-x"}:
            raise AssertionError(
                f"panel {panel.key!r} column {column!r} is empty in every row, which means "
                f"its artifact no longer carries that field. Remove the column or restore "
                f"the field; do not leave a placeholder where a number used to be."
            )


def measured(panels: list[Panel]) -> list[Panel]:
    return [p for p in panels if p.status == MEASURED]
