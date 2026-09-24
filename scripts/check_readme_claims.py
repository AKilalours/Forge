"""Fail if a published number in the README is not what the committed artifact says.

WHY THIS EXISTS. Three times in this project's life a figure was computed, written
somewhere a reader would see it, and then never checked against the thing that produced
it. Reconciling them by hand works exactly once, on the day someone remembers to do it.
A README is edited far more often than a result is regenerated, so the drift is one-way
and silent: the artifacts stay right and the prose slowly stops matching them.

So the reconciliation is a test. It reads the tables out of README.md, reads the JSON in
reports/experiments/, and compares them at the precision the README chose to display. A
rounded 0.885 against a stored 0.885148 passes. A 0.885 against a stored 0.9 does not.

Run: python scripts/check_readme_claims.py
Exits non-zero on the first mismatch, listing every one it found.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "experiments"
BENCHMARKS = ("hc3", "raid", "mage")
ARMS = {"A": "baseline", "B": "mirror"}


def load(name: str) -> dict:
    """name may include a subdirectory, e.g. "scaling/scaling_summary.json"."""
    return json.loads((REPORTS / name).read_text())


def summary(arm_key: str) -> dict:
    """The training run's own record, which is where the in-distribution table came from.

    Read from reports/experiments/, not outputs/. outputs/ is gitignored because it also
    holds model weights, so on a clean checkout it does not exist, and a check that reads
    from there passes on the author's laptop and crashes in CI. That is the same shape as
    publishing a number whose artifact was never committed, which is what this script
    exists to prevent.
    """
    return json.loads((REPORTS / f"indist_{ 'baseline' if arm_key == 'A' else 'mirror' }.json").read_text())



def _uninstallable_test_gates() -> set[str]:
    """Import names that test modules refuse to run without and this environment lacks.

    Reuses the same scan as tests/unit/test_dependency_declaration.py: every module-level
    pytest.importorskip in tests/. If any of those cannot be imported here, the collected
    count is short and must not be compared against the badge.
    """
    import importlib.util

    pattern = re.compile(
        r"^(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?_?pytest\.importorskip\(\s*['\"]([^'\"]+)['\"]"
    )
    names: set[str] = set()
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        for line in path.read_text(encoding="utf-8").splitlines():
            m = pattern.match(line)
            if m:
                names.add(m.group(1).split(".")[0])
    missing = set()
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.add(name)
        except (ImportError, ValueError):
            missing.add(name)
    return missing


def parse_collected_count(text: str) -> int | None:
    """Pull the test count out of `pytest --collect-only` output.

    Separated from the subprocess call so it can be tested against real pytest output
    without running pytest. See tests/unit/test_readme_claim_parser.py.

    pytest has printed this line three different ways across versions, and at -qq it
    does not print it at all, so the node-id count is the fallback. Returning None here
    means the caller must report a broken check, never a passing one.
    """
    for pattern in (
        r"(\d+)\s+tests?\s+collected",
        r"collected\s+(\d+)\s+items?",
        r"(\d+)/\d+\s+tests?\s+collected",
    ):
        m = re.search(pattern, text)
        if m:
            return int(m.group(1))

    # Fallback: count the node ids themselves. One line per collected test.
    node_ids = {
        ln.strip() for ln in text.splitlines()
        if "::" in ln and not ln.startswith((" ", "\t", "=", "-")) and "warning" not in ln.lower()
    }
    return len(node_ids) or None


def suite_size_claims(md: str) -> list[tuple[int, int]]:
    """Every README claim about the size of the whole suite, as (line, count).

    WHY THIS IS A FUNCTION AND NOT A REGEX INLINE IN main(). The badge was gated and the
    prose was not, so the prose drifted: this README carried a badge reading 985 beside two
    sentences reading 918 and nothing failed, because the only check ever written looked at
    the badge. Pulling the scan out here means it can be tested against text that has the
    drift in it, rather than only against a README that currently happens to agree.

    WHAT COUNTS AS A CLAIM ABOUT THE SUITE. Not every "N tests" in the README is one: line
    339 says a module's "11 tests" run in their own CI job, which is a true statement about
    eleven tests and has nothing to do with the total. Guessing which is which from the
    number would be a heuristic, and heuristics that almost work are the whole subject of
    this file. So the marking is explicit instead:

        **994 tests**                         bold, anywhere
        tests/unit/    # 994 tests, ...       on a line naming the test directory

    A count written any other way is NOT gated. That is a real gap and it is stated here
    rather than hidden: if you write a sentence claiming the suite size, bold the number,
    or it will be free to rot exactly the way 918 did.
    """
    claims: list[tuple[int, int]] = []
    for lineno, line in enumerate(md.splitlines(), start=1):
        for m in re.finditer(r"\*\*(\d+) tests\*\*", line):
            claims.append((lineno, int(m.group(1))))
        if "tests/unit/" in line:
            for m in re.finditer(r"(\d+) tests\b", line):
                claims.append((lineno, int(m.group(1))))
    return claims


def collected_test_count() -> tuple[int | None, str]:
    """Ask pytest how many tests exist, for the badge that claims a number.

    Returns (count, reason). reason distinguishes two things the first version of this
    function collapsed into one None:

      "unavailable"  pytest could not be launched. A machine without the dev extra is
                     allowed to skip this check.
      "unparsed"     pytest ran and the count could not be recovered. That is a broken
                     check, not a missing dependency, and the caller must fail on it.

    -o addopts= is load-bearing. pyproject.toml sets addopts = "-q -ra", so passing our
    own -q made it -qq, and at -qq pytest suppresses the "N tests collected" line
    entirely. pytest exited 0, printed no count, and the old code reported "pytest could
    not be run" on a machine where pytest works fine. The badge was therefore never
    checked, here or in CI. Clearing addopts makes this call independent of whatever the
    project config happens to say today.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts="],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
    except Exception as exc:
        return None, f"unavailable: pytest could not be launched ({exc!r})"

    out = proc.stdout + proc.stderr
    count = parse_collected_count(out)

    # A COUNT FROM AN INCOMPLETE ENVIRONMENT IS ALSO A LIE, and it exits 0.
    #
    # Module-level pytest.importorskip makes a whole file contribute zero tests when its
    # dependency is absent, and pytest reports success. So a machine without torch or
    # pillow collects a few hundred fewer tests than a developer machine and says nothing
    # is wrong. This function previously only rejected non-zero exits, so it happily
    # reported 916 for a 962-test suite on a container missing the optional extras.
    #
    # That is the third variation of the same defect in this one function. First it could
    # not tell "could not launch pytest" from "ran it and could not read the answer". Then
    # it could not tell "collected everything" from "collected whatever imported" after a
    # collection error. Now: "collected everything" from "collected everything that was
    # INSTALLED". Each fix addressed the case in front of it rather than the shape, which
    # is: a count is only meaningful when the environment that produced it is complete.
    missing = _uninstallable_test_gates()
    if missing:
        return None, (
            "unavailable: this environment cannot import "
            f"{', '.join(sorted(missing))}, so the test modules gated on them contribute "
            "no tests and any count is short by however many they hold"
        )

    # A collection ERROR makes the count a lie rather than a number. pytest still prints
    # "N tests collected" alongside "M errors during collection", and N is only the
    # modules that happened to import. Reporting it would compare the badge against
    # whatever a broken environment managed to load, and on a machine missing the project
    # that is a small number that looks like a real answer. Exit 0 is the only trustworthy
    # collection.
    if proc.returncode != 0:
        detail = "collection errors" if count is not None else "pytest reported a problem"
        return None, (
            f"unavailable: pytest exited {proc.returncode} ({detail}), so any count it "
            "printed reflects what imported, not what exists"
        )

    if count is not None:
        return count, "ok"
    if proc.returncode == 4 or "no tests ran" in out.lower():
        return None, "unavailable: pytest found no tests to collect"
    tail = "\n".join(out.strip().splitlines()[-6:])
    return None, f"unparsed: pytest exited {proc.returncode} and printed no count. Last lines:\n{tail}"


def table_rows(md: str, header_fragment: str) -> dict[str, list[str]]:
    """Return {row label: [cells]} for the markdown table whose header contains the
    fragment. Bold markers are stripped: **0.885** and 0.885 are the same claim, the
    asterisks are emphasis for a reader, not part of the number."""
    lines = md.splitlines()
    start = next(i for i, ln in enumerate(lines) if header_fragment in ln)
    rows: dict[str, list[str]] = {}
    for ln in lines[start + 2:]:
        if not ln.strip().startswith("|"):
            break
        cells = [c.strip().replace("**", "") for c in ln.strip().strip("|").split("|")]
        rows[cells[0].lower()] = cells[1:]
    return rows


def close(published: str, stored: float, *, percent: bool = False) -> bool:
    """Does the stored value round to what the README prints?

    The README's own precision sets the tolerance. If it prints 62.9%, anything that
    rounds to 62.9% passes. This is the right rule: the README is allowed to round, it
    is not allowed to be wrong.
    """
    text = published.replace("%", "").strip()
    try:
        want = float(text)
    except ValueError:
        return False
    value = stored * 100 if percent else stored
    decimals = len(text.split(".")[1]) if "." in text else 0
    return round(value, decimals) == round(want, decimals)


def missing_links() -> list[str]:
    """Every local file the README and the evidence page point at must exist.

    WRITTEN IN TWO PASSES, AFTER SHIPPING BOTH KINDS OF BREAKAGE.

    Images first. Cutting the README to one screen moved its long-form half into docs/,
    and every `images/...` path in the moved text silently stopped resolving, because the
    same relative path now started one directory deeper. Nothing failed: GitHub renders a
    broken image as a small grey icon, and the only reader who notices is the one you were
    trying to impress.

    Then documents, after `docs/jd_coverage.md` was deleted and left five live links
    pointing at it across the README, two docs and three modules. A dead link in a
    repository someone is reviewing reads as carelessness about exactly the thing this
    project claims to be careful about, and it is invisible to a checker that only counts
    numbers. Anchors are stripped before the check because the file is what has to exist;
    verifying the heading too would fail on every legitimate rename of a section.
    """
    import re

    extensions = "png|jpe?g|gif|svg|webp|md|html|json|py|yaml|yml|cu|txt"
    pattern = re.compile(
        r'(?:src=|href=|]\()\s*"?((?!https?:|mailto:|data:|#)[^"\')\s>]+'
        rf'\.(?:{extensions})(?:#[^"\')\s>]*)?)')

    problems = []
    for name in ("README.md", "docs/evidence.md"):
        page = ROOT / name
        if not page.exists():
            continue
        for match in pattern.finditer(page.read_text()):
            raw = match.group(1)
            target = (page.parent / raw.split("#", 1)[0]).resolve()
            if not target.exists():
                problems.append(f"{name} points at {raw}, which does not exist")
    return sorted(set(problems))


def main(check_test_count: bool = True) -> int:
    """check_test_count is False when called from inside pytest, because the count is
    obtained BY running pytest and a test that spawns the collector it is running under
    is slow at best and recursive at worst."""
    # BOTH FILES. The long-form tables moved to docs/evidence.md when the README was cut
    # to one screen, and every check below is keyed on a table header. Reading only the
    # README would have left those checks matching nothing and passing silently, which is
    # the failure mode this whole script exists to prevent, arriving through the front
    # door.
    md = "\n".join(
        (ROOT / name).read_text()
        for name in ("README.md", "docs/evidence.md")
        if (ROOT / name).exists()
    )
    bad: list[str] = []
    bad += missing_links()

    # ---- out-of-distribution table -----------------------------------------
    ood = table_rows(md, "| Benchmark | AUROC · A |")
    for bench in BENCHMARKS:
        row = ood[bench]
        for i, (arm_key, arm) in enumerate(ARMS.items()):
            cell = load(f"ood_{arm}_{bench}.json")
            checks = [
                (f"AUROC {arm_key}", row[0 + i], cell["auroc_mean_pooled"], False),
                (f"miss rate {arm_key}", row[2 + i], cell["deployed"]["fnr"], True),
                (f"ECE {arm_key}", row[4 + i], cell["ece"], False),
            ]
            for label, published, stored, pct in checks:
                if not close(published, stored, percent=pct):
                    bad.append(f"{bench} {label}: README {published!r} vs artifact {stored!r}")

    # ---- McNemar table ------------------------------------------------------
    mc = table_rows(md, "| Benchmark | B catches, A misses |")
    book = load("ood_mcnemar.json")
    for bench in BENCHMARKS:
        row, cell = mc[bench], book[bench]
        pairs = [
            ("B catches A misses", row[0], cell["discordant"]["random_miss_mirror_catch"]),
            ("A catches B misses", row[1], cell["discordant"]["random_catch_mirror_miss"]),
            ("chi2", row[2], cell["mcnemar"]["chi2"]),
        ]
        for label, published, stored in pairs:
            if not close(published, float(stored)):
                bad.append(f"{bench} {label}: README {published!r} vs artifact {stored!r}")

    # ---- in-distribution table ----------------------------------------------
    # Backed by each run's own summary.json. Uncovered until now, which meant the
    # headline "the two arms are the same detector" rested on nothing checkable.
    ind = table_rows(md, "| Metric | Arm A · random |")
    for arm_key in ARMS:
        s = summary(arm_key)["val"]
        i = 0 if arm_key == "A" else 1
        for label, row_key, stored, pct in [
            ("AUROC", "auroc", s["auroc"], False),
            ("FNR at budget", "fnr at the 0.1% fpr budget", s["fnr"], True),
            ("ECE", "expected calibration error", s["ece"], False),
            ("threshold", "deployed threshold", s["threshold"], False),
            ("realised FPR", "realised fpr", s["fpr_at_budget"], True),
        ]:
            published = ind[row_key][i]
            if not close(published, stored, percent=pct):
                bad.append(f"in-distribution {label} {arm_key}: README {published!r} vs artifact {stored!r}")

    # ---- performance tables --------------------------------------------------
    # Added when the A100 runs landed. Without this the three performance tables would
    # be the only published numbers in the README with no gate behind them, which is
    # the exact hole this script was written to close for the evaluation tables.
    prof = load("profile/train_step_comparison.json")
    steps = table_rows(md, "| Arm | Median step |")
    for row_key, arm in [("gradient checkpointing on", "grad_ckpt_on"),
                         ("gradient checkpointing off", "grad_ckpt_off")]:
        row, cell = steps[row_key], prof["arms"][arm]
        for label, published, stored in [
            ("median step", row[0], cell["median_step_ms"]),
            ("p90 step", row[1], cell["p90_step_ms"]),
            ("peak memory", row[2], cell["peak_memory_gib"]),
        ]:
            if not close(published.split()[0], float(stored)):
                bad.append(f"{arm} {label}: README {published!r} vs artifact {stored!r}")

    # The op-share table reads the ranked list from the checkpointing-off arm, which is
    # the arm the README says it is reading.
    shares = {e["op"]: e["share"] for e in prof["arms"]["grad_ckpt_off"]["top"]}
    ops = table_rows(md, "| Operation | Share of device time |")
    for row_key, cells in ops.items():
        op = row_key.strip("`")
        if op not in shares:
            bad.append(f"op share: README names {op!r}, which is not in the profile's ranked list")
            continue
        if not close(cells[0], shares[op], percent=True):
            bad.append(f"op share {op}: README {cells[0]!r} vs artifact {shares[op]!r}")

    # ---- scaling table --------------------------------------------------------
    scal = {r["world_size"]: r
            for r in load("scaling/nvidia-a100-sxm4-80gb/scaling_summary.json")["rows"]}
    sc = table_rows(md, "| GPUs | Examples/s |")
    for row_key, cells in sc.items():
        ws = int(row_key)
        r = scal[ws]
        for label, published, stored, pct in [
            ("examples/s", cells[0], r["examples_per_second"], False),
            ("tokens/s", cells[1].replace(",", ""), r["tokens_per_second"], False),
            ("s/step", cells[2], r["seconds_per_step"], False),
            ("peak GiB", cells[3], r["peak_memory_gib"], False),
            ("speedup", cells[4], r["speedup"], False),
            ("efficiency", cells[5], r["scaling_efficiency"], True),
        ]:
            if not close(published, float(stored), percent=pct):
                bad.append(f"scaling ws={ws} {label}: README {published!r} vs artifact {stored!r}")

    # The two prose numbers the performance section argues from. Read from the artifact's
    # own stated cost rather than recomputed here, so there is one source for the figure.
    trade = prof["gradient_checkpointing_cost"]
    cost, saved = trade["step_time_overhead"] * 100, trade["memory_saved_gib"]
    m = re.search(r"\*\*\+([\d.]+)% step time\*\* to save \*\*([\d.]+) GiB\*\*", md)
    if not m:
        bad.append("the checkpointing trade sentence is missing from the README")
    else:
        if not close(m.group(1), cost):
            bad.append(f"checkpointing time cost: README {m.group(1)!r} vs computed {cost!r}")
        if not close(m.group(2), saved):
            bad.append(f"checkpointing memory saved: README {m.group(2)!r} vs computed {saved!r}")

    # ---- inference sweep ------------------------------------------------------
    # Published with the table, not in a later commit. Publishing numbers first and
    # gating them afterwards is how the evaluation tables went unchecked for months.
    inf_path = REPORTS / "inference" / "cpu_latency.json"
    if inf_path.exists():
        inf = {r["batch_windows"]: r for r in json.loads(inf_path.read_text())["batch_sweep"]}
        sweep = table_rows(md, "| Batch (windows) | Median |")
        for row_key, cells in sweep.items():
            b = int(row_key)
            if b not in inf:
                bad.append(f"inference table has batch {b}, which the artifact does not")
                continue
            r = inf[b]
            for label, published, stored in [
                ("median", cells[0].replace(" ms", ""), r["median_ms"]),
                ("p95", cells[1].replace(" ms", ""), r["p95_ms"]),
                ("windows/s", cells[2], r["windows_per_second"]),
                ("samples", cells[3], r["samples"]),
            ]:
                if not close(published, float(stored)):
                    bad.append(f"inference batch {b} {label}: README {published!r} vs "
                               f"artifact {stored!r}")
    else:
        bad.append(
            "README publishes an inference sweep but reports/experiments/inference/"
            "cpu_latency.json is not committed"
        )

    # ---- strategy comparison and the GPU sweep --------------------------------
    # Gated in the same commit that publishes them. Publishing first and gating later is
    # how the evaluation tables went unchecked for months.
    l40s = REPORTS / "scaling" / "nvidia-l40s"
    if "| GPUs | FSDP ex/s |" in md:
        for strat, fname in (("fsdp", "scaling_summary.json"),
                             ("deepspeed", "scaling_summary_deepspeed.json")):
            path = l40s / fname
            if not path.exists():
                bad.append(f"README publishes the L40S {strat} arm but {fname} is missing")
                continue
            rows = {r["world_size"]: r for r in json.loads(path.read_text())["rows"]}
            table = table_rows(md, "| GPUs | FSDP ex/s |")
            col = 0 if strat == "fsdp" else 1
            for ws in (1, 2):
                published = table[str(ws)][col]
                if not close(published, float(rows[ws]["examples_per_second"])):
                    bad.append(f"L40S {strat} ws={ws} ex/s: README {published!r} vs "
                               f"artifact {rows[ws]['examples_per_second']!r}")
                peak = table[str(ws)][2 + col].replace(" GiB", "")
                if not close(peak, float(rows[ws]["peak_memory_gib"])):
                    bad.append(f"L40S {strat} ws={ws} peak: README {peak!r} vs "
                               f"artifact {rows[ws]['peak_memory_gib']!r}")
            for label, row_key, field in [("speedup", "speedup", "speedup"),
                                          ("efficiency", "efficiency", "scaling_efficiency")]:
                published = table[row_key][col]
                stored = rows[2][field]
                if not close(published, float(stored), percent=(label == "efficiency")):
                    bad.append(f"L40S {strat} {label}: README {published!r} vs "
                               f"artifact {stored!r}")

    cuda_path = REPORTS / "inference" / "cuda_latency.json"
    if "| Batch (windows) | Median | Windows/s |" in md:
        if not cuda_path.exists():
            bad.append("README publishes a GPU sweep but cuda_latency.json is missing")
        else:
            cu = {r["batch_windows"]: r
                  for r in json.loads(cuda_path.read_text())["batch_sweep"]}
            for row_key, cells in table_rows(md, "| Batch (windows) | Median | Windows/s |").items():
                b = int(row_key)
                if b not in cu:
                    bad.append(f"GPU table has batch {b}, the artifact does not")
                    continue
                for label, published, stored in [
                    ("median", cells[0].replace(" ms", ""), cu[b]["median_ms"]),
                    ("windows/s", cells[1], cu[b]["windows_per_second"]),
                ]:
                    if not close(published, float(stored)):
                        bad.append(f"GPU batch {b} {label}: README {published!r} vs "
                                   f"artifact {stored!r}")

    # ---- thread sweep ---------------------------------------------------------
    th_path = REPORTS / "inference" / "cpu_threads.json"
    if "| Threads | Windows/s |" in md:
        if not th_path.exists():
            bad.append("README publishes a thread sweep but cpu_threads.json is missing")
        else:
            th = {r["threads"]: r for r in json.loads(th_path.read_text())["rows"]}
            for row_key, cells in table_rows(md, "| Threads | Windows/s |").items():
                n = int(row_key)
                if n not in th:
                    bad.append(f"thread table has {n} threads, the artifact does not")
                    continue
                r_ = th[n]
                for label, published, stored, pct in [
                    ("windows/s", cells[0], r_["windows_per_second"], False),
                    ("speedup", cells[1], r_["speedup_over_one_thread"], False),
                    ("efficiency", cells[2], r_["efficiency"], True),
                ]:
                    if not close(published, float(stored), percent=pct):
                        bad.append(f"threads={n} {label}: README {published!r} vs "
                                   f"artifact {stored!r}")

    # ---- spark local sweep ----------------------------------------------------
    spark_path = REPORTS / "spark" / "spark_summary.json"
    if "| Partitions | Threads each |" in md:
        if not spark_path.exists():
            bad.append("README publishes a Spark sweep but spark_summary.json is missing")
        else:
            sp = {r["partitions"]: r for r in json.loads(spark_path.read_text())["rows"]}
            for row_key, cells in table_rows(md, "| Partitions | Threads each |").items():
                n = int(row_key)
                if n not in sp:
                    bad.append(f"spark table has {n} partitions, the artifact does not")
                    continue
                r_ = sp[n]
                # No efficiency column. Every row of this sweep uses the same total
                # thread budget, so speedup/partitions is not parallel efficiency and the
                # multi_threading run returned 114.3% of it, which is what made the
                # mislabelling undeniable. The README publishes speedup against an ideal
                # of 1.0 and says so.
                for label, published, stored in [
                    ("docs/s", cells[1], r_["documents_per_second_compute"]),
                    ("speedup", cells[2], r_["speedup"]),
                    ("skew", cells[3], r_["skew"]),
                ]:
                    if not close(published, float(stored)):
                        bad.append(f"spark p={n} {label}: README {published!r} vs "
                                   f"artifact {stored!r}")


    # ---- beam local sweep -----------------------------------------------------
    # Gated on the same terms as the Spark sweep, plus two the Spark table does not need.
    # The comparison between the two runners is only legitimate if they scanned the same
    # number of documents, so that is checked across artifacts rather than trusted, and the
    # runner name is checked because "DirectRunner" in this artifact would mean the sweep
    # varied a worker count Prism never read.
    # WHICH SWEEP. The Beam runner has two local execution modes that produce different
    # startup costs and model-load counts for the same throughput, so its artifacts are
    # written one directory per mode. The README therefore has to say which one it is
    # publishing, in a marker rather than in prose, because a gate that guessed would pass
    # against whichever sweep happened to be on disk.
    beam_mode = re.search(r"<!-- beam-mode: (\w+) -->", md)
    beam_path = (REPORTS / "beam" / beam_mode.group(1) / "beam_summary.json"
                 if beam_mode else None)
    if "| Partitions | Model loads |" in md:
        if beam_mode is None:
            bad.append("README publishes a Beam sweep with no <!-- beam-mode: ... --> "
                       "marker, so there is no way to tell which sweep the table is from")
        elif not beam_path.exists():
            bad.append(f"README publishes the {beam_mode.group(1)} Beam sweep but "
                       f"{beam_path} is missing")
        else:
            doc = json.loads(beam_path.read_text())
            bm = {r["partitions"]: r for r in doc["rows"]}
            if doc.get("running_mode") != beam_mode.group(1):
                bad.append(f"README says the Beam sweep is {beam_mode.group(1)!r} but the "
                           f"artifact it points at was run as {doc.get('running_mode')!r}")
            if doc.get("runner") != "FnApiRunner":
                bad.append(f"beam artifact runner is {doc.get('runner')!r}, not FnApiRunner: "
                           "DirectRunner resolves to Prism, which ignores the worker count "
                           "the sweep varies, so the table would not be a scaling table")
            for row_key, cells in table_rows(md, "| Partitions | Model loads |").items():
                n = int(row_key)
                if n not in bm:
                    bad.append(f"beam table has {n} partitions, the artifact does not")
                    continue
                r_ = bm[n]
                for label, published, stored in [
                    ("model loads", cells[0], r_["model_loads"]),
                    ("docs/s", cells[1], r_["documents_per_second_compute"]),
                    ("speedup", cells[2], r_["speedup"]),
                    ("skew", cells[3], r_["skew"]),
                ]:
                    if not close(published, float(stored)):
                        bad.append(f"beam p={n} {label}: README {published!r} vs "
                                   f"artifact {stored!r}")
            if spark_path.exists():
                sp_doc = json.loads(spark_path.read_text())
                if doc["documents_scanned"] != sp_doc["documents_scanned"]:
                    bad.append(
                        f"the two runners scanned different numbers of documents "
                        f"(spark {sp_doc['documents_scanned']}, beam "
                        f"{doc['documents_scanned']}). A comparison between them is not a "
                        f"comparison. Re-run both with the same --max-docs."
                    )
                # THE CHECK THE DOCUMENT COUNT DOES NOT MAKE. Equal counts are a property
                # of the cap, not of the pool. The Spark sweep ran when data/silver held 6
                # human shards; the v0.2-min regeneration then added 24 generated shards to
                # the same tree, and a Beam sweep over the same root would have reported the
                # same 40 documents out of a different corpus. Only the fingerprint, which
                # hashes the shard list with each shard's row count, refuses that.
                pools = [(sp_doc.get("pool") or {}).get("fingerprint"),
                         (doc.get("pool") or {}).get("fingerprint")]
                if not all(pools):
                    bad.append(
                        "one of the published sweeps has no pool fingerprint, so there is "
                        "no evidence the two runners read the same shards. Re-run the sweep "
                        "that predates the fingerprint."
                    )
                elif pools[0] != pools[1]:
                    bad.append(
                        f"the two runners read different pools (spark {pools[0]}, beam "
                        f"{pools[1]}). Same document count, different corpus. Re-run both "
                        f"with the same --root and --pattern."
                    )

    # ---- beam, the second execution mode --------------------------------------
    # Its own marker and its own header. The two Beam tables started out with the same
    # header, which table_rows keys on, so the threading rows silently overwrote the
    # processing rows and the gate checked one artifact against the other table's numbers.
    # Two tables that mean different things cannot share a header.
    thr_mode = re.search(r"<!-- beam-threading-mode: (\w+) -->", md)
    if "| Partitions | Loads |" in md:
        if thr_mode is None:
            bad.append("README publishes a second Beam table with no "
                       "<!-- beam-threading-mode: ... --> marker")
        else:
            thr_path = REPORTS / "beam" / thr_mode.group(1) / "beam_summary.json"
            if not thr_path.exists():
                bad.append(f"README publishes the {thr_mode.group(1)} Beam sweep but "
                           f"{thr_path} is missing")
            else:
                thr = json.loads(thr_path.read_text())
                tm = {r["partitions"]: r for r in thr["rows"]}
                if thr.get("running_mode") != thr_mode.group(1):
                    bad.append(f"the second Beam table says {thr_mode.group(1)!r} but its "
                               f"artifact was run as {thr.get('running_mode')!r}")
                if beam_path is not None and beam_path.exists():
                    other = json.loads(beam_path.read_text())
                    if (other.get("pool") or {}).get("fingerprint") != \
                            (thr.get("pool") or {}).get("fingerprint"):
                        bad.append("the two Beam modes read different pools, so they are "
                                   "not two views of one sweep")
                for row_key, cells in table_rows(md, "| Partitions | Loads |").items():
                    n = int(row_key)
                    if n not in tm:
                        bad.append(f"beam threading table has {n} partitions, the artifact "
                                   f"does not")
                        continue
                    r_ = tm[n]
                    for label, published, stored in [
                        ("model loads", cells[0], r_["model_loads"]),
                        ("docs/s", cells[1], r_["documents_per_second_compute"]),
                        ("speedup", cells[2], r_["speedup"]),
                        ("skew", cells[3], r_["skew"]),
                    ]:
                        if not close(published, float(stored)):
                            bad.append(f"beam threading p={n} {label}: README "
                                       f"{published!r} vs artifact {stored!r}")

    # ---- the README headline table ---------------------------------------------
    # The README was cut from 880 lines to one screen, and the six figures that survived
    # the cut are the ones a visitor actually reads. Every other check here is keyed on a
    # table header that moved to docs/evidence.md, so without this block the most-read
    # numbers in the repository would have been the only ungated ones.
    head = "| | AUROC in distribution | AUROC on HC3 | Missed on HC3, at its own threshold |"
    if head in md:
        ood_cells = {(c["arm"], c["benchmark"]): c
                     for c in load("ood_summary.json")["cells"]}
        for label, arm_key, arm in (("Random prompts", "A", "baseline"),
                                    ("Matched mirrors", "B", "mirror")):
            row = re.search(rf"^\|\s*\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*\|(.+)$",
                            md, re.M)
            if row is None:
                bad.append(f"README headline table has no row for {label}")
                continue
            published = [c.strip().strip("*") for c in row.group(1).split("|") if c.strip()]
            if len(published) != 3:
                bad.append(f"README headline row {label} has {len(published)} cells, not 3")
                continue
            cell = ood_cells[(arm, "hc3")]
            for name, shown, stored, as_pct in (
                ("in-distribution AUROC", published[0], summary(arm_key)["val"]["auroc"],
                 False),
                ("HC3 AUROC", published[1], cell["auroc_mean_pooled"], False),
                ("HC3 miss rate", published[2], cell["deployed"]["fnr"], True),
            ):
                if not close(shown, float(stored), percent=as_pct):
                    bad.append(f"README headline {label} {name}: README {shown!r} vs "
                               f"artifact {stored!r}")

    # ---- the generated page ---------------------------------------------------
    # docs/index.html is built from the artifacts by scripts/build_evidence_page.py, so it
    # cannot state a figure they do not have. What it CAN do is go stale: an artifact
    # changes, nobody rebuilds, and the published page keeps showing the old run with no
    # sign that it is out of date. A rebuild is deterministic, so a byte comparison settles
    # it. This is the same contract as the README checks, enforced the cheaper way.
    page = ROOT / "docs" / "index.html"
    if page.exists():
        import subprocess

        before = page.read_text()
        build = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_evidence_page.py")],
            capture_output=True, text=True, cwd=ROOT)
        if build.returncode != 0:
            bad.append(f"docs/index.html could not be rebuilt: {build.stderr.strip()}")
        elif page.read_text() != before:
            bad.append(
                "docs/index.html is stale: rebuilding it from the artifacts produces "
                "different output. Run scripts/build_evidence_page.py and commit the "
                "result."
            )

    # ---- badges --------------------------------------------------------------
    auroc_badge = re.search(r"In--distribution%20AUROC-([\d.]+)-", md)
    if not auroc_badge or not close(auroc_badge.group(1), summary("A")["val"]["auroc"]):
        bad.append("AUROC badge does not match arm A's summary.json")

    fpr_badge = re.search(r"FPR%20budget-([\d.]+)%25-", md)
    if not fpr_badge or not close(fpr_badge.group(1), summary("A")["fpr_budget"], percent=True):
        bad.append("FPR budget badge does not match arm A's summary.json")

    tests_badge = re.search(r"Tests-(\d+)%20passing", md)
    counted, reason = collected_test_count() if check_test_count else (None, "skipped")
    if tests_badge is None:
        bad.append("no test-count badge found")
    elif counted is not None:
        if int(tests_badge.group(1)) != counted:
            bad.append(f"test count badge says {tests_badge.group(1)}, pytest collects {counted}")
    elif reason.startswith("unparsed"):
        # The check is broken. Saying nothing here is how it went unrun.
        bad.append(f"the test-count badge check did not work: {reason}")
    elif reason.startswith("unavailable"):
        print(f"note: the test-count badge was not checked. {reason}")

    # THE BADGE WAS GATED AND THE PROSE WAS NOT, so the prose drifted. For sixty-seven
    # tests' worth of commits this README carried a badge reading 985 and two sentences
    # reading 918, and nothing failed, because the check above only ever looked at the
    # badge. A number is checked or it is decoration; there is no third state. Every
    # written claim about how many tests exist is now compared against the same count.
    if counted is not None:
        for line, claimed in suite_size_claims(md):
            if claimed != counted:
                bad.append(f"README line {line} says {claimed} tests, "
                           f"pytest collects {counted}")

    # ---- the one prose number worth pinning ---------------------------------
    # It is the whole argument of the RAID paragraph, so it gets checked like a table.
    m = re.search(r"reverses in \*\*([\d.]+)% of", md)
    stored = load("ood_significance_raid.json")["fraction_of_resamples_where_the_gap_reverses"]
    if not m or not close(m.group(1), stored, percent=True):
        bad.append(f"RAID bootstrap reversal: README {m.group(1) if m else 'MISSING'!r} vs artifact {stored!r}")

    if bad:
        print(f"{len(bad)} README claim(s) do not match the committed artifacts:\n")
        for b in bad:
            print(f"  {b}")
        return 1
    print("every checked README claim matches its committed artifact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
