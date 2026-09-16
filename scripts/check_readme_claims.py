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


def main(check_test_count: bool = True) -> int:
    """check_test_count is False when called from inside pytest, because the count is
    obtained BY running pytest and a test that spawns the collector it is running under
    is slow at best and recursive at worst."""
    md = (ROOT / "README.md").read_text()
    bad: list[str] = []

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
                for label, published, stored, pct in [
                    ("docs/s", cells[1], r_["documents_per_second_compute"], False),
                    ("speedup", cells[2], r_["speedup"], False),
                    ("efficiency", cells[3], r_["efficiency"], True),
                    ("skew", cells[4], r_["skew"], False),
                ]:
                    if not close(published, float(stored), percent=pct):
                        bad.append(f"spark p={n} {label}: README {published!r} vs "
                                   f"artifact {stored!r}")

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
