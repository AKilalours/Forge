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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "experiments"
BENCHMARKS = ("hc3", "raid", "mage")
ARMS = {"A": "baseline", "B": "mirror"}


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text())


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


def main() -> int:
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
