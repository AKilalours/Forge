"""Determine MAGE's label convention from the data, not from a card and not from the score.

WHY THIS EXISTS. The out-of-distribution run stopped on MAGE:

    machine-labelled documents mean score 0.0550
    human-labelled documents   mean score 0.1240

The guard refuses to report a number when the class it calls machine scores lower than the
class it calls human, because exactly two things produce that, and they mean opposite
things:

    the labels are inverted        -> a fixable parsing bug
    the detector does not transfer -> the actual finding

THE WRONG WAY TO RESOLVE IT is to flip the label and see whether the number improves. That
procedure always succeeds, because one of the two orientations always looks better, and it
converts an AUROC of 0.05 into a reported 0.95 with nothing crashing. It is the single
easiest way to publish a false result in this whole project.

THE RIGHT WAY is to ask the data. Every MAGE row carries `src`, naming where the text came
from. Machine rows name a GENERATOR; human rows name a source CORPUS. That association is a
property of the dataset and is independent of how well any detector performs on it, so it
settles the convention without consulting a single score.

This prints the evidence and a recommendation. It changes nothing. CPU only.
"""

from __future__ import annotations

import argparse
import collections
import re

# Model families that appear in MAGE's src strings. Presence of any of these in a src value
# means that row was machine-generated, whatever the numeric label says.
GENERATOR_MARKERS = (
    "gpt", "llama", "opt", "flan", "t5", "bloom", "glm", "dolly", "vicuna",
    "alpaca", "koala", "stablelm", "cohere", "davinci", "chatgpt", "gpt-j",
    "gpt-neo", "mixtral", "mistral", "falcon", "claude", "palm", "bard",
    "machine", "generated", "fake",
)
HUMAN_MARKERS = ("human", "real", "original")


def looks_generated(src: str) -> bool | None:
    """True, False, or None when the source string carries no evidence either way."""
    low = re.sub(r"[^a-z0-9]+", " ", src.lower())
    if any(f" {m} " in f" {low} " or m in low for m in HUMAN_MARKERS):
        return False
    if any(m in low for m in GENERATOR_MARKERS):
        return True
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    from forge.evaluation.benchmarks import load_hf

    pairs: collections.Counter = collections.Counter()
    by_label: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for i, row in enumerate(load_hf("yaful/MAGE", None, args.split)):
        if i >= args.limit:
            break
        label = int(row["label"])
        src = str(row.get("src", ""))
        pairs[(label, looks_generated(src))] += 1
        by_label[label][src] += 1

    print(f"\nsampled {sum(pairs.values())} rows from MAGE split={args.split}\n")
    for label in sorted(by_label):
        top = by_label[label].most_common(8)
        print(f"label {label}:  {sum(by_label[label].values())} rows")
        for src, count in top:
            verdict = looks_generated(src)
            tag = {True: "generator", False: "human corpus", None: "no evidence"}[verdict]
            print(f"    {src[:44]:<46} {count:>6}   {tag}")
        print()

    print("label x source-evidence:")
    for (label, evidence), count in sorted(pairs.items(), key=lambda kv: (-kv[1])):
        tag = {True: "generator", False: "human corpus", None: "no evidence"}[evidence]
        print(f"    label={label}  {tag:<14} {count:>6}")

    generated_by_label = {
        label: pairs[(label, True)] / max(1, pairs[(label, True)] + pairs[(label, False)])
        for label in by_label
    }
    print("\nfraction of evidence-bearing sources that name a generator:")
    for label, fraction in sorted(generated_by_label.items()):
        print(f"    label={label}: {fraction:.1%}")

    decided = [lab for lab, f in generated_by_label.items() if f > 0.9]
    if len(decided) == 1:
        print(
            f"\nCONCLUSION: label={decided[0]} is MACHINE-generated. "
            f"Set parse_mage(machine_label={decided[0]})."
        )
    else:
        print(
            "\nINCONCLUSIVE: the source strings do not separate the labels. Do NOT guess "
            "from the detector's scores. Read the dataset card and record what it says."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
