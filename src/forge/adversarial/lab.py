"""Adversarial evaluation runner.

Produces the delta-FNR table, in both preprocessing conditions, with no-ops and invalid
attacks excluded from the scores and reported separately.

Reading the output. For an attack flagged `defused_by_preprocessing`, the `preprocessed`
column should show delta-FNR near zero and the `raw` column shows what an attacker would
achieve against a deployment that skipped normalisation. The gap between the two columns
is the measured value of the preprocessing defence. For an attack that survives, both
columns describe the model, and the `preprocessed` one is the production number.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from forge.adversarial.attacks import (
    ATTACKS,
    RUNNABLE_OFFLINE,
    apply_attack,
    is_noop,
    preserves_meaning,
)
from forge.cleaning.normalize import normalize
from forge.evaluation.metrics import false_negative_rate, false_positive_rate

# THE CONDITIONS, each of which is a deployment choice rather than a variation on one.
#
# "raw" and "normalised" are what a deployment can already choose between, and the gap
# between them is the measured value of the existing preprocessing. The other two are
# CANDIDATE DEFENCES for the two attacks that normalisation does nothing about: homoglyph
# substitution at 0.306 and case perturbation at 0.978, both of which move the preprocessed
# column by less than a thousandth.
#
# THE CONTROL THAT MAKES THIS HONEST. Every condition is scored on the CLEAN documents too,
# and clean_fnr is recorded per condition. A defence is only worth adopting if it lowers
# the attacked FNR by more than it raises the clean one, and both of these transforms move
# every document, not just the attacked ones. The deployed checkpoints were trained on
# normalised, unfolded, cased text, so folding or casefolding at inference is a distribution
# shift whose cost has to be measured in the same run or the comparison is rigged. Reporting
# an attacked column against a clean baseline measured under a different transform is the
# single easiest way to publish a defence that does not work.
#
# THE HOLE THE FIRST VERSION OF THIS CONTROL LEFT, found by running it.
#
# The per-condition clean baseline scores CLEAN AI DOCUMENTS. It catches a transform that
# stops the detector recognising AI text. It is blind to the opposite failure, and the
# opposite failure is what a transform like casefolding is most likely to cause.
#
# FNR is the fraction of AI documents scored BELOW the threshold. A transform that pushes
# every score UP drives FNR to zero on every attack and on the clean baseline at once, and
# reads as a perfect, free defence. That is exactly what the first casefolded run produced:
# 0.000 in every cell of every row, clean and attacked alike, including case_perturb at
# 0.978. A number that good against seventeen attacks is a measurement artefact, not a
# defence.
#
# What it cannot be, from this lab alone, is distinguished from a real defence, because
# this lab only ever scores AI documents. The cost of inflated scores lands entirely on
# HUMAN documents as false positives, and the deployed threshold was fitted at FPR 0.001
# on cased, unfolded text. So human documents are scored through every condition too, and
# the FPR at the same threshold is reported beside the FNR. A defence that halves FNR and
# triples FPR is not a defence, and until this column existed there was no way to see it.
#
# casefolded is included expecting it to FAIL, and it is worth running for that reason.
# Case perturbation destroys information: no transform recovers the original casing, so
# folding case can only help by making attacked and clean text look alike to the tokenizer.
# Against a cased DeBERTa-v3 whose threshold was fitted on cased text, the likely outcome
# is that the clean column degrades as much as the attacked column improves. If so, the
# honest conclusion is that case robustness needs training-time augmentation and cannot be
# bought at inference, which is a retrain rather than a normalisation change.
CONDITIONS: dict[str, Callable[[str], str]] = {
    "raw": lambda t: t,
    "normalised": normalize,
    "folded": lambda t: normalize(t, fold_confusables=True),
    "casefolded": lambda t: normalize(t, fold_confusables=True).casefold(),
}

# What a default run measures. The two candidate defences double the number of forward
# passes per cell, so they are opt in: on CPU this lab already takes minutes.
DEFAULT_CONDITIONS = ("raw", "normalised")


@dataclass(frozen=True)
class ConditionCost:
    """What a condition does to HUMAN documents, at the deployed threshold.

    Measured once per run, not per attack, because it does not depend on the attack: it is
    the price of the transform itself. `fpr` at the fitted threshold is the number that
    says whether a condition can be deployed at all.
    """

    fpr: float
    n_human: int


@dataclass
class AttackResult:
    attack: str
    severity: float
    defused_by_preprocessing: bool
    fnr: dict[str, float]
    clean: dict[str, float]
    n_scored: int
    n_noop: int
    n_invalid: int

    # The original four fields are properties now. The two names this report has always
    # published keep meaning exactly what they meant, so the artifact schema and its
    # readers are unchanged; the conditions dict is additive.
    @property
    def fnr_raw(self) -> float:
        return self.fnr["raw"]

    @property
    def fnr_preprocessed(self) -> float:
        return self.fnr["normalised"]

    @property
    def clean_fnr(self) -> float:
        return self.clean["normalised"]

    @property
    def delta_raw(self) -> float:
        return self.fnr_raw - self.clean["raw"]

    @property
    def delta_preprocessed(self) -> float:
        return self.fnr_preprocessed - self.clean_fnr

    def delta(self, condition: str) -> float:
        """Attacked FNR minus the clean FNR MEASURED UNDER THE SAME CONDITION.

        Not minus a single global baseline. delta_raw used to subtract the normalised
        clean FNR from the raw attacked FNR, which silently charged the raw column for
        whatever normalisation was worth on clean text. It happened to be harmless here
        because clean FNR is 0.000 in both, and it would not have been harmless for a
        transform that costs clean accuracy, which is exactly what the two candidate
        defences might do.
        """
        return self.fnr[condition] - self.clean[condition]

    def as_dict(self) -> dict:
        out = {
            "attack": self.attack, "severity": self.severity,
            "defused_by_preprocessing": self.defused_by_preprocessing,
            "clean_fnr": round(self.clean_fnr, 6),
            "fnr_raw": round(self.fnr_raw, 6),
            "fnr_preprocessed": round(self.fnr_preprocessed, 6),
            "delta_fnr_raw": round(self.delta_raw, 6),
            "delta_fnr_preprocessed": round(self.delta_preprocessed, 6),
            "preprocessing_benefit": round(self.delta_raw - self.delta_preprocessed, 6),
            "n_scored": self.n_scored, "n_noop": self.n_noop, "n_invalid": self.n_invalid,
        }
        # Per condition, attacked and clean side by side, because the pair is the claim.
        out["conditions"] = {
            name: {"fnr": round(self.fnr[name], 6),
                   "clean_fnr": round(self.clean[name], 6),
                   "delta_fnr": round(self.delta(name), 6)}
            for name in self.fnr
        }
        return out


def run_attacks(
    ai_texts: list[str],
    doc_ids: list[str],
    score_fn: Callable[[list[str]], list[float]],
    threshold: float,
    attacks: list[str] | None = None,
    conditions: tuple[str, ...] = DEFAULT_CONDITIONS,
    human_texts: list[str] | None = None,
) -> tuple[list[AttackResult], dict[str, ConditionCost]]:
    """ai_texts must all be genuinely AI-generated: FNR is the metric an evader moves.

    human_texts is what stops a score-inflating transform reading as a perfect defence.
    Without it this function can only report FNR, and FNR alone cannot tell a defence from
    a transform that pushes every score above the threshold. Returns the per-condition
    human cost alongside the results; it is empty when no human documents are supplied,
    and a caller that publishes a new condition without it is publishing half a number.
    """
    unknown = [c for c in conditions if c not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown conditions {unknown}; known: {sorted(CONDITIONS)}")
    if "normalised" not in conditions:
        raise ValueError(
            "the normalised condition is the production path and every published delta is "
            "stated against it, so it cannot be dropped from a run."
        )

    names = attacks or list(RUNNABLE_OFFLINE)
    labels = [1] * len(ai_texts)
    # ONE CLEAN BASELINE PER CONDITION. This is the control: a transform that costs clean
    # accuracy must be charged for it, and the only way to see that is to score the clean
    # documents through the same transform.
    clean = {
        c: false_negative_rate(labels, score_fn([CONDITIONS[c](t) for t in ai_texts]),
                               threshold)
        for c in conditions
    }

    cost: dict[str, ConditionCost] = {}
    if human_texts:
        human_labels = [0] * len(human_texts)
        for c in conditions:
            cost[c] = ConditionCost(
                fpr=false_positive_rate(
                    human_labels,
                    score_fn([CONDITIONS[c](t) for t in human_texts]),
                    threshold),
                n_human=len(human_texts),
            )

    out: list[AttackResult] = []
    for name in names:
        spec = ATTACKS[name]
        for sev in spec.severities:
            kept: list[str] = []
            n_noop = n_invalid = 0
            for text, did in zip(ai_texts, doc_ids, strict=True):
                attacked = apply_attack(text, name, did, sev)
                if is_noop(text, attacked):
                    n_noop += 1
                    continue
                if not preserves_meaning(text, attacked):
                    n_invalid += 1
                    continue
                kept.append(attacked)
            if not kept:
                out.append(AttackResult(name, sev, spec.defused_by_preprocessing,
                                        fnr=dict(clean), clean=dict(clean),
                                        n_scored=0, n_noop=n_noop, n_invalid=n_invalid))
                continue
            y = [1] * len(kept)
            out.append(
                AttackResult(
                    attack=name, severity=sev,
                    defused_by_preprocessing=spec.defused_by_preprocessing,
                    fnr={c: false_negative_rate(
                        y, score_fn([CONDITIONS[c](t) for t in kept]), threshold)
                        for c in conditions},
                    clean=dict(clean),
                    n_scored=len(kept), n_noop=n_noop, n_invalid=n_invalid,
                )
            )
    return out, cost


def render_cost(cost: dict[str, ConditionCost]) -> str:
    """The human column, printed separately because it is per condition and not per attack.

    Printed even when empty, saying so, because a silent absence is how the casefolded
    column nearly got published as a free perfect defence.
    """
    if not cost:
        return ("NO HUMAN DOCUMENTS SCORED. Every number above is FNR on AI documents "
                "only, so a transform that inflates all scores is indistinguishable here "
                "from a real defence. Pass human documents before trusting a new "
                "condition.")
    n = next(iter(cost.values())).n_human
    body = ", ".join(f"{c}={v.fpr:.4f}" for c, v in cost.items())
    return (
        f"false positive rate on {n} human documents at the deployed threshold: {body}"
        f"\n  A condition whose FPR rises has not defended anything, it has moved the "
        f"operating point."
        f"\n  RESOLUTION IS 1/{n} = {1 / n:.4f}. This cannot confirm an FPR budget of "
        f"0.001 and is not trying to; it is here to catch a transform whose FPR is not "
        f"small, which is what perfect FNR across every attack actually looks like."
    )


def render_table(results: list[AttackResult]) -> str:
    """Three columns per condition: clean, attacked, and the delta between them.

    The clean value is a COLUMN and not a preamble line, for two reasons. A reader
    comparing a candidate defence against the production path needs both halves on the
    same row, or a defence that trades clean accuracy for attacked accuracy reads as a
    straight win. And the header has to stay on the first line: the ordering test asserts
    that the worst attack is the first row after the rule, and a preamble line above the
    header quietly shifted every index by one. That test was right and the preamble was
    wrong, so the data moved into the table rather than the test moving to accommodate it.
    """
    if not results:
        return "no attacks scored"
    conditions = list(results[0].fnr)
    hdr = f"{'attack':<22}{'sev':>6}"
    for c in conditions:
        tag = c[:6]
        hdr += f"{tag + '.cln':>10}{tag + '.atk':>10}{tag + '.d':>9}"
    hdr += f"{'noop':>6}{'inval':>7}"
    lines = [hdr, "-" * len(hdr)]
    for r in sorted(results, key=lambda r: -r.delta_preprocessed):
        row = f"{r.attack:<22}{r.severity:>6}"
        for c in conditions:
            row += f"{r.clean[c]:>10.3f}{r.fnr[c]:>10.3f}{r.delta(c):>9.3f}"
        lines.append(row + f"{r.n_noop:>6}{r.n_invalid:>7}")
    return "\n".join(lines)
