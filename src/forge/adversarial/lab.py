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
from forge.evaluation.metrics import false_negative_rate


@dataclass
class AttackResult:
    attack: str
    severity: float
    defused_by_preprocessing: bool
    fnr_raw: float
    fnr_preprocessed: float
    clean_fnr: float
    n_scored: int
    n_noop: int
    n_invalid: int

    @property
    def delta_raw(self) -> float:
        return self.fnr_raw - self.clean_fnr

    @property
    def delta_preprocessed(self) -> float:
        return self.fnr_preprocessed - self.clean_fnr

    def as_dict(self) -> dict:
        return {
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


def run_attacks(
    ai_texts: list[str],
    doc_ids: list[str],
    score_fn: Callable[[list[str]], list[float]],
    threshold: float,
    attacks: list[str] | None = None,
) -> list[AttackResult]:
    """ai_texts must all be genuinely AI-generated: FNR is the metric an evader moves."""
    names = attacks or list(RUNNABLE_OFFLINE)
    labels = [1] * len(ai_texts)
    clean = false_negative_rate(labels, score_fn([normalize(t) for t in ai_texts]), threshold)

    out: list[AttackResult] = []
    for name in names:
        spec = ATTACKS[name]
        for sev in spec.severities:
            keep_raw, keep_pre, n_noop, n_invalid = [], [], 0, 0
            for text, did in zip(ai_texts, doc_ids, strict=True):
                attacked = apply_attack(text, name, did, sev)
                if is_noop(text, attacked):
                    n_noop += 1
                    continue
                if not preserves_meaning(text, attacked):
                    n_invalid += 1
                    continue
                keep_raw.append(attacked)
                keep_pre.append(normalize(attacked))
            if not keep_raw:
                out.append(AttackResult(name, sev, spec.defused_by_preprocessing,
                                        clean, clean, clean, 0, n_noop, n_invalid))
                continue
            y = [1] * len(keep_raw)
            out.append(
                AttackResult(
                    attack=name, severity=sev,
                    defused_by_preprocessing=spec.defused_by_preprocessing,
                    fnr_raw=false_negative_rate(y, score_fn(keep_raw), threshold),
                    fnr_preprocessed=false_negative_rate(y, score_fn(keep_pre), threshold),
                    clean_fnr=clean, n_scored=len(keep_raw), n_noop=n_noop, n_invalid=n_invalid,
                )
            )
    return out


def render_table(results: list[AttackResult]) -> str:
    hdr = f"{'attack':<22}{'sev':>6}{'clean':>8}{'raw':>8}{'prep':>8}{'d_raw':>8}{'d_prep':>8}{'noop':>6}{'inval':>7}"
    lines = [hdr, "-" * len(hdr)]
    for r in sorted(results, key=lambda r: -r.delta_preprocessed):
        lines.append(
            f"{r.attack:<22}{r.severity:>6}{r.clean_fnr:>8.3f}{r.fnr_raw:>8.3f}"
            f"{r.fnr_preprocessed:>8.3f}{r.delta_raw:>8.3f}{r.delta_preprocessed:>8.3f}"
            f"{r.n_noop:>6}{r.n_invalid:>7}"
        )
    return "\n".join(lines)
