"""One analysis, two shells.

`api/forge_app.py` served this over HTTP and `streamlit_app.py` needs the identical result
in-process. Writing the scoring loop twice would put the two pages one edit apart from
disagreeing about what a verdict is, which is the failure this project has shipped in
several forms already. Both call `analyse` and render the same payload.

The decision policy is not reimplemented here either. Thresholds come from each arm's
committed summary.json and the verdict comes from forge.inference.decision.decide, so there
is exactly one place where a score becomes a claim.
"""

from __future__ import annotations

CAVEAT = (
    "Trained on four generator families at 1.7B to 3.8B parameters. Against unseen "
    "generators these checkpoints miss 63% to 96% of AI text at this threshold and "
    "their ECE rises from 0.004 to 0.18-0.44. A confident score here is not evidence "
    "of a confident model. See docs/evaluation.md."
)
ABSTENTION = (
    "Off. The uncertain band is derived from validation scores "
    "(decision.band_from_validation), which are not committed, and a band chosen by "
    "eye would silently change the false-positive rate."
)


def analyse(text: str, arms: tuple[str, ...] | None = None) -> dict:
    """Score with BOTH arms by default and report them side by side.

    `arms` narrows that to a subset, and exists for one host rather than as a feature: on
    Streamlit Community Cloud two float32 arms at 701 MB each, plus torch, plus the page,
    do not fit in about 2.7 GB, and the failure mode is the container being killed
    mid-request, which a reader cannot interpret. A page that holds one arm and says so is
    honest; one that is killed is not. The default is unchanged, so the FastAPI reference
    page and the tests still score both.

    Both rather than one, because the comparison IS the project. A single verdict hides the
    thing the experiment measured, and the arms genuinely disagree near the threshold: on
    RAID, 122 documents arm A misses arm B catches.

    An arm that cannot load reports WHY, in `unavailable`, and never falls back to a score.
    That includes running out of memory, which is a real possibility on a small host: one
    arm is roughly 740 MB in float32. A page that then shows the arm it does have, and says
    only one is loaded, is honest; one that invents the second is not.
    """
    from forge.inference.decision import decide
    from forge.inference.scorer import ARMS, ArmUnavailable, load_arm

    selected = tuple(arms) if arms else ARMS
    unknown = [a for a in selected if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}, expected a subset of {ARMS}")

    words = len(text.split())
    arms: list[dict] = []
    unavailable: dict[str, str] = {}

    for name in selected:
        try:
            arm = load_arm(name)
            scored = arm.score(text)
        except ArmUnavailable as error:
            unavailable[name] = str(error)
            continue
        except MemoryError:
            unavailable[name] = (
                "not enough memory to hold a second arm alongside the first on this host"
            )
            continue
        except Exception as error:                      # noqa: BLE001 - surfaced, not hidden
            unavailable[name] = f"{type(error).__name__}: {error}"
            continue

        decision = decide(scored.mean, arm.policy)
        arms.append({
            "arm": scored.arm,
            "label": scored.label,
            "verdict": decision.verdict.value,
            "ai_probability": scored.mean,
            "max_window_probability": scored.maximum,
            "confidence": decision.confidence,
            "threshold": arm.policy.threshold,
            "fpr_budget": arm.policy.fpr_budget,
            "model_version": arm.policy.model_version,
            "abstained": decision.abstained,
            "n_windows": scored.n_windows,
            "windows": scored.window_probabilities[:64],
            "val_fnr": arm.summary["val"].get("fnr"),
            "val_ece": arm.summary["val"].get("ece"),
        })

    return {
        "available": bool(arms),
        "words": words,
        "arms": arms,
        "unavailable": unavailable,
        "aggregation": "mean over windows; max shown alongside because it inflates FPR",
        "caveat": CAVEAT,
        "abstention": ABSTENTION,
    }
