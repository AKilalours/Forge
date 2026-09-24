"""The result cards, built in Python instead of in the browser.

WHY THIS EXISTS. `api/forge_app.py` renders its cards in JavaScript from the JSON the API
returns. Streamlit has no such API to call: the analysis happens in the same process as the
page. So the same markup is built here, from the same payload dictionaries, against the
same stylesheet in `forge.ui.theme`.

THE RISK THIS FILE CARRIES, STATED PLAINLY. Two renderers for one payload can disagree, and
this project has shipped that mistake in several forms already. The mitigation is that both
consume `Report.as_dict()` and the scorer's own objects rather than reformatting values
themselves: a wording change belongs upstream, in the report, where both pick it up. When
this file starts deciding what a number means rather than where it sits, it has gone wrong.
"""

from __future__ import annotations

import html

# NO AI DETECTED, never HUMAN. A detector cannot establish that a person made something. It
# reports whether the input resembles the AI distribution it was trained on. "Human" is a
# claim about the world; "no AI detected" is a claim about the model, and only the second is
# true. Kept identical to the FastAPI page's WORD map.
VERDICT_WORD = {"ai": "AI DETECTED", "human": "NO AI DETECTED", "uncertain": "UNCERTAIN"}
VERDICT_CLASS = {"ai": "hot", "human": "ok", "uncertain": "warn"}
VERDICT_MARK = {"ai": "!", "human": "✓", "uncertain": "~"}

DIRECTION = {"toward_ai": "toward AI", "toward_capture": "toward capture", "neutral": ""}


def esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def banner(*, available: bool, verdict: str | None, probability: float | None,
           reason: str = "", score_label: str = "AI probability", score_note: str = "",
           threshold: float | None = None, caveat: str = "", soft: bool = False) -> str:
    """The headline. One shape for text and image, as on the FastAPI page.

    `threshold` marks the deployed decision boundary on the gauge. It is not decoration: the
    text arms decide at 0.992, so a document scoring 79.8% is a HUMAN verdict with the
    needle three quarters of the way toward the red end. Drawn without the boundary, the
    gauge and the headline contradict each other and the honest answer looks like a bug.
    """
    has_score = probability is not None
    pct = None if (not available or not has_score) else probability * 100
    cls = "pending" if not available else VERDICT_CLASS.get(verdict or "", "warn")
    mark = "?" if not available else VERDICT_MARK.get(verdict or "", "~")
    # A NEGATIVE FROM A DETECTOR THAT MISSES HALF ITS TARGETS IS NOT A GREEN TICK. `soft`
    # downgrades the confident styling for exactly that case; the word is unchanged,
    # because the word was already right. What was wrong was the page dressing a weak
    # negative as a clean result, which is how a reader concludes "this image is fine"
    # from a model whose measured recall is 0.55.
    if soft and available and verdict == "human":
        cls, mark = "warn", "~"
    word = "No verdict" if not available else VERDICT_WORD.get(verdict or "", str(verdict).upper())
    gauge = ""
    if pct is not None:
        position = min(100.0, max(0.0, pct))
        bound = ""
        klass = "gauge"
        if threshold is not None and 0.0 < threshold < 1.0:
            klass = "gauge hasbound"
            bound = f'<div class="bound" style="left:{min(100.0, threshold * 100):.4f}%"></div>'
        gauge = (
            f'<div class="{klass}"><div class="needle" style="left:{position}%"></div>'
            f'{bound}</div>'
            '<div class="gaugeends"><span>human</span><span>uncertain</span><span>AI</span></div>'
        )
    score_text = ("not available" if not available else "declared") if pct is None else f"{pct:.1f}%"
    return f"""<div class="assess big {cls}">
  <div class="verdictbox">
    <div class="mark">{mark}</div>
    <div><h2>{esc(word)}</h2><p>{esc(reason)}</p>
      {f'<p class="caveat">{esc(caveat)}</p>' if caveat else ''}</div>
  </div>
  <div class="scorebox">
    <div class="scorelabel">{esc(score_label)}</div>
    <div class="score {'na' if pct is None else ''}">{esc(score_text)}</div>
    {gauge}
    <div class="scorenote">{esc(score_note)}</div>
  </div></div>"""


def _row(key, value, status=None) -> str:
    shown = esc(value) if value not in (None, "") else '<span class="pill na">not available</span>'
    pill = f' <span class="pill {_pill_class(status)}">{esc(_pill_label(status))}</span>' if status else ""
    return f'<div class="row"><span class="k">{esc(key)}</span><span class="v">{shown}{pill}</span></div>'


def _pill_class(status: str) -> str:
    return {"high": "hot", "medium": "warn", "present": "ok", "low": "ok",
            "not_found": "na", "not_checked": "na"}.get(status, "na")


def _pill_label(status: str) -> str:
    return {"not_found": "not found", "not_checked": "not checked"}.get(status, status)


def _stream_row(stream: dict) -> str:
    width = stream["strength"] if stream.get("available") else 100
    tone = "" if stream.get("available") else " na"
    direction = DIRECTION.get(stream.get("direction") or "", "")
    tag = f' <span class="dir">{esc(direction)}</span>' if stream.get("available") and direction else ""
    # `state` is the word the panel shows INSTEAD of the number: not detected, weak,
    # partial, detected, not available. Reading `strength` here would put "50%" next to a
    # camera-metadata row, which reads as "50% likely to be a photograph" and is not what
    # the number means. Guessing a key name is also how a pill ships empty, so this asserts.
    label = stream.get("state")
    if not label:
        raise KeyError(f"evidence stream {stream.get('key')!r} carries no 'state'")
    return f"""<div class="stream">
  <div class="sname">{esc(stream['label'])}<span class="pill {'na' if not stream.get('available') else ''}">
    {esc(label)}</span></div>
  <div class="bar"><div class="fill{tone}" style="width:{width}%"></div></div>
  <div class="sub">{esc(stream.get('summary'))}{tag}</div></div>"""


TIMING_ORDER = ("preview", "detector", "forensics", "perceptual_hash", "evidence",
                "attribution", "forensic_maps", "detector_robustness")
TIMING_NAME = {"preview": "Preview", "detector": "Detector inference",
               "forensics": "Forensic analysis", "perceptual_hash": "Perceptual hash",
               "evidence": "Evidence engine", "attribution": "Attribution map",
               "forensic_maps": "Pixel diagnostics", "detector_robustness": "Robustness"}


def _tech_row(row: dict) -> str:
    """A manipulation row: the categorical grade, then the raw statistic, then the caveat.

    The raw number is shown UNDER the grade rather than as the value, because a reader who
    sees "0.361" beside "Recompression" will read it as a probability. It is not one.
    """
    pill = (f'<span class="pill {_pill_class(row["status"])}">{esc(_pill_label(row["status"]))}</span>'
            if row.get("status") else '<span class="pill na">not available</span>')
    value = (f'<div class="tech">raw statistic {esc(row["value"])}</div>'
             if row.get("value") not in (None, "") else "")
    note = f'<div class="tech">{esc(row["note"])}</div>' if row.get("note") else ""
    return (f'<div class="techrow"><div class="row"><span class="k">{esc(row["label"])}</span>'
            f'<span class="v">{pill}</span></div>{value}{note}</div>')


def _robustness_card(rows: list[dict]) -> str:
    """Does the VERDICT survive the transforms an image meets in the wild?

    Distinct from signal stability, which asks whether a forensic SIGNAL survives. A signal
    can vanish while the verdict holds, and the verdict can flip while every signal is
    intact. Reporting one as the other is the mistake this panel exists to avoid.
    """
    if not rows:
        return ""
    scored = [r for r in rows if not r.get("error")]
    held = sum(1 for r in scored if not r.get("changed"))
    chips = []
    for row in rows:
        if row.get("error"):
            chips.append(f'<div class="chip"><div class="n">{esc(row["attack"])}</div>'
                         '<div class="d">failed</div></div>')
            continue
        flipped = bool(row.get("changed"))
        delta = ("" if row["attack"] == "original" else
                 f' · {"+" if row["delta"] >= 0 else ""}{row["delta"] * 100:.1f} pts')
        chips.append(
            f'<div class="chip"><div class="n">{esc(row["attack"])}'
            f'<span class="pill {"hot" if flipped else "ok"}">'
            f'{"flipped" if flipped else "held"}</span></div>'
            f'<div class="d">{row["ai_probability"] * 100:.1f}%{delta}</div></div>'
        )
    return (
        f'<div class="card wide"><h3>Robustness <span class="sum">{held} of {len(scored)} '
        'edits keep the same verdict</span></h3>'
        '<p class="caveat" style="margin:0 0 12px">Every edit is re-scored and compared with '
        'the original. "flipped" means that edit changes the answer.</p>'
        f'<div class="rgrid">{"".join(chips)}</div></div>'
    )


def _attribution_card(attribution: dict | None) -> str:
    if not attribution or not attribution.get("image"):
        return ""
    peak = attribution["peak"]
    reading = attribution.get("reading") or ""
    return f"""<div class="card wide">
  <h3>What the verdict rests on <span class="sum">{attribution['grid']}×{attribution['grid']} occlusion</span></h3>
  <div class="attr"><img src="{attribution['image']}" alt="attribution map">
    <div><p class="caveat">{esc(reading)}</p>
      <div class="row"><span class="k">Score with everything visible</span>
        <span class="v">{attribution['base_probability'] * 100:.1f}%</span></div>
      <div class="row"><span class="k">Largest single drop when hidden</span>
        <span class="v">{peak['drop'] * 100:.1f} pts</span></div>
      <p class="caveat">Measured by hiding each region and re-scoring, not by a gradient
      approximation, so it holds for any detector this project loads.</p>
    </div></div></div>"""


def _maps_card(maps: list[dict]) -> str:
    if not maps:
        return ""
    figures = "".join(
        f'<figure><img src="{m["image"]}" alt="{esc(m["title"])}">'
        f'<figcaption><b>{esc(m["title"])}</b><br>'
        f'{esc(m.get("short") or m.get("what_it_shows"))}</figcaption></figure>'
        for m in maps
    )
    return ('<details class="card wide fold" open><summary>Pixel statistics</summary>'
            f'<div class="foldbody"><div class="maps">{figures}</div>'
            '<p class="caveat">Computed from the file without any model. The attribution '
            'panel above is what the detector used.</p></div></details>')


def _details_card(payload: dict) -> str:
    timings = payload.get("timings_ms") or {}
    elapsed = payload.get("elapsed_ms") or 0
    timing_rows = "".join(
        f'<div class="row"><span class="k">{esc(TIMING_NAME.get(k, k))}</span>'
        f'<span class="v">{timings[k]:,} ms</span></div>'
        for k in TIMING_ORDER if k in timings
    )
    detector = payload.get("detector") or {}
    footer = (f'<p class="tech">Detector {esc(detector.get("model_id"))}, uncalibrated '
              "baseline.</p>" if detector.get("available") else "")
    return (
        f'<details class="card wide fold"><summary>Details '
        f'<span class="sum">{elapsed / 1000:.1f} s on CPU</span></summary>'
        '<div class="foldbody"><h3>Manipulation analysis</h3>'
        + "".join(_tech_row(r) for r in payload.get("manipulation", []))
        + '<h3 style="margin-top:18px">Provenance</h3>'
        + "".join(_row(r["label"], r["value"], r.get("status"))
                  for r in payload.get("provenance", []))
        + '<h3 style="margin-top:18px">Timing</h3>'
        + timing_rows
        + _row("Total", f"{elapsed:,} ms")
        + footer
        + "</div></details>"
    )


def _image_threshold(assessment: dict) -> float | None:
    """The AI boundary, parsed from the band the assessment reports.

    `band` is "[low, high)": below low is not AI, at or above high is AI. The gauge marks
    the AI end, because that is where the verdict flips to ai.
    """
    band = assessment.get("band")
    if not band:
        return None
    try:
        return float(band.strip("[)").split(",")[1])
    except (IndexError, ValueError):
        return None


def detector_caveat(detector: dict) -> str:
    """One sentence about what a negative from this detector is worth.

    Built from the payload's measured numbers, never written as a literal, so a
    re-measured detector changes the page rather than leaving a stale claim behind.
    """
    recall = detector.get("recall_at_threshold")
    if not detector.get("available") or recall is None or recall >= 1.0:
        return ""
    sample = detector.get("recall_sample") or {}
    total = sum(v for v in sample.values() if isinstance(v, int)) or None
    where = f" on {total} images" if total else ""
    fitted = ", and that operating point was fitted on the same images" if detector.get(
        "recall_in_sample") else ""
    return (
        f"This detector caught {recall:.0%} of known AI images{where} at its operating "
        f"point{fitted}. A negative is weak evidence: it misses roughly "
        f"{1 - recall:.0%} of AI images, and newer generators are not in that sample."
    )


def image_result(payload: dict, compact: bool = False) -> str:
    """The image tab, section for section as the FastAPI page renders it.

    Banner, then the three-card grid, then robustness, attribution, pixel statistics and
    details. The order is not decoration: the verdict is first, what it rests on comes
    before the model-free statistics, and the statistics are folded because a reader who
    opens them should have already seen the panel that says which one the detector used.
    """
    assessment = payload["assessment"]
    evidence = payload["evidence"]
    streams = evidence.get("streams", [])
    primary = next((s for s in streams if s["key"] == "visual_model"), None)
    supporting = [s for s in streams if s["key"] != "visual_model"]

    caveat = detector_caveat(payload.get("detector") or {})
    head = banner(
        available=bool(assessment["available"]),
        verdict=assessment.get("verdict"),
        probability=assessment.get("confidence"),
        reason=assessment.get("reason", ""),
        score_label="Determination" if assessment.get("confidence") is None else "AI probability",
        score_note=assessment.get("detail", ""),
        threshold=_image_threshold(assessment),
        caveat=caveat,
        soft=bool(caveat),
    )

    preview = (f'<img class="prev" src="{payload["preview"]}" alt="preview">'
               if payload.get("preview") else "")
    image_card = (
        f'<div class="card"><h3>Image</h3>{preview}<div style="margin-top:11px">'
        + _row("File", payload.get("filename"))
        + _row("Size", f'{payload["size_bytes"] / 1024:.0f} KB')
        + "</div></div>"
    )

    conflict = evidence.get("conflict")
    conflict_note = ("" if conflict == "low" else
                     f'<p class="caveat">{esc(evidence.get("conflict_reason"))}</p>')
    evidence_card = (
        '<div class="card"><h3>Evidence</h3><div class="grouplabel">Detector</div>'
        + (_stream_row(primary) if primary else "")
        + '<div class="grouplabel">Supporting signals</div>'
        + "".join(_stream_row(s) for s in supporting)
        + '<div class="row" style="margin-top:8px"><span class="k">Agreement</span>'
        f'<span class="v"><span class="pill {_pill_class(conflict)}">'
        f'{esc("consistent" if conflict == "low" else conflict)}</span></span></div>'
        + conflict_note
        + "</div>"
    )

    signals_card = (
        '<div class="card"><h3>File signals</h3>'
        + "".join(_row(r["label"], r["value"], r.get("status"))
                  for r in payload.get("authenticity", []))
        + "</div>"
    )

    if compact:
        # THE DEPLOYED PAGE SHOWS THE VERDICT, WHAT IT RESTS ON, AND THE TWO PANELS THAT
        # SHOW THE ANALYSIS RATHER THAN ASSERT IT: the occlusion map, which marks the
        # regions the decision actually used, and the pixel statistics, which are computed
        # from the file with no model at all. What is dropped is the eleven-chip
        # robustness row and the timing table: a visitor cannot act on either, and they
        # pushed the verdict off the screen. Both remain on the FastAPI reference page.
        return head + (
            '<div class="grid">' + image_card + evidence_card + signals_card
            + _attribution_card(payload.get("attribution_map"))
            + _maps_card(payload.get("maps") or [])
            + "</div>"
        )

    return head + (
        '<div class="grid">'
        + image_card + evidence_card + signals_card
        + _robustness_card(payload.get("robustness") or [])
        + _attribution_card(payload.get("attribution_map"))
        + _maps_card(payload.get("maps") or [])
        + _details_card(payload)
        + "</div>"
    )


def _spark(windows: list[float]) -> str:
    """Per-window probabilities as bars. Omitted for a single window, where it says nothing."""
    if len(windows) < 2:
        return ""
    bars = "".join(f'<i style="height:{max(2, round(p * 100))}%"></i>' for p in windows)
    return f'<div class="spark">{bars}</div>'


def _arm_card(arm: dict) -> str:
    tone = {"ai": "hot", "human": "ok"}.get(arm["verdict"], "warn")
    return (
        f'<div class="card"><h3>{esc(arm["label"])}</h3>'
        + _row("Verdict", arm["verdict"], tone)
        + _row("AI probability (mean over windows)", f'{arm["ai_probability"] * 100:.2f}%')
        + _row("Highest single window", f'{arm["max_window_probability"] * 100:.2f}%')
        + _row("Deployed threshold", f'{arm["threshold"]:.6f}')
        + _row("FPR budget", arm["fpr_budget"])
        + _row("Distance from threshold", f'{arm["confidence"] * 100:.1f}%')
        + _row("Windows scored", arm["n_windows"])
        + _row("Its validation FNR", f'{arm["val_fnr"] * 100:.3f}%')
        + _row("Its validation ECE", arm["val_ece"])
        + _spark(arm.get("windows") or [])
        + "</div>"
    )


def text_result(payload: dict) -> str:
    """The text tab, from the payload `forge.inference.text_api.analyse` returns.

    Deliberately the same structure as the FastAPI page's JavaScript, because it is the same
    page: one banner carrying the deployed arm's verdict, both arms side by side beneath it
    because the comparison is the experiment, and the limitations card last. A product has
    one verdict, not two, so the headline is the mirror arm and the other sits beside it.
    """
    unavailable = "".join(
        f'<div class="card err"><h3>{esc(name)} unavailable</h3><p>{esc(reason)}</p></div>'
        for name, reason in (payload.get("unavailable") or {}).items()
    )

    if not payload.get("available"):
        head = banner(
            available=False, verdict=None, probability=None,
            reason="Neither arm could be loaded, so no score is produced. A number here "
                   "would be fabricated.",
            score_label="AI probability, calibrated",
            score_note="Load a trained checkpoint into outputs/ to enable scoring.",
        )
        return head + (f'<div class="grid">{unavailable}'
                       f'<div class="card"><h3>Input</h3>'
                       f'{_row("Words", payload.get("words"))}</div></div>')

    arms = payload["arms"]
    primary = next((a for a in arms if a["arm"] == "mirror"), arms[0])
    if len(arms) > 1:
        agree = arms[0]["verdict"] == arms[1]["verdict"]
        opening = ("Both arms agree." if agree else
                   "THE ARMS DISAGREE: "
                   + ", ".join(f'{a["label"]} says {a["verdict"]}' for a in arms) + ".")
    elif payload.get("unavailable"):
        # An arm WAS asked for and did not load. That is a shortfall and it is stated,
        # alongside the reason, further down the page.
        opening = "Only one arm is loaded."
    else:
        # Only one arm was ever requested. That is the deployed page's design, not a
        # shortfall, and "Only one arm is loaded" read to a visitor as though something
        # had failed. A page apologising for working correctly is worse than a silent one.
        opening = ""

    plural = "" if primary["n_windows"] == 1 else "s"
    head = banner(
        available=True,
        verdict=primary["verdict"],
        probability=primary["ai_probability"],
        reason=(f'{opening} ' if opening else '') + (
            f'Mean over {primary["n_windows"]} window{plural}, '
            f'{payload["words"]} words.'),
        score_label="AI probability, calibrated",
        threshold=primary["threshold"],
        score_note=(
            f'{primary["label"]}, threshold {primary["threshold"]:.6f} at a '
            f'{primary["fpr_budget"]} false-positive budget. Validation ECE '
            f'{primary["val_ece"]}. Out of distribution this model is badly miscalibrated: '
            "see the note below."
        ),
    )

    provenance = " · ".join(esc(a["model_version"]) for a in arms)
    limits = (
        '<div class="card wide"><h3>What this score is not</h3>'
        f'<p>{esc(payload["caveat"])}</p><p>{esc(payload["abstention"])}</p>'
        '<p class="tech" style="margin-top:9px">Provenance, for reproducibility rather than '
        f'display: {provenance}</p></div>'
    )

    cards = "".join(_arm_card(a) for a in arms)
    return head + f'<div class="grid">{cards}{unavailable}{limits}</div>'


EXTRA_CSS = """
/* Streamlit paints its own background behind this markup, so the page ground is set here
   rather than on body, which the host controls. */
.forgewrap{background:var(--bg);color:var(--ink);padding:2px 0 10px;
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.forgewrap .grid{margin-bottom:14px}
.forgewrap ul.lim{margin:4px 0 0;padding-left:18px;color:var(--muted);font-size:12.5px}
.forgewrap ul.lim li{margin:5px 0}
"""


def document(body: str) -> str:
    """Wrap rendered cards in the shared stylesheet, for embedding in a Streamlit page."""
    from forge.ui.theme import CSS

    return f"<style>{CSS}{EXTRA_CSS}</style><div class='forgewrap'>{body}</div>"
