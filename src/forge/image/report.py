"""One analysis of one image, assembled into the panels a person actually reads.

WHY REPORTS ARE SEPARATE FROM FORENSICS. The forensic functions produce facts about a file.
A report is a document shown to a human, and the dangerous step is the one where facts
become a verdict. That step gets its own file, with its own rules, rather than being
scattered through a template.

THE TWO RULES.

1. THE VERDICT COMES FROM THE DETECTOR OR IT DOES NOT EXIST. FORGE-Image's heads are not
   trained, so `assessment.available` is False and no confidence is shown. The reference
   design this page follows puts a large "Human, 98.2%" at the top; that number comes from
   a calibrated model, and manufacturing it from metadata would be worst on exactly the
   images a detector must never accuse. Screenshots, chat-app downloads and re-saved
   photographs all have no EXIF and library quantisation tables, so any naive combination
   scores ordinary human images as suspicious.

2. EVIDENCE STREAMS ARE REPORTED, NOT SUMMED. See forge.image.evidence for why. The short
   version: the signals describe how a file was written, not what is in it, and averaging
   them destroys the one genuinely informative case, which is disagreement between them.

WHAT IS REAL TODAY, with no model and no GPU: container format, EXIF, camera consistency,
C2PA presence, self-declared generation markers, quantisation tables, error level analysis,
resampling periodicity, noise consistency, colour statistics, perceptual hash, and a
transform-stability check that shows which signals survive resizing, recompression and
metadata removal. That last one is a real robustness result about the FILE; detector
robustness needs the detector and says so.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from forge.image.attribution import occlusion_attribution
from forge.image.evidence import Evidence, build_evidence
from forge.image.forensics import (
    HIGH,
    LOW,
    MEDIUM,
    NOT_CHECKED,
    NOT_FOUND,
    PRESENT,
    Finding,
    analyze,
)
from forge.image.phash import dhash, distance, to_hex

REPORT_VERSION = "image_report_v2"


@dataclass(frozen=True)
class Assessment:
    """The headline. Empty until a detector exists, and it says why."""

    available: bool = False
    verdict: str | None = None
    confidence: float | None = None
    band: str | None = None
    # Product wording, not development notes. The interface says WHAT the state is; the
    # argument for why a metadata-derived score would be dishonest belongs in the docs and
    # in this file, not on the page. `detail` carries the longer form for anyone who opens
    # the limitations section.
    reason: str = "Visual detector unavailable. No AI probability is produced for images."
    detail: str = (
        "No verdict is produced when the detector is missing or its polarity has not been "
        "measured. A score derived instead from metadata, compression or noise would respond "
        "mainly to missing metadata, so it would read highest on screenshots, chat-app "
        "downloads and re-saved photographs: ordinary human images. Everything else shown "
        "here is supporting evidence and none of it establishes authorship."
    )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Attribution:
    """Per-region AI/human segmentation. Needs a localisation head. Not the attribution map."""

    available: bool = False
    ai_area_fraction: float | None = None
    mixed_content: bool | None = None
    heatmap: str | None = None
    # This is the LOCALISATION field: a per-region AI/human segmentation, which needs a head
    # FORGE-Image does not have. It is not the attribution map. Attribution is computed by
    # forge.image.attribution, by occlusion over the loaded detector, and travels in the
    # response under `attribution_map`. The two were easy to confuse, so both the field name
    # and this reason now say which is which.
    reason: str = (
        "No per-region AI/human segmentation is produced; that needs a localisation head "
        "this project has not trained. Region importance for the loaded detector is reported "
        "separately as an occlusion attribution map."
    )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Row:
    """One labelled line in a panel."""

    label: str
    value: str | None
    status: str
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Report:
    filename: str
    size_bytes: int
    phash: str
    findings: list[Finding]
    evidence: Evidence
    assessment: Assessment
    attribution: Attribution
    authenticity: list[Row]
    manipulation: list[Row]
    provenance: list[Row]
    stability: list[Row]
    stability_available: bool
    preview: str | None = None
    elapsed_ms: int = 0
    report_version: str = REPORT_VERSION
    cannot_conclude: list[str] = field(default_factory=list)
    # Per stage, in milliseconds. Measured, not estimated. Without this, "53 seconds" is
    # a number nobody can act on: the only useful next question is WHICH stage, and
    # guessing at that is how people optimise the part that was already fast.
    timings_ms: dict = field(default_factory=dict)
    # What produced the verdict, and whether the verdict survives transformation. Both are
    # empty when no detector loaded; neither is ever filled from forensics.
    detector: dict = field(default_factory=dict)
    robustness: list = field(default_factory=list)
    attribution_map: dict = field(default_factory=dict)

    def by_name(self, name: str) -> Finding | None:
        return next((f for f in self.findings if f.name == name), None)

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "phash": self.phash,
            "findings": [f.as_dict() for f in self.findings],
            "evidence": self.evidence.as_dict(),
            "assessment": self.assessment.as_dict(),
            "attribution": self.attribution.as_dict(),
            "authenticity": [r.as_dict() for r in self.authenticity],
            "manipulation": [r.as_dict() for r in self.manipulation],
            "provenance": [r.as_dict() for r in self.provenance],
            "stability": [r.as_dict() for r in self.stability],
            "stability_available": self.stability_available,
            "preview": self.preview,
            "elapsed_ms": self.elapsed_ms,
            "report_version": self.report_version,
            "cannot_conclude": self.cannot_conclude,
            "timings_ms": self.timings_ms,
            "detector": self.detector,
            "robustness": self.robustness,
            "attribution_map": self.attribution_map,
        }


def _get(findings: list[Finding], name: str) -> Finding | None:
    return next((f for f in findings if f.name == name), None)


def _authenticity_rows(findings: list[Finding]) -> list[Row]:
    exif = _get(findings, "exif")
    fields = exif.value if exif and isinstance(exif.value, dict) else {}
    consistency = _get(findings, "camera_consistency")
    c2pa = _get(findings, "c2pa")
    declaration = _get(findings, "ai_declaration")
    fmt = _get(findings, "file_type")

    camera = " ".join(
        v for v in (fields.get("camera_make"), fields.get("camera_model")) if v
    ) or None
    captured = fields.get("datetime_original") or fields.get("datetime")

    return [
        Row("Camera", camera, PRESENT if camera else NOT_FOUND,
            "read from EXIF; editable by anything that writes the file"),
        Row("Capture date", captured, PRESENT if captured else NOT_FOUND,
            "an EXIF claim, not a verified timestamp"),
        Row("Metadata consistency",
            consistency.value if consistency and consistency.value else None,
            consistency.status if consistency else NOT_CHECKED,
            "describes the metadata's internal shape, not the pixels"),
        Row("C2PA signature",
            "Markers present" if c2pa and c2pa.status == PRESENT else None,
            c2pa.status if c2pa else NOT_CHECKED,
            "structural detection only; not cryptographically verified"),
        Row("Generation marker",
            "; ".join(declaration.value[:2]) if declaration and declaration.value else None,
            declaration.status if declaration else NOT_CHECKED,
            "the file's own declaration; absence means nothing"),
        Row("File type", fmt.value if fmt else None, fmt.status if fmt else NOT_CHECKED,
            "read from the magic bytes, not the extension"),
    ]


def _manipulation_rows(findings: list[Finding]) -> list[Row]:
    error = _get(findings, "error_level")
    resample = _get(findings, "resample")
    noise = _get(findings, "noise_consistency")
    colour = _get(findings, "colour")
    summary = _get(findings, "manipulation_summary")

    def _val(finding: Finding | None, key: str) -> str | None:
        if not finding or not isinstance(finding.value, dict):
            return None
        value = finding.value.get(key)
        return None if value is None else str(value)

    return [
        Row("Recompression", _val(error, "mean_residual"),
            error.status if error else NOT_CHECKED,
            "error level analysis responds to any re-encode, innocent ones included"),
        Row("Residual patchiness", _val(error, "patchiness"),
            error.status if error else NOT_CHECKED,
            "uneven residual suggests regions with different compression history"),
        Row("Resizing", _val(resample, "peak_to_median"),
            resample.status if resample else NOT_CHECKED,
            "a later JPEG pass destroys this signal, so absence proves nothing"),
        Row("Noise uniformity", _val(noise, "variation"),
            noise.status if noise else NOT_CHECKED,
            "smooth skies and denoising produce uneven noise in genuine photographs"),
        Row("Clipping", _val(colour, "clipped_fraction"),
            colour.status if colour else NOT_CHECKED,
            "heavy clipping suggests aggressive grading, nothing about authorship"),
        Row("Strongest signal", summary.value if summary else None,
            summary.status if summary else NOT_CHECKED,
            "the strongest single signal, deliberately not a combined score"),
    ]


def _provenance_rows(findings: list[Finding]) -> list[Row]:
    exif = _get(findings, "exif")
    c2pa = _get(findings, "c2pa")
    fmt = _get(findings, "file_type")
    tables = _get(findings, "jpeg_tables")
    quality = (
        str(tables.value.get("estimated_quality"))
        if tables and isinstance(tables.value, dict)
        else None
    )
    standard = (
        tables.value.get("standard_tables")
        if tables and isinstance(tables.value, dict)
        else None
    )
    return [
        Row("C2PA manifest", "Present" if c2pa and c2pa.status == PRESENT else None,
            c2pa.status if c2pa else NOT_CHECKED,
            "almost nothing signs its output today; absence is the normal case"),
        Row("EXIF block",
            f"{exif.detail.get('field_count')} fields" if exif and exif.status == PRESENT else None,
            exif.status if exif else NOT_CHECKED,
            "stripped by every social platform and every screenshot"),
        Row("Container", fmt.value if fmt else None, fmt.status if fmt else NOT_CHECKED,
            "the real format, from the magic bytes"),
        Row("Encoder quality", quality, tables.status if tables else NOT_CHECKED,
            "inverted from the quantisation tables"),
        Row("Quantisation tables",
            None if standard is None else ("Library defaults" if standard else "Custom / camera"),
            tables.status if tables else NOT_CHECKED,
            "library defaults describe most generated images AND most re-saved photographs"),
    ]


def _stability_rows(data: bytes) -> tuple[list[Row], bool]:
    """Which signals survive the transforms an image meets in the wild?

    THIS IS NOT DETECTOR ROBUSTNESS. Detector robustness asks whether the model's verdict
    holds under attack, and needs the model. This asks whether the FILE's signals hold, and
    is answerable today.

    It is also genuinely informative: it demonstrates, on the reader's own image, that
    metadata removal destroys the provenance evidence entirely while the perceptual hash
    survives compression. That is the practical reason a detector cannot lean on metadata.
    """
    from forge.image.attacks import ATTACKS, apply_attack
    from forge.image.forensics import ai_declaration, exif

    try:
        original_hash = dhash(data)
        had_exif = exif(data).status == PRESENT
        had_declaration = ai_declaration(data).status == HIGH
    except Exception:  # noqa: BLE001
        return [], False

    rows: list[Row] = []
    for attack in ATTACKS:
        try:
            transformed = apply_attack(data, attack.name)
            drift = distance(original_hash, dhash(transformed))
            kept_exif = exif(transformed).status == PRESENT
            kept_declaration = ai_declaration(transformed).status == HIGH
        except Exception:  # noqa: BLE001
            rows.append(Row(attack.name, None, NOT_CHECKED, "transform failed on this file"))
            continue

        lost = []
        if had_exif and not kept_exif:
            lost.append("EXIF")
        if had_declaration and not kept_declaration:
            lost.append("generation marker")
        status = HIGH if lost else (MEDIUM if drift > 6 else LOW)
        detail = f"hash drift {drift}/64"
        if lost:
            detail += ", lost " + " and ".join(lost)
        rows.append(
            Row(attack.name, detail, status,
                "signal survival under this transform, not detector robustness")
        )
    return rows, True


def _cannot_conclude(findings: list[Finding], stability: list[Row], detection=None) -> list[str]:
    """What this page does NOT establish, stated next to what it does.

    THE REGRESSION. The first line was fixed text saying the trained detector "does not
    exist yet", and it kept printing directly underneath a verdict the detector had just
    produced: 0.9% AI, no AI detected, and then a sentence denying the detector exists. The
    section is the honest part of the page, which makes a false line in it worse here than
    almost anywhere else.

    The honest sentence depends on whether a verified detector ran, so the list now depends
    on it too. With one, the limitation is real and specific: a measured baseline at 55%
    recall, in sample, is not a trained FORGE image model. Without one, nothing on the page
    speaks to generation at all.
    """
    verified = detection is not None and getattr(detection, "polarity_verified", False)
    lines = [
        "Whether this image was generated, with confidence. The visual detector is a "
        "measured baseline, not a trained FORGE model: its operating point was fitted on a "
        "small labelled set and its recall there was 55%, so a low score is weak evidence "
        "of a photograph rather than proof of one."
        if verified else
        "Whether this image was generated. No verified visual detector is loaded, and "
        "nothing else on this page is a substitute for one.",
    ]
    exif = _get(findings, "exif")
    if exif and exif.status != PRESENT:
        lines.append(
            "Anything, from the missing EXIF. Screenshots, social-platform downloads and "
            "chat apps strip metadata from ordinary photographs."
        )
    tables = _get(findings, "jpeg_tables")
    if tables and isinstance(tables.value, dict) and tables.value.get("standard_tables"):
        lines.append(
            "Anything, from the library quantisation tables. They mean software wrote the "
            "file, which is equally true of a re-saved photograph."
        )
    declaration = _get(findings, "ai_declaration")
    if declaration and declaration.status != HIGH:
        lines.append(
            "Anything, from the absent generation marker. Most generators write none and "
            "any editor removes them."
        )
    if any(r.status == HIGH for r in stability):
        lines.append(
            "That the signals shown above would survive redistribution. The stability panel "
            "shows which of them a routine transform destroys."
        )
    return lines


def _detector_robustness(data: bytes, detector) -> list[dict]:
    """Does the VERDICT survive the transforms an image meets in the wild?

    Distinct from signal stability, which asks whether a forensic signal survives. A signal
    can vanish while the verdict holds, and the verdict can flip while every signal is
    intact. Reporting one as the other is the mistake this pair of panels exists to avoid.

    The original is scored first and every row is compared against it, so a row says whether
    the transform moved the answer, not merely what the answer was.
    """
    from forge.image.attacks import ATTACKS, apply_attack

    base = detector.detect(data)
    rows = [{
        "attack": "original", "ai_probability": round(base.ai_probability, 4),
        "verdict": base.verdict, "changed": False, "delta": 0.0,
    }]
    for attack in ATTACKS:
        try:
            variant = detector.detect(apply_attack(data, attack.name))
        except Exception as error:  # noqa: BLE001 - a failed transform is reported as one
            rows.append({"attack": attack.name, "error": f"{type(error).__name__}: {error}"})
            continue
        rows.append({
            "attack": attack.name,
            "ai_probability": round(variant.ai_probability, 4),
            "verdict": variant.verdict,
            "changed": variant.verdict != base.verdict,
            "delta": round(variant.ai_probability - base.ai_probability, 4),
        })
    return rows


def _assessment(findings: list[Finding], detection) -> Assessment:
    """The verdict, in priority order. A declaration outranks a probability.

    DECLARATION FIRST. When the file carries a generation marker written by the producing
    software (an IPTC digitalSourceType of trained algorithmic media, a C2PA generative
    assertion, a Stable Diffusion parameter block), that is not an inference about the
    pixels. It is a label the generator wrote about its own output, and nothing else writes
    those strings. Letting a mediocre baseline detector overrule it with a low probability
    was a design mistake: the ChatGPT image in testing declared itself generated and the
    page reported NO AI DETECTED at 1.5%.

    The asymmetry from `ai_declaration` is preserved exactly. PRESENT is believed. ABSENT
    means nothing at all, because most generators write no marker, every editor strips
    metadata, and a screenshot of a generated image carries none of it. Absence never
    contributes to a human verdict.

    A declaration also does not need the detector. It stands on its own, so a declared image
    is reported as AI even when no detector is loaded or its polarity is unmeasured.
    """
    declaration = _get(findings, "ai_declaration")
    if declaration is not None and declaration.status == HIGH:
        markers = ", ".join(str(v) for v in (declaration.value or []))
        detail = f"Generation marker in the file: {markers}."
        if detection is not None:
            # The declaration decides, but the detector's number is still a measurement and
            # is still shown. "declared" with no figure at all reads as a missing value, and
            # a disagreement here is worth seeing rather than hiding.
            detail += (
                f" The visual detector independently scored {detection.ai_probability:.1%}"
                + (
                    ", which agrees."
                    if detection.ai_probability >= detection.threshold_ai
                    else ", which does not agree; the marker outranks it."
                )
            )
        return Assessment(
            available=True,
            verdict="ai",
            confidence=detection.ai_probability if detection is not None else None,
            band=None,
            reason="Declared generated by the software that wrote the file.",
            detail=detail,
        )

    if detection is not None and detection.polarity_verified:
        return Assessment(
            available=True,
            verdict=detection.verdict,
            confidence=detection.ai_probability,
            band=f"[{detection.threshold_ai:.3f}, {detection.confident_ai:.3f})",
            # NO SENTENCE UNDER THE VERDICT. This used to read "Not AI below 0.319, AI at
            # or above 0.319", which is not a useful sentence: when the human ceiling and
            # the confident-AI floor round to the same figure the uncertain band is narrower
            # than the rounding, so it reads as a contradiction sitting under the headline.
            # The operating point is still in the payload as `band`, in the Details section
            # and in /health, where a number belongs. The banner states the verdict.
            reason="",
            # Explicitly empty. The dataclass default is the "no detector exists yet"
            # paragraph, and it was rendering next to a live verdict.
            detail="",
        )

    return Assessment(
        reason=(
            "Detector not configured."
            if detection is None
            else "Detector polarity not verified."
        ),
        detail=(
            "No visual detector is loaded."
            if detection is None
            else "Which class this detector calls AI has not been measured against labelled "
                 "images, so a verdict could be inverted. Run scripts/image_detector_probe.py."
        ),
    )


def build_report(
    data: bytes,
    filename: str = "upload",
    preview: str | None = None,
    with_stability: bool = False,
    with_detector: bool = True,
) -> Report:
    """Analyse one image end to end. Pure CPU: no GPU, no network, no model weights.

    `with_stability` gates the transform-survival pass, which re-encodes the image ten times
    and dominates the wall clock: about 35 of the 39 seconds a 2.8 MB JPEG takes. It answers
    a question a reader asks second, after "what does this file say", so the interface runs
    it on demand rather than making every upload wait for it. Off means the rows are absent
    and `stability_available` is False, never that they are empty because nothing survived.
    """
    import time

    started = time.perf_counter()
    timings: dict[str, int] = {}

    def _timed(name, fn, default=None):
        mark = time.perf_counter()
        try:
            return fn()
        except Exception:  # noqa: BLE001 - a stage that fails still reports its cost
            return default
        finally:
            timings[name] = int((time.perf_counter() - mark) * 1000)

    findings = _timed("forensics", lambda: analyze(data), default=[])
    fingerprint = _timed("perceptual_hash", lambda: to_hex(dhash(data)), default="")
    stability, stability_available = (
        _timed("transform_stability", lambda: _stability_rows(data), default=([], False))
        if with_stability
        else ([], False)
    )
    # The detector decides the verdict. Nothing else is allowed to.
    detection, detector_info, detector_obj = None, {}, None
    if with_detector:
        mark = time.perf_counter()
        try:
            from forge.image.detector import load_detector

            detector_obj = load_detector()
            detection = detector_obj.detect(data)
            detector_info = {
                "available": True, "model_id": detection.model_id,
                "calibrated": detection.calibrated, "labels": list(detection.labels),
                "polarity_verified": detection.polarity_verified,
            }
        except Exception as error:  # noqa: BLE001 - absence is reported, never substituted
            detector_info = {"available": False, "reason": str(error)}
        timings["detector"] = int((time.perf_counter() - mark) * 1000)
    else:
        timings["detector"] = 0

    evidence = _timed(
        "evidence",
        lambda: build_evidence(
            findings,
            detector_available=detection is not None,
            probability=detection.ai_probability if detection else None,
            model_id=detection.model_id if detection else "",
            polarity_verified=bool(detection and detection.polarity_verified),
        ),
    )

    attribution: dict = {}
    if detector_obj is not None:
        built = _timed(
            "attribution",
            lambda: occlusion_attribution(detector_obj, data),
        )
        attribution = built.as_dict() if built is not None else {}

    robustness: list[dict] = []
    if detector_obj is not None:
        robustness = _timed(
            "detector_robustness",
            lambda: _detector_robustness(data, detector_obj),
            default=[],
        )

    return Report(
        filename=filename,
        size_bytes=len(data),
        phash=fingerprint,
        findings=findings,
        evidence=evidence,
        assessment=_assessment(findings, detection),
        attribution=Attribution(),
        authenticity=_authenticity_rows(findings),
        manipulation=_manipulation_rows(findings),
        provenance=_provenance_rows(findings),
        stability=stability,
        stability_available=stability_available,
        preview=preview,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        cannot_conclude=_cannot_conclude(findings, stability, detection),
        timings_ms=timings,
        detector=detector_info,
        robustness=robustness,
        attribution_map=attribution,
    )
