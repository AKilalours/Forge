"""Arm A for images: AI images generated WITHOUT reference to any photograph.

This is the control the whole image experiment is measured against, and it is easy to get
subtly wrong in a way that makes the comparison meaningless.

WHAT MUST BE IDENTICAL TO THE MIRROR ARM
    the generator families and their pinned revisions
    the generation spec: scheduler, steps, guidance, resolution
    the normalisation applied to the output
    the number of accepted images

WHAT MUST DIFFER, AND IS THE ONLY THING THAT MAY
    where the prompt came from. A mirror's prompt describes a specific photograph. A
    control's prompt is drawn from a fixed inventory of scenes that no photograph informed.

If anything in the first list differs, a win for the mirror arm is attributable to that
difference instead of to matching. The text track came within one config field of exactly
this failure: both arms pointed at the same directory and would have produced identical
numbers supporting the conclusion "matching makes no difference".

WHY AN INVENTORY RATHER THAN RANDOM WORDS. The control has to be a plausible attempt at
generating AI images, not a strawman. Someone building a detector without FORGE's mirror
idea would prompt a diffusion model with ordinary scene descriptions, so that is what the
control does. A control made deliberately weak proves nothing.

WHY THE LENGTH ANALOGUE IS RESOLUTION AND ASPECT. The text control samples its target
lengths from the human corpus so the two arms match in length distribution without matching
in content. The image equivalent is that the control renders at the same resolution and
aspect as the mirror arm, and both are normalised to the same square, so neither arm can be
identified by its geometry.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field

from forge.common.splits import assign_split
from forge.image.caption import Caption, to_prompt
from forge.image.generators import GenerationSpec
from forge.image.mirror import ImageMirror, MirrorStats
from forge.image.normalize import POLICY_V1, NormalizationPolicy, normalize_bytes
from forge.image.phash import dhash, to_hex

UNMATCHED_VERSION = "image_random_v1"

CaptionSimilarity = Callable[[bytes, str], float]

# A fixed, ordinary inventory. Deliberately mundane: these are the sorts of scenes a person
# building a detector would prompt for. Frozen with the dataset version, because changing
# the inventory changes the control arm's distribution.
SCENES: tuple[tuple[str, str, str], ...] = (
    ("a kitchen counter with utensils", "warm indoor light", "eye level"),
    ("a city street with parked cars", "flat overcast light", "eye level"),
    ("a wooden desk with stationery", "soft window light", "slightly above"),
    ("a park path lined with trees", "dappled afternoon light", "eye level"),
    ("a beach with scattered stones", "bright midday light", "low angle"),
    ("a bookshelf filled with books", "dim indoor light", "eye level"),
    ("a mountain ridge under cloud", "cool diffuse light", "low angle"),
    ("a bedroom with an unmade bed", "early morning light", "slightly above"),
    ("a market stall with produce", "mixed artificial light", "eye level"),
    ("a quiet office with two chairs", "flat ceiling light", "high angle"),
    ("a garden with potted plants", "golden late light", "eye level"),
    ("a railway platform at rest", "harsh overhead light", "low angle"),
)

OBJECTS = ("a few everyday objects", "several small items", "a handful of ordinary things")
PALETTES = ("muted neutrals", "cool greys and blues", "warm browns and creams", "soft greens")
TIMES = ("morning", "midday", "late afternoon", "evening")
LAYOUTS = ("centred subject", "subject off to one side", "subject low in the frame")
RELATIONS = ("objects grouped near the centre", "objects spread across the frame")


@dataclass
class UnmatchedStats(MirrorStats):
    """Same shape as the mirror arm's, so the two are directly comparable."""

    inventory_version: str = UNMATCHED_VERSION
    families_used: set = field(default_factory=set)


def _pick(sequence, index: int, field: str):
    """Choose from `sequence` by hashing (index, field).

    NOT modular arithmetic on the index. The first version used strides:

        OBJECTS[(index * 5) % len(OBJECTS)]
        PALETTES[(index * 13) % len(PALETTES)]

    with a comment claiming co-prime strides would keep the fields from moving in lockstep.
    That reasoning is wrong. A stride's period is len(sequence) // gcd(stride, len), and
    every one of those lengths divided the scene inventory's length, so the whole caption
    repeated every twelve indices: twelve distinct captions for twenty thousand images.

    Hashing each field with its own name gives independent draws, stays deterministic, and
    does not depend on any arithmetic relationship between the inventory sizes, so adding a
    scene later cannot quietly reintroduce a short period.
    """
    digest = hashlib.sha256(f"unmatched-v1:{field}:{index}".encode()).digest()
    return sequence[int.from_bytes(digest[:4], "big") % len(sequence)]


def spec_for(index: int) -> Caption:
    """Deterministic caption number `index` from the inventory.

    Index-derived rather than random so the control arm is reproducible and so a run can be
    resumed or extended without redrawing what it already generated.
    """
    scene, lighting, perspective = _pick(SCENES, index, "scene")
    return Caption(
        scene=scene,
        objects=_pick(OBJECTS, index, "objects"),
        spatial_relations=_pick(RELATIONS, index, "relations"),
        composition=_pick(LAYOUTS, index, "composition"),
        lighting=lighting,
        colour_palette=_pick(PALETTES, index, "palette"),
        camera_perspective=perspective,
        time_of_day=_pick(TIMES, index, "time"),
        notes={"arm": "unmatched", "inventory_index": index},
    )


def generate_unmatched(
    n: int,
    generator,
    caption_similarity: CaptionSimilarity,
    spec: GenerationSpec = GenerationSpec(),
    caption_match_min: float = 0.22,
    norm_policy: NormalizationPolicy = POLICY_V1,
    batch_size: int = 16,
) -> tuple[list[ImageMirror], list[tuple[str, bytes]], UnmatchedStats]:
    """Generate `n` control images. No photograph is read, and none is needed.

    The only validity check available is prompt adherence: there is no source image to
    compare against, by construction. That asymmetry is real and must be recorded, because
    it means the two arms are filtered to different degrees, and if the mirror arm wins,
    "its data passed a stricter filter" is the first alternative explanation a reader will
    reach for.
    """
    stats = UnmatchedStats(attempted=n)
    records: list[ImageMirror] = []
    images: list[tuple[str, bytes]] = []

    prepared = [(i, spec_for(i)) for i in range(n)]

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        prompts = [to_prompt(caption) for _, caption in chunk]
        try:
            candidates = generator.generate(prompts, spec)
        except Exception:
            stats.rejected["generation_failed"] += len(chunk)
            continue

        for (index, caption), prompt, candidate in zip(chunk, prompts, candidates, strict=True):
            try:
                normalised = normalize_bytes(candidate, norm_policy)
            except Exception:
                stats.rejected["normalisation_failed"] += 1
                continue

            if caption_similarity(normalised, prompt) < caption_match_min:
                stats.rejected["ignored_prompt"] += 1
                continue

            # No human source, so this image gets its own group and therefore its own split.
            sample_id = f"rand_img_{index:06d}"
            group = f"grp_{sample_id}"
            images.append((sample_id, normalised))
            records.append(
                ImageMirror(
                    sample_id=sample_id,
                    source_image_id="",
                    source_group_id=group,
                    split=assign_split(group).value,
                    phash=to_hex(dhash(normalised)),
                    caption=caption.as_dict(),
                    prompt_version=UNMATCHED_VERSION,
                    generator_family=generator.family,
                    generator_model_id=generator.model_id,
                    generator_revision=generator.revision,
                    generation=spec.as_dict(),
                    norm_policy=norm_policy.version,
                    # Explicitly false. This is the control: nothing is matched.
                    topic_match=False,
                    composition_match=False,
                    lighting_match=False,
                    notes={"arm": "unmatched", "inventory_index": index},
                )
            )
            stats.families_used.add(generator.family)
            stats.accepted += 1

    return records, images, stats


def assert_arms_comparable(mirror_spec: GenerationSpec, control_spec: GenerationSpec) -> None:
    """The two arms must differ in prompts and in nothing else.

    Called before training rather than trusted, because every field here is one a config
    edit could change silently, and any difference makes a win attributable to the setting
    instead of to matching.
    """
    if mirror_spec.as_dict() != control_spec.as_dict():
        differences = {
            key: (value, control_spec.as_dict()[key])
            for key, value in mirror_spec.as_dict().items()
            if control_spec.as_dict()[key] != value
        }
        raise RuntimeError(
            f"the two arms were generated with different settings: {differences}. A win for "
            "either arm would be attributable to that difference rather than to matching."
        )
