"""Generate an AI image matched to a photograph, and refuse the ones that are not matched.

THE SHAPE, identical to the text mirror engine:

    photograph -> caption -> prompt -> generator -> candidate -> validate -> mirror

and the photograph is discarded after captioning. Only the caption crosses.

WHAT "VALID" MEANS HERE, and why it needs two checks rather than one.

A generated image can fail to be a mirror in two opposite ways, and a single similarity
threshold catches only one of them.

  TOO DISSIMILAR. The generator ignored the prompt, or the caption was too vague to steer
  it. The pair is then a random photograph next to a random generation, which is the
  control arm, not the mirror arm. Training on it while calling it a mirror would silently
  turn arm B into a second copy of arm A.

  TOO SIMILAR. The generator reproduced something close to a memorised training image, or
  the caption was specific enough to name a particular real photograph. Now the AI example
  contains the human example's content, the two classes overlap, and the detector is being
  asked to separate an image from itself.

So similarity to the source must land inside a BAND. That is the image analogue of the
text validator's length-ratio window, and it exists for the same reason: a mirror that is
too close is contamination and one that is too far is not a mirror.

NORMALISATION IS APPLIED TO THE OUTPUT HERE. Generated images leave the pipeline as
lossless PNG at the generator's native resolution; photographs arrived as JPEG. If the two
populations were stored differently the detector would learn the encoder. See
forge/image/normalize.py, which is the same function ingestion used.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from forge.common.splits import assign_split
from forge.image.caption import Caption, CaptionRejected, to_prompt
from forge.image.caption import validate as validate_caption
from forge.image.generators import GenerationSpec
from forge.image.normalize import POLICY_V1, NormalizationPolicy, normalize_bytes
from forge.image.phash import dhash, to_hex

MIRROR_VERSION = "image_mirror_v1"

# A similarity function returns a value in [0, 1]. In production it is CLIP cosine
# similarity rescaled; in tests it is injected. Keeping it injected means the validity
# LOGIC is testable without downloading a model, which is the same trick that made the
# text validator testable.
Similarity = Callable[[bytes, bytes], float]
CaptionSimilarity = Callable[[bytes, str], float]


@dataclass(frozen=True)
class ValidityPolicy:
    """Thresholds, all in [0, 1]. Frozen with the dataset version."""

    caption_match_min: float = 0.22     # the generator rendered what was asked
    source_similarity_min: float = 0.45  # below this it is not a mirror at all
    source_similarity_max: float = 0.92  # above this it is a reproduction, not a mirror

    def __post_init__(self) -> None:
        if not 0 <= self.source_similarity_min < self.source_similarity_max <= 1:
            raise ValueError(
                "source similarity band must satisfy 0 <= min < max <= 1; a band that is "
                "empty or inverted rejects everything or accepts everything"
            )


@dataclass(frozen=True)
class ImageMirror:
    sample_id: str
    source_image_id: str
    source_group_id: str
    split: str
    phash: str
    caption: dict
    prompt_version: str
    generator_family: str
    generator_model_id: str
    generator_revision: str
    generation: dict
    norm_policy: str
    topic_match: bool = True
    composition_match: bool = True
    lighting_match: bool = True
    notes: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class MirrorStats:
    attempted: int = 0
    accepted: int = 0
    rejected: Counter = None

    def __post_init__(self) -> None:
        if self.rejected is None:
            self.rejected = Counter()

    def as_dict(self) -> dict:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": dict(self.rejected),
            "acceptance_rate": round(self.accepted / self.attempted, 4) if self.attempted else 0.0,
        }


def check_validity(
    candidate: bytes,
    source: bytes,
    prompt: str,
    policy: ValidityPolicy,
    image_similarity: Similarity,
    caption_similarity: CaptionSimilarity,
) -> str | None:
    """Return a rejection reason, or None if the candidate is a usable mirror."""
    match = caption_similarity(candidate, prompt)
    if match < policy.caption_match_min:
        return "ignored_prompt"

    similarity = image_similarity(candidate, source)
    if similarity < policy.source_similarity_min:
        return "not_a_mirror"
    if similarity > policy.source_similarity_max:
        # Deliberately its own reason rather than folded into "too different". This one is
        # a contamination signal and should be counted separately, because a rising rate
        # means the generator is reproducing training images.
        return "reproduced_source"
    return None


def generate_mirrors(
    sources: Sequence[tuple[str, bytes]],
    captioner,
    generator,
    image_similarity: Similarity,
    caption_similarity: CaptionSimilarity,
    spec: GenerationSpec = GenerationSpec(),
    policy: ValidityPolicy = ValidityPolicy(),
    norm_policy: NormalizationPolicy = POLICY_V1,
    batch_size: int = 16,
) -> tuple[list[ImageMirror], list[tuple[str, bytes]], MirrorStats]:
    """Caption, generate, validate. Returns records, normalised image bytes, and stats.

    Batched, because a diffusion pipeline called one image at a time wastes most of the
    GPU. The text track measured that cost: one prompt per call turned a three-hour run
    into a 130-hour one.
    """
    stats = MirrorStats(attempted=len(sources))
    records: list[ImageMirror] = []
    images: list[tuple[str, bytes]] = []

    prepared: list[tuple[str, bytes, Caption, str]] = []
    for image_id, raw in sources:
        try:
            caption = captioner.describe(raw)
            validate_caption(caption)
            prompt = to_prompt(caption)
        except CaptionRejected:
            stats.rejected["caption_rejected"] += 1
            continue
        except Exception:
            stats.rejected["caption_failed"] += 1
            continue
        prepared.append((image_id, raw, caption, prompt))

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        try:
            candidates = generator.generate([p for _, _, _, p in chunk], spec)
        except Exception:
            stats.rejected["generation_failed"] += len(chunk)
            continue

        for (image_id, raw, caption, prompt), candidate in zip(chunk, candidates, strict=True):
            try:
                normalised = normalize_bytes(candidate, norm_policy)
            except Exception:
                stats.rejected["normalisation_failed"] += 1
                continue

            reason = check_validity(
                normalised, raw, prompt, policy, image_similarity, caption_similarity
            )
            if reason is not None:
                stats.rejected[reason] += 1
                continue

            fingerprint = dhash(normalised)
            sample_id = f"mirror_{image_id}"
            images.append((sample_id, normalised))
            records.append(
                ImageMirror(
                    sample_id=sample_id,
                    source_image_id=image_id,
                    # Inherited, never reassigned: the mirror shares its source's group so
                    # the two cannot land in different splits.
                    source_group_id=image_id,
                    split=assign_split(image_id).value,
                    phash=to_hex(fingerprint),
                    caption=caption.as_dict(),
                    prompt_version=MIRROR_VERSION,
                    generator_family=generator.family,
                    generator_model_id=generator.model_id,
                    generator_revision=generator.revision,
                    generation=spec.as_dict(),
                    norm_policy=norm_policy.version,
                    notes={"captioner": getattr(captioner, "name", "unknown")},
                )
            )
            stats.accepted += 1

    return records, images, stats


def assert_split_inheritance(
    sources: Sequence[tuple[str, bytes]], mirrors: Sequence[ImageMirror]
) -> None:
    """A mirror must sit in its source's split. Checked, not trusted.

    The text track has the identical assertion, and it exists because this is the failure
    that inflates every metric while looking like success.
    """
    for mirror in mirrors:
        expected = assign_split(mirror.source_image_id).value
        if mirror.split != expected:
            raise RuntimeError(
                f"{mirror.sample_id} is in split {mirror.split} but its source belongs in "
                f"{expected}. Training on this would produce inflated metrics."
            )
