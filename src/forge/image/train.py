"""The image training loop, and the sample index it trains on.

WHAT THIS FILE IS FOR. Everything upstream of here produces PIECES: a manifest of
photographs, mirrors, unmatched generations, composites with masks. Assembling those into a
training set is where the experiment is actually decided, and it is where the mistakes are
silent. This module does the assembly explicitly and asserts the properties the comparison
depends on, rather than leaving them to be true by accident.

THE FOUR DECISIONS THAT CHANGE RESULTS

1. THE ARMS DIFFER IN ONE PLACE ONLY. Arm A trains on unmatched generations, Arm B on
   mirrors. Both draw the SAME human photographs and the SAME composites. If the human pool
   differed between arms, any gap would be a data gap rather than a treatment effect, and
   the writeup's central claim would be unsupported. `assert_arms_comparable` checks the
   shared parts are identical by id, not merely equal in count. Equal counts are the check
   that looks right and proves nothing.

2. THE AI BUDGET IS EQUALISED BY TRUNCATION, DETERMINISTICALLY. The mirror arm rejects more
   than the control arm does, so it finishes smaller. Training the control on more data and
   then reporting that it did better would be measuring dataset size. Both arms are cut to
   the smaller AI count, and which samples survive the cut is chosen by hash so the cut is
   reproducible and independent of the order files were written in.

3. COMPOSITES BELONG TO NEITHER ARM. They are the only localisation supervision that exists,
   so withholding them from one arm would cripple its local head for reasons unrelated to
   the treatment. They enter both arms unchanged and are excluded from the AI budget count,
   because they are not the treatment.

4. THE THRESHOLD IS FIT ON VAL AND APPLIED TO TEST, ONCE. Choosing the operating point on
   the split you report is how an FPR-at-budget number becomes fiction.

THE LEAKAGE RULE, restated because this is where it would break. A photograph, every mirror
generated from it, every composite built on it, share `group_id` and therefore a split. That
is enforced upstream by `assign_split(image_id)`; it is RE-CHECKED here, on the assembled
index, because an assembly bug is exactly the kind that upstream tests cannot see.

torch is imported lazily so this module imports, and the index logic tests, on a machine
with no GPU and no torch.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

TRAIN_VERSION = "image_train_v1"

# Sample kinds. `arm` says which experimental arm a sample belongs to; SHARED samples appear
# in both, so a difference between arms can only come from the treatment kinds.
SHARED = "shared"
ARM_MIRROR = "mirror"
ARM_UNMATCHED = "unmatched"

KIND_HUMAN = "human"
KIND_MIRROR = "mirror"
KIND_UNMATCHED = "unmatched"
KIND_COMPOSITE_AI = "composite_ai"
KIND_COMPOSITE_HUMAN = "composite_human"

_ARM_OF_KIND = {
    KIND_HUMAN: SHARED,
    KIND_COMPOSITE_AI: SHARED,
    KIND_COMPOSITE_HUMAN: SHARED,
    KIND_MIRROR: ARM_MIRROR,
    KIND_UNMATCHED: ARM_UNMATCHED,
}

# Only these carry a per-pixel ground truth. Everything else is whole-image, and gets a
# constant mask (see decision 2 in model.py).
_HAS_MASK = {KIND_COMPOSITE_AI, KIND_COMPOSITE_HUMAN}


class IndexAssemblyError(RuntimeError):
    """Raised when the assembled dataset violates a property the experiment depends on."""


@dataclass(frozen=True)
class ImageSample:
    """One training item, described without carrying pixels.

    `pixels_ref` and `mask_ref` are opaque keys handed to a reader callable. Keeping the
    bytes out of the index means the index can be written to disk, diffed, and inspected,
    and means no part of the corpus is redistributed by accident.
    """

    sample_id: str
    group_id: str
    split: str
    label: int          # 1 = contains generated content
    kind: str
    pixels_ref: str
    mask_ref: str | None = None
    notes: dict = field(default_factory=dict)

    @property
    def arm(self) -> str:
        return _ARM_OF_KIND[self.kind]

    def __post_init__(self) -> None:
        if self.kind not in _ARM_OF_KIND:
            raise ValueError(f"unknown sample kind {self.kind!r}")
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")
        if self.kind in _HAS_MASK and not self.mask_ref:
            raise ValueError(
                f"{self.sample_id!r} is a composite with no mask_ref. Composites are the "
                "only localisation supervision there is; one without a mask is a silent "
                "hole in the local head's training signal."
            )


def _rank(sample_id: str) -> str:
    """Deterministic order for truncation. Independent of read order, stable across runs."""
    return hashlib.sha256(f"image-budget-v1:{sample_id}".encode()).hexdigest()


def cap_by_budget(samples: Sequence[ImageSample], budget: int) -> list[ImageSample]:
    """Keep exactly `budget` samples, allotted across splits, chosen by hash.

    Per split rather than globally: a global cut would drift the split proportions, and val
    would then measure a different corpus mix than train.

    EXACTLY `budget`, by largest-remainder allocation. An earlier draft rounded each split
    independently, which lands on budget +/- 1 depending on the split mix. The two arms have
    slightly different split mixes, so their rounding errors differ, and `assert_arms_comparable`
    would then fail on arms that were correctly budgeted. Worse, if it did not fail, the arms
    would differ by a sample or two for a reason nobody could name.
    """
    if budget < 0:
        raise ValueError("budget cannot be negative")
    if budget >= len(samples):
        return list(samples)

    by_split: dict[str, list[ImageSample]] = {}
    for sample in samples:
        by_split.setdefault(sample.split, []).append(sample)

    total = len(samples)
    exact = {split: budget * len(group) / total for split, group in by_split.items()}
    share = {split: int(value) for split, value in exact.items()}
    # Hand out the remaining places to the largest fractional parts, splits tie-broken by
    # name so the allocation does not depend on dict order.
    remaining = budget - sum(share.values())
    for split in sorted(exact, key=lambda s: (-(exact[s] - share[s]), s))[:remaining]:
        share[split] += 1

    keep: set[str] = set()
    for split, group in by_split.items():
        ordered = sorted(group, key=lambda s: _rank(s.sample_id))
        keep.update(s.sample_id for s in ordered[: share[split]])
    return [s for s in samples if s.sample_id in keep]


def ai_budget(samples: Iterable[ImageSample]) -> int:
    """Count the treatment samples only. Composites are supervision, not treatment."""
    return sum(1 for s in samples if s.arm in (ARM_MIRROR, ARM_UNMATCHED))


def build_arm(
    shared: Sequence[ImageSample],
    treatment: Sequence[ImageSample],
    budget: int | None = None,
) -> list[ImageSample]:
    """One arm: the shared pool plus its own generated samples, capped to `budget`."""
    for sample in shared:
        if sample.arm != SHARED:
            raise IndexAssemblyError(
                f"{sample.sample_id!r} is a {sample.arm} sample in the shared pool. The "
                "shared pool is what makes the arms comparable; a treatment sample in it "
                "would appear in both arms."
            )
    arms = {s.arm for s in treatment}
    if len(arms) > 1:
        raise IndexAssemblyError(f"treatment pool mixes arms {sorted(arms)}; an arm trains on one")
    capped = treatment if budget is None else cap_by_budget(treatment, budget)
    return list(shared) + list(capped)


def assert_arms_comparable(arm_a: Sequence[ImageSample], arm_b: Sequence[ImageSample]) -> None:
    """The comparison is only meaningful if these hold. Checked by id, not by count."""
    shared_a = {s.sample_id for s in arm_a if s.arm == SHARED}
    shared_b = {s.sample_id for s in arm_b if s.arm == SHARED}
    if shared_a != shared_b:
        only_a, only_b = sorted(shared_a - shared_b)[:3], sorted(shared_b - shared_a)[:3]
        raise IndexAssemblyError(
            f"the arms do not share the same human and composite pool: {len(shared_a - shared_b)} "
            f"only in A (e.g. {only_a}), {len(shared_b - shared_a)} only in B (e.g. {only_b}). "
            "Any measured gap would include a data difference."
        )
    budget_a, budget_b = ai_budget(arm_a), ai_budget(arm_b)
    if budget_a != budget_b:
        raise IndexAssemblyError(
            f"unequal generated budgets: {budget_a} vs {budget_b}. The larger arm would win "
            "on data volume and the result would say nothing about the treatment."
        )
    if not budget_a:
        raise IndexAssemblyError("both arms have zero generated samples; there is nothing to compare")


def assert_no_group_leakage(samples: Iterable[ImageSample]) -> None:
    """No group may straddle two splits. Re-checked on the assembled index."""
    placement: dict[str, tuple[str, str]] = {}
    for sample in samples:
        prior = placement.setdefault(sample.group_id, (sample.split, sample.sample_id))
        if prior[0] != sample.split:
            raise IndexAssemblyError(
                f"group {sample.group_id!r} appears in {prior[0]} (as {prior[1]!r}) and in "
                f"{sample.split} (as {sample.sample_id!r}). A photograph and the images "
                "derived from it are in two splits, so test scores are inflated."
            )


def assert_both_classes(samples: Sequence[ImageSample], split: str) -> None:
    """A split with one class makes AUROC undefined and FPR meaningless."""
    labels = {s.label for s in samples if s.split == split}
    if labels != {0, 1}:
        raise IndexAssemblyError(
            f"the {split} split carries labels {sorted(labels)}; both classes are required"
        )


def index_summary(samples: Sequence[ImageSample]) -> dict:
    kinds = Counter(s.kind for s in samples)
    splits = Counter(s.split for s in samples)
    per_split_label = {
        split: dict(Counter(s.label for s in samples if s.split == split))
        for split in sorted(splits)
    }
    return {
        "version": TRAIN_VERSION,
        "samples": len(samples),
        "kinds": dict(sorted(kinds.items())),
        "splits": dict(sorted(splits.items())),
        "labels_per_split": per_split_label,
        "ai_budget": ai_budget(samples),
        "groups": len({s.group_id for s in samples}),
        "with_mask": sum(1 for s in samples if s.mask_ref),
    }


def write_index(samples: Sequence[ImageSample], path: str | Path) -> dict:
    """Persist an assembled index, sorted, so two runs of the same config diff empty."""
    assert_no_group_leakage(samples)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in sorted(samples, key=lambda s: s.sample_id):
            fh.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
    return {**index_summary(samples), "path": str(path)}


def read_index(path: str | Path) -> list[ImageSample]:
    out = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(ImageSample(**json.loads(line)))
    return out


# --------------------------------------------------------------------------------------
# The torch side. Everything below needs torch; everything above does not.
# --------------------------------------------------------------------------------------


def make_dataset(
    samples: Sequence[ImageSample],
    read_pixels: Callable[[str], bytes],
    read_mask: Callable[[str], bytes] | None,
    image_size: int,
):
    """A torch Dataset over an index.

    Readers are injected rather than imported so the loop is testable without a corpus on
    disk, and so the same loop can read from a local cache, a tar shard, or a fake.

    A whole-image sample gets a CONSTANT mask: all ones if it is generated, all zeros if it
    is a photograph. That is not a placeholder, it is the correct ground truth, and it is
    what keeps the local head sane on ordinary inputs (model.py, decision 2).
    """
    import io

    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    class ForgeImageDataset(Dataset):
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int):
            sample = samples[index]
            with Image.open(io.BytesIO(read_pixels(sample.pixels_ref))) as img:
                img = img.convert("RGB").resize((image_size, image_size))
                pixels = torch.from_numpy(_as_array(img).copy()).permute(2, 0, 1).float() / 255.0

            if sample.mask_ref and read_mask is not None:
                with Image.open(io.BytesIO(read_mask(sample.mask_ref))) as raw:
                    # NEAREST, not the default smooth filter. A stored mask is deliberately
                    # unfeathered (composites.py) so that IoU does not depend on an arbitrary
                    # threshold. Resampling it bicubically puts a soft ramp back along every
                    # boundary, and a bicubic kernel overshoots, so mask values leave [0, 1]
                    # entirely. The mask is a hard fact about which pixels came from the
                    # patch, and resizing must not turn it into an opinion.
                    grey = raw.convert("L").resize(
                        (image_size, image_size), Image.Resampling.NEAREST
                    )
                    mask = torch.from_numpy(_as_array(grey).copy()).float() / 255.0
            else:
                mask = torch.full((image_size, image_size), float(sample.label))

            return {
                "pixel_values": pixels,
                "mask": mask,
                "label": torch.tensor(float(sample.label)),
            }

    return ForgeImageDataset()


def _as_array(img):
    import numpy as np

    return np.asarray(img)


def patch_iou(patch_logits, mask, grid: int, threshold: float = 0.5) -> float:
    """Localisation quality, scored at PATCH resolution.

    Scored where the head predicts, using the same downsampling the loss used. Scoring at
    pixel resolution instead would grade the upsampler, and would make the number look
    better or worse depending on an interpolation mode nobody chose deliberately.

    The mask is binarised at 0.5 patch coverage: a patch counts as evidence when most of it
    came from the pasted region.
    """
    import torch

    from forge.image.model import downsample_mask

    # downsample_mask takes a batch. Scoring happens one sample at a time, so add and drop
    # the batch axis here rather than making the model's contract ambiguous. The first draft
    # of this function passed a 2-D mask straight through and every call raised.
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    predicted = torch.sigmoid(patch_logits) >= threshold
    target = downsample_mask(mask, grid).squeeze(0) >= 0.5
    intersection = (predicted & target).sum().item()
    union = (predicted | target).sum().item()
    if not union:
        # Both empty: the head correctly found nothing in an image that contains nothing.
        return 1.0
    return intersection / union


@dataclass
class ImageEval:
    auroc: float
    fpr_at_budget: float
    fnr: float
    threshold: float
    localisation_iou: float      # composites only; nan when the split holds none
    n_human: int
    n_ai: int
    n_localised: int

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def score_split(model, loader, device, samples: Sequence[ImageSample], grid: int):
    """Run the model over a split and return raw probabilities plus localisation IoUs.

    Returns raw material rather than metrics, so the threshold can be fit on one split and
    applied to another without the model being run twice.
    """
    import numpy as np
    import torch

    model.eval()
    probabilities, labels, ious = [], [], []
    localisable = [s.kind in _HAS_MASK for s in samples]
    seen = 0
    with torch.no_grad():
        for batch in loader:
            pixels = batch["pixel_values"].to(device)
            out = model(pixels)
            batch_probabilities = torch.sigmoid(out["logit"].float()).cpu().numpy()
            probabilities.append(batch_probabilities)
            labels.append(batch["label"].cpu().numpy())
            for i in range(len(batch_probabilities)):
                if seen + i < len(localisable) and localisable[seen + i]:
                    ious.append(
                        patch_iou(out["patch_logits"][i].cpu(), batch["mask"][i], grid)
                    )
            seen += len(batch_probabilities)
    return np.concatenate(probabilities), np.concatenate(labels), ious


def summarise(probabilities, labels, ious, threshold: float) -> ImageEval:
    from forge.evaluation import metrics as M

    return ImageEval(
        auroc=float(M.auroc(labels, probabilities)),
        fpr_at_budget=float(M.false_positive_rate(labels, probabilities, threshold)),
        fnr=float(M.false_negative_rate(labels, probabilities, threshold)),
        threshold=float(threshold),
        localisation_iou=float(sum(ious) / len(ious)) if ious else float("nan"),
        n_human=int((labels == 0).sum()),
        n_ai=int((labels == 1).sum()),
        n_localised=len(ious),
    )
