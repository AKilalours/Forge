"""What FORGE publishes about an image corpus: everything except the pixels.

THE RULE, inherited unchanged from the text track. Open Images lists its images as CC BY
2.0 while the maintainers explicitly disclaim any warranty about the license status of any
individual image, and say the user must verify each one. FORGE therefore treats the license
string as a CLAIM CARRIED FROM THE SOURCE, not as a fact it has established, and never
redistributes the pixels.

What is published is a manifest: ids, source URLs, the recorded license and attribution,
a perceptual hash, the normalisation policy version, and the split assignment. A reader can
reconstruct the corpus from it, and the reconstruction is verifiable because the hashes
must match. Nothing in the manifest is anyone's copyrighted content.

This is not legal caution for its own sake. A project shown to a company is partly a claim
about how its author handles data they do not own.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forge.common.splits import Split, assign_split
from forge.image.normalize import POLICY_VERSION

MANIFEST_VERSION = "image_manifest_v1"


@dataclass(frozen=True)
class ImageRecord:
    """One human image, described without reproducing it."""

    image_id: str
    source: str                  # e.g. "open_images_v7"
    source_url: str
    license: str                 # AS RECORDED UPSTREAM, not as verified by us
    attribution: str             # author string as recorded upstream
    phash: str                   # hex dhash of the NORMALISED bytes
    width: int                   # of the original, before normalisation
    height: int
    original_format: str
    norm_policy: str = POLICY_VERSION
    split: str = ""
    notes: dict = field(default_factory=dict)

    def with_split(self) -> ImageRecord:
        """Assign the split from the image id, using the text track's hashing unchanged.

        The grouping key is the image id, so a photograph, every mirror generated from it
        and any composite built on it all land together. That is the same leakage rule the
        text track enforces, and reusing the function rather than reimplementing it is the
        payoff for having written it modality-agnostically.
        """
        return ImageRecord(**{**asdict(self), "split": assign_split(self.image_id).value})


def _assert_no_pixels(record: ImageRecord) -> None:
    """A manifest that carries image data is not a manifest.

    Cheap guard against a future field like `thumbnail_b64` being added for convenience
    and quietly turning a metadata file into redistribution.
    """
    for key, value in asdict(record).items():
        if isinstance(value, (bytes, bytearray)):
            raise ValueError(f"field {key!r} carries raw bytes; manifests never do")
        if isinstance(value, str) and len(value) > 4096:
            raise ValueError(
                f"field {key!r} is {len(value)} characters. Nothing in a manifest should "
                "be that large; this looks like embedded image data."
            )


def write_manifest(records: list[ImageRecord], path: str | Path) -> dict:
    """Write JSONL and return a summary. One record per line, sorted by id.

    Sorted so two runs over the same corpus produce byte-identical files and a diff means
    the corpus changed, not that the filesystem returned things in another order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(records, key=lambda r: r.image_id)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    licenses: dict[str, int] = {}

    with path.open("w", encoding="utf-8") as fh:
        for record in ordered:
            _assert_no_pixels(record)
            if record.image_id in seen:
                raise ValueError(f"duplicate image_id {record.image_id!r} in manifest")
            seen.add(record.image_id)
            if not record.split:
                raise ValueError(
                    f"{record.image_id!r} has no split. Call with_split() before writing; "
                    "an unsplit manifest cannot be checked for leakage."
                )
            counts[record.split] = counts.get(record.split, 0) + 1
            licenses[record.license] = licenses.get(record.license, 0) + 1
            fh.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    return {
        "manifest_version": MANIFEST_VERSION,
        "norm_policy": POLICY_VERSION,
        "images": len(ordered),
        "splits": counts,
        "licenses": licenses,
        "path": str(path),
    }


def read_manifest(path: str | Path) -> list[ImageRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(ImageRecord(**json.loads(line)))
    return records


def assert_split_disjoint(records: list[ImageRecord]) -> None:
    """No image id may appear in two splits. Re-checked rather than trusted."""
    placement: dict[str, str] = {}
    for record in records:
        prior = placement.setdefault(record.image_id, record.split)
        if prior != record.split:
            raise RuntimeError(
                f"{record.image_id!r} appears in both {prior} and {record.split}. "
                "Training on this corpus would produce inflated metrics."
            )


def split_counts(records: list[ImageRecord]) -> dict[str, int]:
    out: dict[str, int] = {s.value: 0 for s in Split}
    for record in records:
        out[record.split] = out.get(record.split, 0) + 1
    return out
