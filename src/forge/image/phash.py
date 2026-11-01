"""Perceptual hashing for near-duplicate image detection.

WHY NOT A CONTENT HASH. The text track deduplicates with exact hashes plus MinHash over
shingles. Neither transfers: two JPEG encodings of the same photograph share no bytes, and
an image has no tokens to shingle. Re-encoding, resizing and requantising all change every
byte while changing nothing a person would notice.

Near-duplicates matter here for the same reason they do in text. If a photograph appears
twice and one copy is mirrored into the AI class, the detector sees near-identical content
on both sides of the label and learns whichever incidental difference distinguishes them.
Worse, duplicates that straddle a split inflate every metric.

DHASH, not average hash. Average hash compares each pixel to the image mean, which makes it
sensitive to global brightness and blind to structure. Dhash compares each pixel to its
right-hand neighbour, so it encodes gradients: it survives brightness and contrast changes,
JPEG requantisation and moderate resizing, which are exactly the transformations an image
picks up on its way through a hosting pipeline.

Implemented here rather than imported so the bit layout is fixed and reproducible. A
library upgrade must never silently change what "duplicate" means in a frozen dataset.
"""

from __future__ import annotations

import io

HASH_SIZE = 8              # 8x9 grayscale grid -> 64 comparisons -> 64 bits
DEFAULT_MAX_DISTANCE = 6   # of 64 bits; ~90% agreement


def dhash(raw: bytes, hash_size: int = HASH_SIZE) -> int:
    """Return a 64-bit gradient hash as an int.

    Reduce to (hash_size+1) x hash_size grayscale, then set one bit per horizontal
    neighbour pair: 1 where the left pixel is brighter than the right.
    """
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as img:
        small = img.convert("L").resize(
            (hash_size + 1, hash_size), Image.Resampling.LANCZOS
        )
        # get_flattened_data replaces getdata in Pillow 14; keep both so the hash is
        # identical across versions. A dedup rule that changes with a library upgrade
        # would silently redefine what "duplicate" means inside a frozen dataset.
        reader = getattr(small, "get_flattened_data", None) or small.getdata
        pixels = list(reader())

    bits = 0
    for row in range(hash_size):
        base = row * (hash_size + 1)
        for col in range(hash_size):
            bits <<= 1
            if pixels[base + col] > pixels[base + col + 1]:
                bits |= 1
    return bits


def to_hex(value: int, hash_size: int = HASH_SIZE) -> str:
    """Fixed-width hex, so manifests sort and compare predictably."""
    return f"{value:0{hash_size * hash_size // 4}x}"


def distance(a: int, b: int) -> int:
    """Hamming distance in bits. 0 is identical, 64 is maximally different."""
    return bin(a ^ b).count("1")


def is_near_duplicate(a: int, b: int, max_distance: int = DEFAULT_MAX_DISTANCE) -> bool:
    return distance(a, b) <= max_distance


class DuplicateIndex:
    """First-wins near-duplicate filter over a stream of images.

    Linear scan. At the v1 scale of 20,000 images that is 200M integer XORs, a few seconds,
    and it is exactly reproducible. A BK-tree or LSH index becomes worth the extra moving
    parts an order of magnitude later, and not before.
    """

    def __init__(self, max_distance: int = DEFAULT_MAX_DISTANCE) -> None:
        self.max_distance = max_distance
        self._seen: list[tuple[int, str]] = []

    def add(self, value: int, image_id: str) -> str | None:
        """Return the id of the image this duplicates, or None if it is new and kept."""
        for existing, existing_id in self._seen:
            if distance(existing, value) <= self.max_distance:
                return existing_id
        self._seen.append((value, image_id))
        return None

    def __len__(self) -> int:
        return len(self._seen)
