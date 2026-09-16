"""Turn a list of upstream image ids into a normalised corpus and a manifest.

WHAT THIS IS NOT. It is not a scraper. It reads a list of ids and license metadata that
the upstream dataset publishes, fetches exactly those, and records what it was told about
each one. Nothing is discovered, followed or crawled.

THREE THINGS THAT SOUND LIKE PLUMBING AND ARE NOT.

1. Normalisation happens HERE, at fetch time, before anything is stored. If images were
   stored as downloaded and normalised later, some stage would eventually read the raw
   ones, and the difference between raw and normalised is precisely the source signature
   the whole track is built to remove. Normalising once, at the boundary, means a raw
   photograph never exists inside FORGE.

2. The perceptual hash is computed on the NORMALISED bytes. Hashing the original would let
   two copies of one photograph, stored at different sizes, pass as distinct.

3. Failures are recorded rather than skipped. A corpus that quietly shrinks when a URL rots
   is a corpus whose size depends on the day it was built. The reasons are counted and
   returned so the manifest's summary can state how many images were requested against how
   many were obtained.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from forge.image.manifest import ImageRecord
from forge.image.normalize import POLICY_V1, ImageTooSmallError, NormalizationPolicy, describe
from forge.image.phash import DuplicateIndex, dhash, to_hex

# Deliberately conservative. This fetches from a third party's infrastructure using their
# published id list; being a considerate client is part of using open data properly.
DEFAULT_DELAY_SECONDS = 0.05
DEFAULT_ATTEMPTS = 3


@dataclass(frozen=True)
class SourceImage:
    """One row of the upstream index. License and author are CLAIMS from that index."""

    image_id: str
    url: str
    license: str
    attribution: str


@dataclass
class FetchStats:
    requested: int = 0
    stored: int = 0
    rejected: Counter = None          # reason -> count

    def __post_init__(self) -> None:
        if self.rejected is None:
            self.rejected = Counter()

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "stored": self.stored,
            "rejected": dict(self.rejected),
            "keep_rate": round(self.stored / self.requested, 4) if self.requested else 0.0,
        }


Downloader = Callable[[str], bytes]


def http_downloader(timeout: float = 20.0) -> Downloader:
    """The real downloader. Separated so the pipeline can be tested without a network."""

    def download(url: str) -> bytes:
        import urllib.request

        request = urllib.request.Request(url, headers={"User-Agent": "forge-image/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    return download


def fetch_corpus(
    sources: Sequence[SourceImage],
    download: Downloader,
    store: Callable[[str, bytes], None],
    policy: NormalizationPolicy = POLICY_V1,
    attempts: int = DEFAULT_ATTEMPTS,
    delay: float = DEFAULT_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[list[ImageRecord], FetchStats]:
    """Fetch, normalise, deduplicate, and return manifest rows plus honest statistics.

    `store` receives the NORMALISED bytes. It is injected so the caller decides where
    pixels live, and so this function has no opinion about a filesystem it does not own.

    Deduplication is first-wins over the whole run, which makes the result depend on the
    order of `sources`. That order is the caller's deterministic sample, so the outcome is
    reproducible; it would not be if this shuffled internally.
    """
    records: list[ImageRecord] = []
    stats = FetchStats(requested=len(sources))
    index = DuplicateIndex()

    for source in sources:
        raw = _download_with_retry(source.url, download, attempts, delay, sleep)
        if raw is None:
            stats.rejected["download_failed"] += 1
            continue

        try:
            original = describe(raw)
        except Exception:
            stats.rejected["undecodable"] += 1
            continue

        try:
            normalised = normalise_or_reject(raw, policy)
        except ImageTooSmallError:
            stats.rejected["too_small"] += 1
            continue
        except Exception:
            stats.rejected["normalisation_failed"] += 1
            continue

        fingerprint = dhash(normalised)
        duplicate_of = index.add(fingerprint, source.image_id)
        if duplicate_of is not None:
            stats.rejected["near_duplicate"] += 1
            continue

        store(source.image_id, normalised)
        records.append(
            ImageRecord(
                image_id=source.image_id,
                source="open_images_v7",
                source_url=source.url,
                license=source.license,
                attribution=source.attribution,
                phash=to_hex(fingerprint),
                width=original["width"],
                height=original["height"],
                original_format=original["format"] or "unknown",
                norm_policy=policy.version,
            ).with_split()
        )
        stats.stored += 1

    return records, stats


def normalise_or_reject(raw: bytes, policy: NormalizationPolicy) -> bytes:
    from forge.image.normalize import normalize_bytes

    return normalize_bytes(raw, policy)


def _download_with_retry(
    url: str,
    download: Downloader,
    attempts: int,
    delay: float,
    sleep: Callable[[float], None],
) -> bytes | None:
    """Retry transient failures, then give up and let the caller record it.

    Retries are bounded and the backoff is linear rather than aggressive. The text track
    measured that its generation retries recovered only a fifth of failures because the
    failures were systematic; the same scepticism applies here. A URL that has rotted will
    not un-rot on the third attempt.
    """
    for attempt in range(attempts):
        try:
            data = download(url)
            if data:
                return data
        except Exception:
            pass
        if attempt < attempts - 1:
            sleep(delay * (attempt + 1))
    return None


def sample_ids(sources: Iterable[SourceImage], n: int, salt: str = "image-corpus-v1") -> list[SourceImage]:
    """Choose n images deterministically, independent of the index's own ordering.

    The upstream index is grouped by class and split, so taking a prefix would take a
    biased slice, which is the mistake the text corpus loader made twice.
    """
    import hashlib

    ordered = sorted(
        sources, key=lambda s: hashlib.sha256(f"{salt}:{s.image_id}".encode()).hexdigest()
    )
    return ordered[:n]
