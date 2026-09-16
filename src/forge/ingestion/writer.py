"""Parquet output and the dataset MANIFEST.

Two rules enforced here rather than left to discipline:

  A non-redistributable document's `text` is written only under data/, which is
  gitignored. The committed and publishable artifact is IDs plus metadata plus hashes.
  See data_spec_v1 section 8.

  A dataset without a MANIFEST is not a dataset version. The manifest pins
  spec_version, code_commit and per-source counts, which is what makes a training run
  a result rather than an anecdote.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from forge.common.hashing import content_sha256
from forge.common.schemas import HumanDocument
from forge.common.splits import SPLIT_SALT

# Read the data lake with this, never with a bare **/*.parquet.
PARTITION_GLOB = "source=*/split=*/*.parquet"


def _code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def _row(doc: HumanDocument) -> dict:
    d = doc.model_dump(mode="json", by_alias=True)
    q = d.pop("quality", {}) or {}
    d["edu_score"] = q.get("edu_score")
    d["pii_flags"] = ",".join(q.get("pii_flags", []))
    return d


def write_parquet(docs: list[HumanDocument], root: str | Path) -> dict[str, int]:
    """Partitioned by source and split. Returns rows written per partition."""
    root = Path(root)
    written: dict[str, int] = {}
    groups: dict[tuple[str, str], list[dict]] = {}
    for doc in docs:
        groups.setdefault((doc.source, doc.split.value), []).append(_row(doc))
    for (source, split), rows in groups.items():
        out = root / f"source={source}" / f"split={split}"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), out / "part-000.parquet", compression="zstd")
        written[f"{source}/{split}"] = len(rows)
    return written


def write_metadata_only(docs: list[HumanDocument], path: str | Path) -> int:
    """The publishable artifact for non-redistributable corpora: no `text` field.

    This deliberately does NOT live inside the partitioned Parquet tree. It has a
    different schema (no `text`), so a reader globbing the data lake would hit a schema
    conflict, and worse, a reader that tolerated it would silently union a partial
    table into the training data. Different artifact, different directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc in docs:
        if doc.redistributable:
            continue
        r = _row(doc)
        r.pop("text", None)
        rows.append(r)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    return len(rows)


def write_manifest(
    root: str | Path,
    dataset_version: str,
    docs: list[HumanDocument],
    stats_by_source: dict[str, dict],
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    by_source: Counter[str] = Counter(d.source for d in docs)
    by_split: Counter[str] = Counter(d.split.value for d in docs)

    sources = []
    for sid, n in sorted(by_source.items()):
        ids = sorted(d.doc_id for d in docs if d.source == sid)
        sources.append(
            {
                "id": sid,
                "n_docs": n,
                "sha256_of_ids": content_sha256("\n".join(ids)),
                "cleaning": stats_by_source.get(sid, {}),
            }
        )

    manifest = {
        "dataset_version": dataset_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _code_commit(),
        "spec_version": "data_spec_v1",
        "processing_version": "clean_v1",
        "split_salt": SPLIT_SALT,
        "sources": sources,
        "counts": {
            "human": len(docs),
            "ai": 0,
            "hard_negative": 0,
            "adversarial": 0,
            "by_split": dict(by_split),
        },
    }
    p = root / "MANIFEST.json"
    p.write_text(json.dumps(manifest, indent=2) + "\n")
    return p
