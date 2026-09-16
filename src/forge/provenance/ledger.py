"""Provenance ledger.

Every document carries source, license, acquisition date, processing version and
content hash. Where a corpus forbids redistribution, only these fields plus the
document id leave the machine; the text never does. See data_spec_v1 section 8.
"""

from __future__ import annotations


def redistributable_view(record: dict) -> dict:
    """Strip `text` from a record whose license forbids republication."""
    if record.get("redistributable"):
        return record
    return {k: v for k, v in record.items() if k != "text"}
