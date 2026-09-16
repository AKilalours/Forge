"""Constructors for the five evaluation regimes. Plan section 13, gap G3.

The regimes were configured and the lab could score them, but nothing built them from
data. These functions do that.

The rule every constructor obeys: **partition by group, never by row.** A held-out domain
or a held-out generation era still has to respect `source_group_id`, or a human document
lands in train while its mirror lands in the OOD test set and the "unseen domain" result
is measuring memorisation instead.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field


@dataclass
class RegimeSplit:
    name: str
    train_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    train_keys: list[str] = field(default_factory=list)
    test_keys: list[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "regime": self.name, "n_train": len(self.train_ids), "n_test": len(self.test_ids),
            "train_keys": sorted(self.train_keys), "test_keys": sorted(self.test_keys),
            "note": self.note,
        }


class RegimeError(ValueError):
    pass


def resolve_group_key(records: Sequence, key_fn: Callable, group_fn: Callable,
                      neutral: str = "") -> dict[str, str]:
    """One key per group, ignoring neutral values.

    A group is a human document plus its mirror. Taking the FIRST record's key is wrong:
    the human is usually first and carries family 'human' with no release date, so the
    mirror's generator family and release date would never be seen and the regime would
    hold out nothing. The group's identity for R3 and R4 comes from its AI member.

    A group with only neutral values (a human with no mirror) keeps the neutral key, and
    since 'human' is never a held-out family, those documents land in train.
    """
    out: dict[str, str] = {}
    for r in records:
        g, k = group_fn(r), str(key_fn(r))
        if g not in out or (out[g] == neutral and k != neutral):
            out[g] = k
    return out


def _partition(records: Sequence, key_fn: Callable, test_keys: set[str],
               id_fn: Callable, group_fn: Callable, name: str,
               neutral: str = "") -> RegimeSplit:
    group_key = resolve_group_key(records, key_fn, group_fn, neutral)

    test_groups = {g for g, k in group_key.items() if k in test_keys}
    split = RegimeSplit(name=name)
    for r in records:
        (split.test_ids if group_fn(r) in test_groups else split.train_ids).append(id_fn(r))
    split.test_keys = sorted(test_keys)
    split.train_keys = sorted({k for k in group_key.values()} - test_keys)

    if not split.test_ids:
        raise RegimeError(f"{name}: no records matched the held-out keys {sorted(test_keys)}")
    if not split.train_ids:
        raise RegimeError(f"{name}: the held-out keys cover the entire dataset")
    return split


def unseen_domain(records, test_domains: Iterable[str], id_fn=lambda r: r.doc_id,
                  domain_fn=lambda r: r.domain, group_fn=lambda r: r.source_group_id) -> RegimeSplit:
    """R2. Train on some domains, test on domains never seen."""
    s = _partition(records, domain_fn, {str(d) for d in test_domains}, id_fn, group_fn,
                   "R2_unseen_domain")
    overlap = set(s.train_keys) & set(s.test_keys)
    if overlap:
        raise RegimeError(f"R2: domains {sorted(overlap)} appear on both sides")
    return s


def unseen_generator(records, test_families: Iterable[str], id_fn=lambda r: r.doc_id,
                     family_fn=lambda r: r.generator_family,
                     group_fn=lambda r: r.source_group_id) -> RegimeSplit:
    """R3. Held-out generator families.

    Human records carry family 'human' and belong in BOTH sides: an unseen-generator test
    set made only of AI documents has no human examples, so its FPR is undefined and the
    result says nothing about the error that matters most.
    """
    test = {str(f) for f in test_families}
    if "human" in test:
        raise RegimeError(
            "R3: 'human' cannot be a held-out generator family. A test set with no human "
            "examples has an undefined FPR, and FPR is the metric that matters most."
        )
    # 'human' is the neutral key: a group takes its mirror's family, not its human's.
    return _partition(records, family_fn, test, id_fn, group_fn,
                      "R3_unseen_generator", neutral="human")


def temporal(records, cutoff: str, id_fn=lambda r: r.doc_id,
             released_fn=lambda r: r.generator_released,
             group_fn=lambda r: r.source_group_id) -> RegimeSplit:
    """R4. Train on generations from models released before `cutoff`, test on after.

    The date is the MODEL's release date, never the date FORGE generated the text.
    Splitting on generation date would produce a temporal regime that measures nothing,
    because all of it was generated in the same week.
    """
    def key(r):
        rel = str(released_fn(r) or "")
        if not rel:
            return "none"          # human records carry no release date
        return "after" if rel > cutoff else "before"

    s = _partition(records, key, {"after"}, id_fn, group_fn, "R4_temporal", neutral="none")
    s.note = f"cutoff {cutoff} on model release date, not generation date"
    return s


def summarize(records, domain_fn=lambda r: r.domain,
              family_fn=lambda r: r.generator_family) -> dict:
    """What is actually available, so regimes are configured against reality.

    Asking for a held-out domain the corpus does not contain fails loudly in _partition;
    this is how you find out before that happens.
    """
    return {
        "n": len(records),
        "domains": dict(Counter(domain_fn(r) for r in records).most_common()),
        "generator_families": dict(Counter(family_fn(r) for r in records).most_common()),
    }
