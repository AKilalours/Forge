"""Generation must hold one engine at a time and must batch.

THE TWO BUGS THIS CAME FROM, both invisible without a GPU.

1. ENGINES HELD CONCURRENTLY. Documents are assigned to generator families round-robin,
   and the runner cached a generator per family, so every family's engine was alive at
   once. A vLLM engine reserves ~90% of the card at startup, so the second one found
   1.03 GiB free of 23.53 and the run died with

       ValueError: Free memory on device (1.03/23.53 GiB) on startup is less than
       desired GPU memory utilization (0.9, 21.17 GiB)

2. ONE PROMPT PER CALL. The runner called generate([single_prompt]) once per document.
   That is correct and unusable: a 640-token completion from a 3B model decodes at
   roughly 60-100 tok/s, about 8 seconds a document, so 60k documents is over 130 hours.
   Continuous batching is the only reason vLLM is on the generation side at all, and
   one-at-a-time never engages it.

Neither shows up with the fake backend, where generators are free and instant, which is
why a green suite said nothing about either. These tests assert the SHAPE of the calls
rather than their speed, so they run without a GPU.
"""

from __future__ import annotations

import pytest

from forge.common.schemas import Split
from forge.generation import run as run_mod
from forge.generation.generators.base import Decoding
from forge.generation.run import HumanRef, generate_mirrors

ROSTER = {
    "decoding_grid": {"temperature": [0.7], "top_p": [0.9], "max_new_tokens": 128},
    "families": [
        {
            "family": f,
            "provider": "open_source",
            "model_id": f"vendor/{f}",
            "revision": "a" * 40,
            "role": "held_in",
            "output_redistributable": True,
        }
        for f in ("alpha", "beta", "gamma")
    ],
}

MIRROR_CFG = {
    "prompt_version": "mirror_v1",
    "prompt_path": "src/forge/generation/prompts/mirror_v1.txt",
    # Wide ratios on purpose: these tests are about SCHEDULING, not about validation.
    # With the defaults the recorder's fixed-length output scored a 1.84 length ratio
    # against the fixture humans and every document was rejected, leaving nothing to
    # assert on. A fixture that silently produces zero documents makes several of these
    # tests vacuously pass, which is worse than failing.
    "validation": {"max_retries": 0, "length_ratio_min": 0.1, "length_ratio_max": 10.0},
}


def _humans(n: int) -> list[HumanRef]:
    body = (
        "The committee met on a Tuesday and reviewed the quarterly figures in detail. "
        "Several members raised concerns about the revised timetable for the works. "
        "A vote was taken and the proposal carried by a comfortable margin.\n\n"
        "Afterwards the chair summarised the decisions and set a date to reconvene. "
        "The minutes record that two members abstained without stating a reason."
    )
    return [
        HumanRef(f"doc{i:04d}", f"grp{i:04d}", f"{body} Item {i}.", "web", Split.TRAIN)
        for i in range(n)
    ]


class _Recorder:
    """Stands in for a real backend and records how it was driven.

    `live` counts instances that have been built and not yet closed. Class-level, so a
    second engine built before the first closes is directly observable.
    """

    live = 0
    peak_live = 0
    batch_sizes: list[int] = []
    build_order: list[str] = []

    @classmethod
    def reset(cls) -> None:
        cls.live = 0
        cls.peak_live = 0
        cls.batch_sizes = []
        cls.build_order = []

    def __init__(self, family: str, model_id: str, revision: str) -> None:
        self.family = family
        self.model_id = model_id
        self.revision = revision
        self._closed = False
        type(self).live += 1
        type(self).peak_live = max(type(self).peak_live, type(self).live)
        type(self).build_order.append(family)

    def generate(self, prompts: list[str], decoding: Decoding) -> list[str]:
        return self.generate_many(prompts, [decoding] * len(prompts))

    def generate_many(self, prompts: list[str], decodings: list[Decoding]) -> list[str]:
        assert not self._closed, "generated after close()"
        assert len(prompts) == len(decodings)
        type(self).batch_sizes.append(len(prompts))
        # Long enough to clear the validator's length-ratio floor.
        return [
            "A short account of the same subject follows below.\n\n"
            + " ".join(f"word{i}" for i in range(120))
            for _ in prompts
        ]

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            type(self).live -= 1


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[_Recorder]:
    _Recorder.reset()
    monkeypatch.setattr(
        run_mod,
        "build_generator",
        lambda spec, backend: _Recorder(spec.family, spec.model_id, spec.revision),
    )
    return _Recorder


def test_only_one_engine_is_alive_at_a_time(recorder: type[_Recorder]) -> None:
    """The regression. Two live engines is the OOM."""
    generate_mirrors(_humans(60), ROSTER, MIRROR_CFG, backend="vllm")
    assert recorder.peak_live == 1, (
        f"{recorder.peak_live} generators were alive at once; a vLLM engine reserves most "
        "of the GPU, so the second one cannot start"
    )


def test_every_engine_is_released(recorder: type[_Recorder]) -> None:
    generate_mirrors(_humans(60), ROSTER, MIRROR_CFG, backend="vllm")
    assert recorder.live == 0, "an engine was left holding GPU memory after the run"


def test_each_family_is_visited_exactly_once(recorder: type[_Recorder]) -> None:
    """Rebuilding a family's engine would pay the model load cost repeatedly."""
    generate_mirrors(_humans(60), ROSTER, MIRROR_CFG, backend="vllm")
    assert len(recorder.build_order) == len(set(recorder.build_order))


def test_prompts_are_sent_in_batches(recorder: type[_Recorder]) -> None:
    """The second regression. Batch size 1 is the 130-hour run."""
    generate_mirrors(_humans(60), ROSTER, MIRROR_CFG, backend="vllm")
    assert recorder.batch_sizes, "no generation calls were made"
    assert max(recorder.batch_sizes) > 1, (
        "every call carried a single prompt, which never engages continuous batching"
    )


def test_batches_do_not_exceed_the_configured_bound(recorder: type[_Recorder]) -> None:
    generate_mirrors(_humans(60), ROSTER, MIRROR_CFG, backend="vllm")
    assert max(recorder.batch_sizes) <= run_mod.GENERATION_BATCH


def test_the_fixture_actually_produces_documents(recorder: type[_Recorder]) -> None:
    """Guard the other tests against passing vacuously on an empty result."""
    result = generate_mirrors(_humans(60), ROSTER, MIRROR_CFG, backend="vllm")
    assert len(result.docs) == 60, result.stats


def test_output_order_follows_the_human_corpus(recorder: type[_Recorder]) -> None:
    """Grouping by family must not reorder the dataset.

    Without this, the parquet row order would depend on which family sorted first, and
    two runs of the same config could differ byte-for-byte.
    """
    humans = _humans(60)
    result = generate_mirrors(humans, ROSTER, MIRROR_CFG, backend="vllm")
    produced = [d.source_human_id for d in result.docs]
    assert produced == sorted(produced, key=lambda i: [h.doc_id for h in humans].index(i))


def test_family_assignment_is_unchanged_by_the_regrouping(recorder: type[_Recorder]) -> None:
    """The experiment depends on WHICH family generates each document.

    Reordering work must not change the assignment, or the dataset is not the one the
    config describes.
    """
    from forge.generation.assignment import assign_family, held_in_families, parse_roster

    families = held_in_families(parse_roster(ROSTER))
    humans = _humans(60)
    expected = {h.doc_id: assign_family(h.doc_id, families).family for h in humans}
    result = generate_mirrors(humans, ROSTER, MIRROR_CFG, backend="vllm")
    for d in result.docs:
        assert d.generator.family == expected[d.source_human_id]


def test_split_and_group_are_still_inherited(recorder: type[_Recorder]) -> None:
    """The leakage invariant, re-asserted after a refactor that moved every record."""
    humans = _humans(60)
    by_id = {h.doc_id: h for h in humans}
    result = generate_mirrors(humans, ROSTER, MIRROR_CFG, backend="vllm")
    assert result.docs
    for d in result.docs:
        h = by_id[d.source_human_id]
        assert d.split is h.split
        assert d.source_group_id == h.source_group_id


# ---------------------------------------- one family per invocation, for a 30 GB disk

def test_mirror_family_runs_reconstruct_the_single_run():
    """Same invariant the control arm holds, for the same reason and the same disk.

    The pod has 30 GB and no volume, four families are about 23 GB of weights, so the
    mirror run is split one family at a time with the weights deleted in between. That is
    only sound if the pieces add up to the unsplit run, which they do because the filter
    is applied after assignment and never to the roster.
    """
    humans = _humans(40)
    whole = generate_mirrors(humans, ROSTER, MIRROR_CFG, backend="fake")
    names = sorted(f["family"] for f in ROSTER["families"])
    pieces = [
        d
        for fam in names
        for d in generate_mirrors(humans, ROSTER, MIRROR_CFG, backend="fake",
                                  only_family=fam).docs
    ]
    assert sorted(d.sample_id for d in pieces) == sorted(d.sample_id for d in whole.docs)
    by_id = {d.sample_id: d for d in pieces}
    for d in whole.docs:
        assert by_id[d.sample_id].generator.family == d.generator.family
        assert by_id[d.sample_id].text == d.text


def test_a_filtered_mirror_run_is_a_share_not_the_whole_corpus():
    """Filtering the roster instead would have given one model every document."""
    humans = _humans(40)
    whole = generate_mirrors(humans, ROSTER, MIRROR_CFG, backend="fake")
    part = generate_mirrors(humans, ROSTER, MIRROR_CFG, backend="fake", only_family="alpha")
    assert 0 < len(part.docs) < len(whole.docs)
    assert {d.generator.family for d in part.docs} == {"alpha"}


def test_an_unknown_mirror_family_is_refused():
    """A typo must not write an empty part and exit 0."""
    with pytest.raises(ValueError, match="not a held-in family"):
        generate_mirrors(_humans(5), ROSTER, MIRROR_CFG, backend="fake", only_family="nope")
