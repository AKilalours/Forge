"""Arm A: conventional random synthetic data. The control the whole project measures against.

This is the naive approach the research question is testing MIRRORS against: pick a topic,
ask a model to write about it, label it AI. No matching to any human document.

    mirror:  human doc -> extract topic/genre/length/structure -> prompt -> AI doc
    random:  topic list -> "write an article about X" -> AI doc

Getting this arm RIGHT is what makes the comparison honest, and there are three ways to
get it wrong that all flatter the mirror arm:

**Equal budget.** Arm A must produce the SAME number of AI documents as Arm B, from the
SAME generator families, over the SAME decoding grid. If mirrors win on volume or on
generator diversity, the result says nothing about matching.

**A fair prompt.** The random prompt has to be what a competent person would actually
write, not a strawman. If Arm A's prompts are deliberately poor, the comparison is
rigged. The template below asks for a specific genre and length, which is what anyone
building a detector this way would do.

**Realistic length distribution.** Random generation still has to produce documents in
the same length RANGE as the corpus, or the detector learns length and Arm A collapses
for a reason unrelated to topic matching. What it must NOT do is match any individual
human document's length, because per-document matching is precisely what mirrors add.

Topics are drawn from a fixed inventory, never from the human corpus. Sampling topics
from the corpus would make this a weak mirror rather than a random baseline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from forge.common.schemas import GeneratorSpec, Label, MirrorSpec, SyntheticDocument
from forge.common.splits import assign_split
from forge.generation.assignment import (
    assert_no_held_out,
    assign_decoding,
    assign_family,
    held_in_families,
    parse_roster,
)
from forge.generation.generators.base import Decoding
from forge.generation.mirror import ValidationStats, strip_wrapper

# Deliberately broad and mundane. Not derived from FineWeb, and not curated to be
# difficult. This is what a topic list looks like when someone builds it in an afternoon.
TOPICS = [
    "coastal erosion", "municipal budgeting", "beekeeping", "railway signalling",
    "medieval bookbinding", "soil chemistry", "harbour dredging", "choral notation",
    "textile dyeing", "glacier monitoring", "orchard grafting", "lighthouse keeping",
    "urban composting", "ferry timetabling", "dry stone walling", "kiln firing",
    "salt marsh restoration", "clock repair", "cheese ripening", "canal locks",
    "seed banking", "bell ringing", "peat cutting", "thatching", "wool spinning",
    "tidal energy", "archive cataloguing", "hedgerow management", "brick making",
    "well drilling", "sail making", "grain milling", "lace making", "quarrying",
    "rope walking", "charcoal burning", "reed cutting", "basket weaving",
    "school timetabling", "library lending", "waste collection", "street lighting",
    "flood defences", "cycle infrastructure", "allotment policy", "park maintenance",
]
GENRES = [
    "explainer", "formal report", "how-to guide", "historical overview",
    "opinion column", "encyclopedia entry", "news article", "review",
]
REGISTERS = ["informational", "formal", "conversational"]

PROMPT = """Write a {genre} in a {register} register about {topic}.

- Length: approximately {target_tokens} tokens.
- Write continuous prose unless the genre calls for another shape.

Output only the text itself. Do not add a preamble, a title line, a sign-off, or any
meta commentary."""


@dataclass
class RandomSpec:
    topic: str
    genre: str
    register: str
    target_tokens: int

    def prompt(self) -> str:
        return PROMPT.format(genre=self.genre, register=self.register, topic=self.topic,
                             target_tokens=self.target_tokens)


def _digest(key: str, salt: str) -> int:
    return int(hashlib.sha256(f"{key}|{salt}".encode()).hexdigest()[:12], 16)


def spec_for(index: int, length_pool: list[int]) -> RandomSpec:
    """Deterministic, so a partial run resumes to the same dataset.

    `length_pool` is the observed token-count distribution of the human corpus. Sampling
    a length from that pool keeps Arm A in the same length RANGE as Arm B without
    matching any individual document, which is the distinction the experiment turns on.
    """
    k = str(index)
    return RandomSpec(
        topic=TOPICS[_digest(k, "topic") % len(TOPICS)],
        genre=GENRES[_digest(k, "genre") % len(GENRES)],
        register=REGISTERS[_digest(k, "register") % len(REGISTERS)],
        target_tokens=length_pool[_digest(k, "len") % len(length_pool)],
    )


@dataclass
class RandomResult:
    docs: list[SyntheticDocument] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def generate_random(
    n: int,
    length_pool: list[int],
    generators_cfg: dict,
    backend: str = "fake",
    length_ratio_min: float = 0.5,
    length_ratio_max: float = 2.0,
    max_retries: int = 2,
    only_family: str | None = None,
    validation: dict | None = None,
) -> RandomResult:
    from forge.generation.mirror import _PREAMBLE
    from forge.generation.run import GENERATION_BATCH, batch_generate, build_generator, release

    if not length_pool:
        raise ValueError("length_pool is empty; read it from the human corpus")

    # THE TWO ARMS WERE VALIDATED DIFFERENTLY, WHICH IS A DIFFERENCE THE EXPERIMENT
    # CLAIMS NOT TO HAVE.
    #
    # Arm B reads length_ratio_min 0.6 and max 1.6 from its mirror config. Arm A used
    # these signature defaults, 0.5 and 2.0, and the CLI never passed anything else. So
    # the control arm kept documents the mirror arm would have thrown away, and the band
    # is asymmetric on top of that: double the target length is allowed, half is not.
    # Measured on the finished corpus, arm A's documents ran a median of 306 words
    # against the human corpus's 257.
    #
    # The claim this project makes is that the arms differ in MATCHING and in nothing
    # else. Two different length policies is a second difference, pointing the same way
    # as the result. Passing arm B's own validation block in makes there be one policy
    # with one source, and the band is recorded in the stats so the artifact says which
    # one was applied rather than leaving it to be inferred from a default.
    if validation:
        length_ratio_min = validation.get("length_ratio_min", length_ratio_min)
        length_ratio_max = validation.get("length_ratio_max", length_ratio_max)
        max_retries = validation.get("max_retries", max_retries)

    roster = parse_roster(generators_cfg)
    families = held_in_families(roster)
    grid = generators_cfg.get("decoding_grid", {})
    used: set[str] = set()
    stats = ValidationStats()
    now = datetime.now(timezone.utc)

    # PASS 1: prepare every document's spec, family and decoding without a model loaded.
    # Same restructuring as the mirror arm, for the same two reasons: a vllm engine
    # reserves most of the GPU so only one may be alive at a time, and one prompt per
    # generate() call never engages continuous batching. See forge/generation/run.py.
    #
    # Arm A must be fixed identically to Arm B or the comparison is between a working
    # pipeline and a broken one rather than between two data strategies.
    prepared: list[tuple[int, object, object, Decoding]] = []
    for i in range(n):
        spec = spec_for(i, length_pool)
        doc_key = f"rand_{i}"
        fam = assign_family(doc_key, families)
        used.add(fam.family)
        prepared.append((i, spec, fam, assign_decoding(doc_key, grid, spec.target_tokens)))

    by_family: dict[str, list[tuple[int, object, object, Decoding]]] = {}
    for item in prepared:
        by_family.setdefault(item[2].family, []).append(item)

    # ONE FAMILY PER INVOCATION, WITHOUT CHANGING WHICH DOCUMENTS IT GETS.
    #
    # The pod this was written for has 30 GB of disk and no volume. Four generator
    # families at 1.7B to 3.8B is about 23 GB of weights, and torch plus vLLM is another
    # 12 to 15 GB, so the run dies partway through the third model. Splitting it means
    # generating one family, deleting its weights, and starting the next.
    #
    # THE FILTER IS APPLIED HERE AND NOT TO `families` ABOVE, and the difference is the
    # whole point. assign_family hashes each document key against the family LIST, so
    # filtering the roster first would hand every document to the surviving model and
    # produce a different corpus from the unsplit run. Assigning against all four and
    # then running one family's share means four invocations reconstruct exactly what one
    # invocation would have written. There is a test that runs it both ways and compares.
    if only_family is not None:
        known = {f.family for f in families}
        if only_family not in known:
            raise ValueError(
                f"{only_family!r} is not a held-in family in this roster. "
                f"Held-in families are {sorted(known)}."
            )
        by_family = {only_family: by_family.get(only_family, [])}
        used = {only_family} if by_family[only_family] else set()

    # PASS 2: one family at a time, batched, released before the next.
    accepted: dict[int, tuple[str, Decoding]] = {}
    identity: dict[str, tuple[str, str, str]] = {}
    for family in sorted(by_family):
        items = by_family[family]
        fam0 = items[0][2]
        gen = build_generator(fam0, backend)
        identity[family] = (
            getattr(gen, "model_id", fam0.model_id),
            getattr(gen, "revision", fam0.revision),
            fam0.provider,
        )
        print(f"[random] family={family} documents={len(items)} backend={backend}", flush=True)
        try:
            pending = items
            for attempt in range(max_retries + 1):
                if not pending:
                    break
                decs = [
                    Decoding(d.temperature, d.top_p, d.max_new_tokens, (d.seed or 0) + attempt)
                    for (_, _, _, d) in pending
                ]
                prompts = [spec.prompt() + (" " * attempt) for (_, spec, _, _) in pending]
                texts: list[str] = []
                for j in range(0, len(prompts), GENERATION_BATCH):
                    texts.extend(
                        batch_generate(
                            gen, prompts[j : j + GENERATION_BATCH], decs[j : j + GENERATION_BATCH]
                        )
                    )
                    done = min(j + GENERATION_BATCH, len(prompts))
                    print(
                        f"[random] family={family} attempt={attempt} {done}/{len(prompts)}",
                        flush=True,
                    )
                still = []
                for item, d, raw in zip(pending, decs, texts, strict=True):
                    i, spec = item[0], item[1]
                    text = strip_wrapper(raw)
                    reason = None
                    if not text:
                        reason = "empty"
                    elif _PREAMBLE.match(text):
                        reason = "assistant_preamble"
                    else:
                        ratio = len(text.split()) / max(spec.target_tokens, 1)
                        if ratio < length_ratio_min:
                            reason = "too_short"
                        elif ratio > length_ratio_max:
                            reason = "too_long"
                    if reason is None:
                        accepted[i] = (text, d)
                        stats.accepted += 1
                    else:
                        stats.reject(reason)
                        still.append(item)
                pending = still
        finally:
            release(gen)

    # PASS 3: assemble in index order, so output does not depend on family scheduling.
    out: list[SyntheticDocument] = []
    for i, spec, fam, _ in prepared:
        got = accepted.get(i)
        if got is None:
            continue
        text, d = got
        model_id, revision, provider = identity[fam.family]
        # No human source, so this document gets its own group and its own split.
        group = f"grp_rand_{i}"
        out.append(SyntheticDocument(
            sample_id=f"rand_{i}", source_human_id=f"none_rand_{i}", source_group_id=group,
            label=Label.AI, text=text, split=assign_split(group),
            generator=GeneratorSpec(
                provider="api" if provider == "api" else "open_source",
                family=fam.family, model_id=model_id, revision=revision,
                temperature=d.temperature, top_p=d.top_p,
                max_new_tokens=d.max_new_tokens, seed=d.seed,
            ),
            mirror=MirrorSpec(
                prompt_version="random_v1", target_tokens=spec.target_tokens,
                # Explicitly false. This is the control: nothing is matched.
                topic_match=False, length_match=False, style_match=False,
                attributes={"topic": spec.topic, "genre": spec.genre,
                            "register": spec.register, "arm": "random", "backend": backend},
            ),
            domain="web", generated_at=now,
        ))

    assert_no_held_out(roster, used)
    s = stats.as_dict()
    s.update(
        families_used=sorted(used), backend=backend, arm="random",
        validation={"length_ratio_min": length_ratio_min,
                    "length_ratio_max": length_ratio_max,
                    "max_retries": max_retries},
    )
    return RandomResult(out, s)


def length_pool_from_corpus(human_root: str | Path) -> list[int]:
    """Token counts of the human corpus, so Arm A matches its length DISTRIBUTION."""
    import glob

    import pyarrow.parquet as pq

    from forge.ingestion.writer import PARTITION_GLOB

    pool = [
        r["token_count"]
        for f in sorted(glob.glob(str(Path(human_root) / PARTITION_GLOB)))
        for r in pq.read_table(f, columns=["token_count"]).to_pylist()
    ]
    if not pool:
        raise FileNotFoundError(f"no human corpus under {human_root}; run `forge ingest` first")
    return pool
