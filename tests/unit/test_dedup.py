"""Dedup tests.

Note on the fixtures: BASE is deliberately varied prose rather than one sentence
repeated. Shingles are stored as a set, so a document built by repeating one sentence
has a tiny distinct-shingle set (39 for a 4x repeat), and appending a single sentence
adds 11 new shingles, dropping true Jaccard to 0.78. That is genuinely below the 0.8
threshold, so a repetitive fixture makes this test assert something false about
near-duplicate detection rather than testing it.
"""

from forge.dedup.exact import ExactDeduper
from forge.dedup.minhash import LshParams, MinHash, MinHashLSH, estimated_jaccard, shingles

BASE = (
    "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the "
    "distant hills. The river runs quietly through the wide green valley, carrying "
    "leaves and small branches toward the old stone bridge downstream. Farmers in the "
    "lower fields had finished the harvest weeks earlier, and the barns along the road "
    "stood full and shuttered against the coming cold. A narrow track climbed from the "
    "village toward the ridge, where a line of oaks marked the boundary of the common "
    "land. In the evenings the wind turned and brought the smell of woodsmoke down into "
    "the houses. Children walked the lane to the schoolhouse each morning, past the mill "
    "pond and the disused forge, and returned in the failing light of the afternoon. "
    "The postmistress kept the only telephone for several miles and took messages for "
    "anyone who asked. Nothing much changed from one season to the next, which suited "
    "most of the people who lived there, and unsettled the few who did not."
)


def test_exact_dedup_catches_a_repeat():
    d = ExactDeduper()
    assert d.is_duplicate("a", "hello world") is None
    assert d.is_duplicate("b", "hello world") == "a"
    assert len(d) == 1


def test_signature_is_deterministic():
    m = MinHash()
    assert m.signature(BASE) == m.signature(BASE)


def test_identical_text_has_jaccard_one():
    m = MinHash()
    assert estimated_jaccard(m.signature(BASE), m.signature(BASE)) == 1.0


def test_minhash_estimate_tracks_true_jaccard():
    """The estimate is the whole point, so check it against the exact value."""
    dup = BASE + " A brief closing remark was added at the very end of this document."
    a, b = shingles(BASE), shingles(dup)
    true = len(a & b) / len(a | b)
    m = MinHash()
    est = estimated_jaccard(m.signature(BASE), m.signature(dup))
    assert abs(est - true) < 0.10, f"estimate {est} strayed from true {true}"


def test_near_duplicate_is_caught():
    idx = MinHashLSH()
    idx.add("orig", BASE)
    dup = BASE + " A brief closing remark was added at the very end of this document."
    assert idx.add_if_new("dup", dup) == "orig"


def test_unrelated_text_is_not_merged():
    idx = MinHashLSH()
    idx.add("orig", BASE)
    other = (
        "Financial regulators published a revised framework for capital adequacy "
        "requirements affecting mid-sized institutions across several jurisdictions. "
        "The directive takes effect at the start of the next reporting period and "
        "introduces a countercyclical buffer calibrated to national credit growth. "
        "Supervisors will assess compliance through the existing pillar two process, "
        "and firms falling below the combined buffer requirement face automatic "
        "restrictions on distributions. Industry groups argued during consultation "
        "that the calibration overstates risk in secured lending portfolios."
    )
    assert idx.add_if_new("other", other) is None
    assert len(idx) == 2


def test_word_shingles_not_character_shingles():
    a = shingles("alpha beta gamma delta epsilon zeta")
    b = shingles("one two three four five six")
    assert not (a & b)


def test_band_parameters_are_consistent():
    p = LshParams(num_perm=128, bands=16)
    assert p.rows == 8
    assert p.rows * p.bands == p.num_perm


def test_bands_must_divide_permutations():
    try:
        _ = LshParams(num_perm=128, bands=7).rows
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-dividing band count")
