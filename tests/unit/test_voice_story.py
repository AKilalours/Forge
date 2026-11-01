from __future__ import annotations

import pytest

from forge.experience.voice_story import (
    capsule_from_answers,
    parsed_quote_candidates,
    quote_candidates,
    render_capsule_html,
)


def test_quote_candidates_are_exact_deduplicated_and_bounded():
    source = (
        "The chipped blue cup waited beside the window every Sunday. "
        "The chipped blue cup waited beside the window every Sunday. "
        "This is another sentence that should remain exactly as the writer supplied it."
    )
    quotes = quote_candidates(source)
    assert [quote.text for quote in quotes] == [
        "The chipped blue cup waited beside the window every Sunday.",
        "This is another sentence that should remain exactly as the writer supplied it.",
    ]


def test_redacto_quotes_keep_the_page_reference():
    parsed = {
        "chunks": [{
            "text": "The chipped blue cup waited beside the window every Sunday.",
            "blocks": [{"type": "Text", "page": 4}],
        }]
    }
    candidates = parsed_quote_candidates(parsed)
    assert candidates[0].page == 4


def test_capsule_requires_author_supplied_reflections():
    with pytest.raises(ValueError, match="detail, intention"):
        capsule_from_answers(
            title="A title", author="", excerpt="The exact line.", detail="",
            intention="", keepsake="",
        )


def test_export_is_self_contained_and_escapes_untrusted_text():
    capsule = capsule_from_answers(
        title="<script>alert(1)</script>", author="A & B", excerpt="A chosen line.",
        detail="A detail", intention="An intention", keepsake="A note",
    )
    page = render_capsule_html(capsule)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "speechSynthesis" in page
    assert "fonts.googleapis" not in page
