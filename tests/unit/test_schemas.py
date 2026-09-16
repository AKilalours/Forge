from datetime import datetime, timezone

import pytest

from forge.common.schemas import HumanDocument, Quality, Split, TokenLabel, TokenLabelSpan


def test_token_label_set_is_exactly_three_and_ordered():
    assert [t.value for t in TokenLabel] == ["human", "ai_assisted", "ai_generated"]


def test_human_document_requires_a_real_hash():
    with pytest.raises(ValueError):
        HumanDocument(
            doc_id="x", source_group_id="grp_x", text="t", source="fw", license="ODC-By-1.0",
            domain="web", text_register="informational", language_score=0.9,
            acquired_at=datetime.now(timezone.utc), processing_version="clean_v1",
            content_sha256="tooshort", token_count=10, quality=Quality(), split=Split.TRAIN,
        )


def test_span_must_be_ordered():
    with pytest.raises(ValueError):
        TokenLabelSpan(start_char=10, end_char=5, label=TokenLabel.HUMAN)
