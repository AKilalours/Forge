"""Character spans to per-token labels.

This is the highest-risk pure function in Phase 3. The spec deliberately stores token
supervision as CHARACTER spans, because token indices are tokenizer-dependent and the
tokenizer will change. That decision moves the risk here, into the alignment step.

Three ways this goes wrong quietly:

  Off-by-one at a boundary. A token straddling a human/AI seam gets whichever label the
  comparison happens to favour. With 512-token windows and one seam per document, a
  systematic off-by-one shifts every boundary in the corpus by one token, and boundary
  F1 reports a real-looking number that is wrong by a constant.

  Special tokens. [CLS] and [SEP] have offset (0, 0). Assigning them a real label trains
  the model to predict authorship for tokens that carry no text, and they appear in every
  window, so the effect is large.

  Whitespace-only tokens between spans. They must inherit a label rather than being
  dropped, otherwise the label sequence has holes the loss silently ignores.

The rule used here: a token is labelled by the span covering the MAJORITY of its
characters. Ties go to the earlier span. That is deterministic and symmetric, and it is
stated here so the choice is visible rather than emergent.
"""

from __future__ import annotations

from forge.common.schemas import TokenLabel

IGNORE_INDEX = -100  # torch's cross-entropy default; special tokens get this

LABEL_TO_ID = {label: i for i, label in enumerate(TokenLabel)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def align_spans_to_tokens(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int, TokenLabel]],
    ignore_index: int = IGNORE_INDEX,
) -> list[int]:
    """Return one label id per token.

    offsets: (start_char, end_char) per token, as returned by a fast tokenizer with
             return_offsets_mapping=True. Special tokens have (0, 0).
    spans:   (start_char, end_char, label), tiling the document exactly.
    """
    if not spans:
        raise ValueError("no spans supplied")
    ordered = sorted(spans, key=lambda s: s[0])

    out: list[int] = []
    for start, end in offsets:
        if end <= start:  # special token, or an empty piece
            out.append(ignore_index)
            continue
        best_label, best_overlap = None, 0
        for s_start, s_end, label in ordered:
            if s_start >= end:
                break
            overlap = min(end, s_end) - max(start, s_start)
            if overlap > best_overlap:  # strict >, so ties keep the earlier span
                best_overlap, best_label = overlap, label
        if best_label is None:
            # Token falls outside every span, e.g. trailing whitespace past the last
            # span. Inherit the nearest preceding span rather than leaving a hole.
            best_label = ordered[-1][2] if start >= ordered[-1][1] else ordered[0][2]
        out.append(LABEL_TO_ID[best_label])
    return out


def document_label_from_tokens(label_ids: list[int], ignore_index: int = IGNORE_INDEX) -> float:
    """AI character fraction implied by a token label sequence. Used for consistency
    checks between the token head and the document head."""
    real = [i for i in label_ids if i != ignore_index]
    if not real:
        return 0.0
    human_id = LABEL_TO_ID[TokenLabel.HUMAN]
    return sum(1 for i in real if i != human_id) / len(real)


def spans_from_token_labels(
    offsets: list[tuple[int, int]],
    label_ids: list[int],
    ignore_index: int = IGNORE_INDEX,
) -> list[tuple[int, int, TokenLabel]]:
    """Inverse direction: turn predicted token labels back into character spans for the API."""
    out: list[tuple[int, int, TokenLabel]] = []
    for (start, end), lid in zip(offsets, label_ids, strict=True):
        if lid == ignore_index or end <= start:
            continue
        label = ID_TO_LABEL[lid]
        if out and out[-1][2] is label and out[-1][1] >= start:
            out[-1] = (out[-1][0], max(out[-1][1], end), label)
        else:
            out.append((start, end, label))
    return out
