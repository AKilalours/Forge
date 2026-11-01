"""User feedback intake.

Raw thumbs-down NEVER becomes a training label. Users disagree with correct predictions
constantly, and a document's author is exactly the person most motivated to dispute a
correct "AI" verdict. A detector that learns from unverified disagreement learns to agree
with whoever complains loudest, and its false-negative rate climbs where it matters most.

So feedback is a state machine with a human in it:

    submitted -> under_review -> verified   -> Failure Atlas
                              -> rejected   -> discarded, counted
                              -> unverifiable -> discarded, counted

Only `verified` items reach the atlas, and only with a label a verifier supplied. The
counts of rejected and unverifiable items are kept, because a rising rejection rate is
itself a signal that something upstream is wrong.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


class FeedbackState(str, Enum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNVERIFIABLE = "unverifiable"


TERMINAL = {FeedbackState.VERIFIED, FeedbackState.REJECTED, FeedbackState.UNVERIFIABLE}

ALLOWED: dict[FeedbackState, set[FeedbackState]] = {
    FeedbackState.SUBMITTED: {FeedbackState.UNDER_REVIEW, FeedbackState.UNVERIFIABLE},
    FeedbackState.UNDER_REVIEW: {FeedbackState.VERIFIED, FeedbackState.REJECTED,
                                 FeedbackState.UNVERIFIABLE},
    FeedbackState.VERIFIED: set(),
    FeedbackState.REJECTED: set(),
    FeedbackState.UNVERIFIABLE: set(),
}


class IllegalTransition(RuntimeError):
    pass


class UnverifiedLabel(RuntimeError):
    pass


@dataclass
class FeedbackItem:
    item_id: str
    text_hash: str
    model_version: str
    predicted: str
    user_claim: str                 # what the user says it was
    submitted_at: datetime
    state: FeedbackState = FeedbackState.SUBMITTED
    verified_label: str | None = None
    verifier: str | None = None
    history: list[tuple[str, str]] = field(default_factory=list)

    def transition(self, to: FeedbackState, verifier: str | None = None,
                   verified_label: str | None = None) -> None:
        if to not in ALLOWED[self.state]:
            raise IllegalTransition(f"cannot move from {self.state.value} to {to.value}")
        if to is FeedbackState.VERIFIED:
            if not verifier or not verified_label:
                raise UnverifiedLabel(
                    "verifying requires both a verifier and a label THEY supplied. The "
                    "user's claim is not a label; it is the thing being checked."
                )
            if verified_label not in {"human", "ai"}:
                raise UnverifiedLabel("verified_label must be one of: human, ai")
            self.verified_label = verified_label
            self.verifier = verifier
        self.history.append((self.state.value, to.value))
        self.state = to


class FeedbackQueue:
    def __init__(self) -> None:
        self._items: dict[str, FeedbackItem] = {}

    def submit(self, item: FeedbackItem) -> None:
        if not item.item_id:
            raise ValueError("feedback item_id must not be empty")
        if item.item_id in self._items:
            raise ValueError(f"feedback item already exists: {item.item_id}")
        self._items[item.item_id] = item

    def get(self, item_id: str) -> FeedbackItem:
        return self._items[item_id]

    def pending(self) -> list[FeedbackItem]:
        return [i for i in self._items.values() if i.state not in TERMINAL]

    def to_atlas(self) -> list[FeedbackItem]:
        """Only verified items, and only ever verified items."""
        return [i for i in self._items.values() if i.state is FeedbackState.VERIFIED]

    def stats(self) -> dict:
        c = Counter(i.state.value for i in self._items.values())
        reviewed = sum(c[s.value] for s in TERMINAL)
        return {
            "total": len(self._items),
            "by_state": dict(c),
            "pending": len(self.pending()),
            "rejection_rate": round(c[FeedbackState.REJECTED.value] / reviewed, 4) if reviewed else 0.0,
        }


def now() -> datetime:
    return datetime.now(timezone.utc)
