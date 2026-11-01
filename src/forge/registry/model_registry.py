"""Model registry and promotion.

A model version is not weights. It is weights plus config plus dataset_version plus code
commit plus metrics plus environment. Anything less cannot be reproduced or audited, and a
detector whose output affects people needs to be auditable.

Promotion is gated, in this order, and every step can refuse:

    registered -> gate passed -> canary -> production

The gate is `forge.evaluation.release_gate.evaluate`. Promotion cannot skip it, and
`promote` refuses a candidate whose gate result is absent rather than treating "not
evaluated" as "no failures found". Those are opposite things and conflating them is how a
model that was never tested reaches production.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from forge.evaluation.release_gate import GateResult

REQUIRED_FIELDS = (
    "version", "weights_uri", "model_config", "dataset_version",
    "code_commit", "metrics", "environment",
)


class Stage(str, Enum):
    REGISTERED = "registered"
    CANARY = "canary"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class PromotionRefused(RuntimeError):
    pass


def validate_entry(entry: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if f not in entry]


@dataclass
class ModelEntry:
    version: str
    weights_uri: str
    model_config: dict
    dataset_version: str
    code_commit: str
    metrics: dict
    environment: dict
    stage: Stage = Stage.REGISTERED
    gate: dict | None = None
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d


class Registry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._entries: dict[str, ModelEntry] = {}

    def register(self, entry: dict) -> ModelEntry:
        missing = validate_entry(entry)
        if missing:
            raise ValueError(
                f"cannot register a model missing {missing}. Weights alone are not a "
                "model version: without dataset_version and code_commit the run cannot "
                "be reproduced or audited."
            )
        e = ModelEntry(**{k: entry[k] for k in REQUIRED_FIELDS})
        if not e.version or e.version in self._entries:
            raise ValueError(f"model version must be non-empty and unique: {e.version!r}")
        self._entries[e.version] = e
        return e

    def get(self, version: str) -> ModelEntry:
        return self._entries[version]

    def record_gate(self, version: str, result: GateResult) -> None:
        e = self._entries[version]
        e.gate = {"passed": result.passed, "failures": list(result.failures),
                  "evaluated_at": datetime.now(timezone.utc).isoformat()}

    def production(self) -> ModelEntry | None:
        for e in self._entries.values():
            if e.stage is Stage.PRODUCTION:
                return e
        return None

    def promote(self, version: str, to: Stage) -> ModelEntry:
        e = self._entries[version]
        if to is Stage.CANARY and e.stage is not Stage.REGISTERED:
            raise PromotionRefused(
                f"{version} is {e.stage.value}; only a registered candidate can enter canary"
            )
        if to is Stage.PRODUCTION and e.stage is not Stage.CANARY:
            raise PromotionRefused(
                f"{version} is {e.stage.value}; production promotion goes through canary"
            )
        if to not in (Stage.CANARY, Stage.PRODUCTION):
            raise PromotionRefused(f"{to.value} is not a promotable target stage")
        if to in (Stage.CANARY, Stage.PRODUCTION):
            if e.gate is None:
                raise PromotionRefused(
                    f"{version} has no gate result. 'Not evaluated' is not 'no failures "
                    "found'; run the release gate before promoting."
                )
            if not e.gate["passed"]:
                raise PromotionRefused(
                    f"{version} failed the release gate: {e.gate['failures']}"
                )
        if to is Stage.PRODUCTION:
            current = self.production()
            if current is not None and current.version != version:
                current.stage = Stage.ARCHIVED
                current.history.append({"event": "archived", "replaced_by": version,
                                        "at": datetime.now(timezone.utc).isoformat()})
        e.history.append({"event": "promote", "from": e.stage.value, "to": to.value,
                          "at": datetime.now(timezone.utc).isoformat()})
        e.stage = to
        return e

    def rollback(self, to_version: str) -> ModelEntry:
        """Promote a previously archived model straight back, skipping canary.

        Deliberately allowed to skip: the model being rolled back to already served
        production traffic, so the canary stage has nothing left to discover, and the
        whole value of a rollback is that it is fast.
        """
        e = self._entries[to_version]
        if e.stage is not Stage.ARCHIVED or not any(
            event.get("event") == "promote" and event.get("to") == Stage.PRODUCTION.value
            for event in e.history
        ):
            raise PromotionRefused(
                f"{to_version} was not a previously served, archived production version"
            )
        current = self.production()
        if current is not None and current.version != to_version:
            current.stage = Stage.ARCHIVED
            current.history.append({"event": "rolled_back_from",
                                    "at": datetime.now(timezone.utc).isoformat()})
        e.history.append({"event": "rollback", "at": datetime.now(timezone.utc).isoformat()})
        e.stage = Stage.PRODUCTION
        return e

    def save(self) -> Path | None:
        if not self.path:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({v: e.as_dict() for v, e in self._entries.items()}, indent=2) + "\n"
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload)
        temporary.replace(self.path)
        return self.path
