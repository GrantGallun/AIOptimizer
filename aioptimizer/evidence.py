"""Machine-readable evidence limitations for experiment result artifacts.

This module does not produce research verdicts. It checks whether an evidence
card records the controls needed to support internal- or external-validity
claims, and reports every missing condition instead of silently upgrading a
benchmark result into a general conclusion.

    python -m aioptimizer.evidence result_evidence.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DIRECT_JUDGES = frozenset({"deterministic", "human"})
PROXY_JUDGES = frozenset({"encoder", "heuristic", "model", "unknown"})


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{field} must be a list of non-empty strings")
    return tuple(value)


@dataclass(frozen=True)
class EvidenceCard:
    experiment_id: str
    sample_size: int
    task_families: tuple[str, ...]
    models: tuple[str, ...]
    seeds: tuple[str, ...]
    preregistered: bool
    hidden_split: bool
    negative_controls: bool
    independent_review: bool
    external_data: bool
    confidence_interval: bool
    versioned_artifact: bool
    judge: str

    def __post_init__(self) -> None:
        """Enforce the schema even when callers bypass ``from_mapping``."""
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("experiment_id must be a non-empty string")
        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
            or self.sample_size < 0
        ):
            raise ValueError("sample_size must be a non-negative integer")
        for field in ("task_families", "models", "seeds"):
            value = getattr(self, field)
            if not isinstance(value, tuple) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ValueError(f"{field} must be a tuple of non-empty strings")
        for field in (
            "preregistered",
            "hidden_split",
            "negative_controls",
            "independent_review",
            "external_data",
            "confidence_interval",
            "versioned_artifact",
        ):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be boolean")
        if self.judge not in DIRECT_JUDGES | PROXY_JUDGES:
            raise ValueError(
                "judge must be deterministic, human, encoder, heuristic, model, or unknown"
            )

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EvidenceCard":
        if not isinstance(value, dict):
            raise ValueError("evidence card must be an object")
        experiment_id = value.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("experiment_id must be a non-empty string")
        sample_size = value.get("sample_size")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 0:
            raise ValueError("sample_size must be a non-negative integer")
        judge = value.get("judge", "unknown")
        if judge not in DIRECT_JUDGES | PROXY_JUDGES:
            raise ValueError(
                "judge must be deterministic, human, encoder, heuristic, model, or unknown"
            )
        flags = {}
        for field in (
            "preregistered",
            "hidden_split",
            "negative_controls",
            "independent_review",
            "external_data",
            "confidence_interval",
            "versioned_artifact",
        ):
            flag = value.get(field, False)
            if not isinstance(flag, bool):
                raise ValueError(f"{field} must be boolean")
            flags[field] = flag
        return cls(
            experiment_id=experiment_id,
            sample_size=sample_size,
            task_families=_string_tuple(value.get("task_families", []), "task_families"),
            models=_string_tuple(value.get("models", []), "models"),
            seeds=_string_tuple(value.get("seeds", []), "seeds"),
            judge=judge,
            **flags,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "sample_size": self.sample_size,
            "task_families": list(self.task_families),
            "models": list(self.models),
            "seeds": list(self.seeds),
            "preregistered": self.preregistered,
            "hidden_split": self.hidden_split,
            "negative_controls": self.negative_controls,
            "independent_review": self.independent_review,
            "external_data": self.external_data,
            "confidence_interval": self.confidence_interval,
            "versioned_artifact": self.versioned_artifact,
            "judge": self.judge,
        }


def audit_evidence(card: EvidenceCard, *, minimum_sample_size: int = 30) -> dict[str, Any]:
    """Return explicit limitations and non-verdict validity check results."""
    if minimum_sample_size < 1:
        raise ValueError("minimum_sample_size must be positive")
    limitations: list[dict[str, str]] = []

    def flag(condition: bool, code: str, message: str, severity: str = "warning") -> None:
        if condition:
            limitations.append({"code": code, "severity": severity, "message": message})

    flag(card.sample_size < minimum_sample_size, "small_sample", f"sample_size is below {minimum_sample_size}")
    flag(len(card.seeds) < 3, "few_seeds", "fewer than three independent seeds are recorded")
    flag(not card.preregistered, "not_preregistered", "arms, metrics, and gate were not frozen in advance")
    flag(not card.hidden_split, "no_hidden_split", "no untouched hidden split is recorded")
    flag(not card.negative_controls, "no_negative_controls", "no negative control is recorded")
    flag(not card.confidence_interval, "no_uncertainty", "no confidence interval or uncertainty estimate is recorded")
    flag(not card.versioned_artifact, "unversioned_artifact", "result artifact is not declared immutable/versioned")
    flag(len(card.task_families) < 2, "single_task_family", "fewer than two task families are recorded")
    flag(len(card.models) < 2, "single_model", "fewer than two model families are recorded")
    flag(not card.external_data, "author_built_only", "no external or independently sourced data is recorded")
    flag(not card.independent_review, "no_independent_review", "producer-independent review is not recorded")
    flag(card.judge in PROXY_JUDGES, "proxy_judge", f"{card.judge} judge is a proxy rather than direct deterministic/human scoring")

    internal_codes = {
        "small_sample", "few_seeds", "not_preregistered", "no_hidden_split",
        "no_negative_controls", "no_uncertainty", "unversioned_artifact",
    }
    external_codes = internal_codes | {
        "single_task_family", "single_model", "author_built_only",
        "no_independent_review", "proxy_judge",
    }
    present = {item["code"] for item in limitations}
    return {
        "schema": "aioptimizer.evidence-audit.v1",
        "experiment_id": card.experiment_id,
        "minimum_sample_size": minimum_sample_size,
        "meets_internal_validity_checks": not bool(present & internal_codes),
        "meets_external_validity_checks": not bool(present & external_codes),
        "limitations": limitations,
        "limitation_count": len(limitations),
        "note": "Validity checks are metadata completeness checks, not a research verdict.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", help="JSON evidence-card path")
    parser.add_argument("--minimum-sample-size", type=int, default=30)
    args = parser.parse_args()
    payload = json.loads(Path(args.card).read_text(encoding="utf-8"))
    card = EvidenceCard.from_mapping(payload.get("evidence", payload))
    print(json.dumps(audit_evidence(card, minimum_sample_size=args.minimum_sample_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
