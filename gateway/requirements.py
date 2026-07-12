"""Explicit deterministic output contracts for requirement-retention receipts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

MAX_REQUIREMENTS = 64
MAX_TERMS_PER_REQUIREMENT = 64
MAX_TERM_CHARS = 4096
ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


@dataclass(frozen=True)
class Requirement:
    id: str
    must_include: tuple[str, ...]
    must_exclude: tuple[str, ...]
    case_sensitive: bool = False


def extract_requirements(body: Any) -> tuple[dict[str, Any], tuple[Requirement, ...]]:
    """Validate and remove the private ``aioptimizer`` request envelope."""
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    if "aioptimizer" not in body:
        return body, ()
    envelope = body["aioptimizer"]
    if not isinstance(envelope, dict) or set(envelope) != {"requirements"}:
        raise ValueError("aioptimizer must contain only requirements")
    rows = envelope["requirements"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_REQUIREMENTS:
        raise ValueError(f"aioptimizer requirements must contain 1-{MAX_REQUIREMENTS} items")
    requirements = tuple(_parse_requirement(row, index) for index, row in enumerate(rows))
    ids = [requirement.id for requirement in requirements]
    if len(set(ids)) != len(ids):
        raise ValueError("aioptimizer requirement ids must be unique")
    clean = dict(body)
    del clean["aioptimizer"]
    return clean, requirements


def _parse_requirement(row: Any, index: int) -> Requirement:
    if not isinstance(row, dict):
        raise ValueError(f"requirement {index} must be an object")
    allowed = {"id", "must_include", "must_exclude", "case_sensitive"}
    if set(row) - allowed:
        raise ValueError(f"requirement {index} has unknown keys")
    identifier = row.get("id")
    if not isinstance(identifier, str) or ID_PATTERN.fullmatch(identifier) is None:
        raise ValueError(
            f"requirement {index} id must be 1-128 safe identifier characters"
        )
    include = _terms(row.get("must_include", []), index, "must_include")
    exclude = _terms(row.get("must_exclude", []), index, "must_exclude")
    if not include and not exclude:
        raise ValueError(f"requirement {index} must define an include or exclude check")
    case_sensitive = row.get("case_sensitive", False)
    if not isinstance(case_sensitive, bool):
        raise ValueError(f"requirement {index} case_sensitive must be boolean")
    return Requirement(identifier, include, exclude, case_sensitive)


def _terms(value: Any, index: int, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > MAX_TERMS_PER_REQUIREMENT:
        raise ValueError(
            f"requirement {index} {field} must be a list of at most "
            f"{MAX_TERMS_PER_REQUIREMENT} strings"
        )
    if any(
        not isinstance(term, str) or not term or len(term) > MAX_TERM_CHARS
        for term in value
    ):
        raise ValueError(
            f"requirement {index} {field} values must be 1-{MAX_TERM_CHARS} characters"
        )
    return tuple(value)


def evaluate_requirements(
    text: str, requirements: tuple[Requirement, ...]
) -> dict[str, Any] | None:
    """Evaluate declared exact checks without retaining their private text."""
    if not requirements:
        return None
    results = []
    for requirement in requirements:
        haystack = text if requirement.case_sensitive else text.casefold()
        includes = [
            term if requirement.case_sensitive else term.casefold()
            for term in requirement.must_include
        ]
        excludes = [
            term if requirement.case_sensitive else term.casefold()
            for term in requirement.must_exclude
        ]
        include_passes = sum(term in haystack for term in includes)
        exclude_passes = sum(term not in haystack for term in excludes)
        passed = include_passes == len(includes) and exclude_passes == len(excludes)
        results.append({
            "id": requirement.id,
            "passed": passed,
            "include_checks": len(includes),
            "include_passes": include_passes,
            "exclude_checks": len(excludes),
            "exclude_passes": exclude_passes,
        })
    passed = sum(result["passed"] for result in results)
    return {
        "requirements": len(results),
        "passed": passed,
        "all_passed": passed == len(results),
        "results": results,
    }
