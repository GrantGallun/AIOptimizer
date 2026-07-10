#!/usr/bin/env python3
"""Maintain Brain Workspace project memory documents."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path


IDEAS_FILE = "ideas.md"
GRAVEYARD_FILE = "hypothesis-graveyard.md"


def today_iso() -> str:
    return dt.date.today().isoformat()


def today_id() -> str:
    return dt.date.today().strftime("%Y%m%d")


def resolve_root(root: str | None) -> Path:
    return Path(root or ".").resolve()


def paths(root: Path) -> tuple[Path, Path]:
    memory_dir = root / "memory"
    return memory_dir / IDEAS_FILE, memory_dir / GRAVEYARD_FILE


def ensure_docs(root: Path) -> tuple[Path, Path]:
    ideas_path, graveyard_path = paths(root)
    ideas_path.parent.mkdir(parents=True, exist_ok=True)

    if not ideas_path.exists():
        ideas_path.write_text(
            f"""# Ideas

Living queue for Brain Workspace ideas that have not been tested yet.

Update rules:
- Add only actionable or recurring ideas.
- Keep untested hypotheses here until evidence is gathered.
- When an idea is tested, add the result to `hypothesis-graveyard.md` and update or remove the active idea entry.
- Avoid churn; do not log obvious implementation minutiae.

Created: {today_iso()}

## Active Ideas

## Candidate Hypotheses

## Open Questions
""",
            encoding="utf-8",
        )

    if not graveyard_path.exists():
        graveyard_path.write_text(
            f"""# Hypothesis Graveyard

Tested hypotheses and their evidence. "Graveyard" means the claim is no longer floating untested; it may be confirmed, refuted, inconclusive, or superseded.

Update rules:
- Move tested hypotheses here whether they succeed or fail.
- Include the test, evidence, decision, and linked idea when available.
- Prefer superseding or correcting old entries over leaving contradictory claims unresolved.

Created: {today_iso()}

## Tested Hypotheses
""",
            encoding="utf-8",
        )

    return ideas_path, graveyard_path


def next_id(path: Path, prefix: str) -> str:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    date_part = today_id()
    pattern = re.compile(rf"\b{re.escape(prefix)}-{date_part}-(\d+)\b")
    highest = 0
    for match in pattern.finditer(text):
        highest = max(highest, int(match.group(1)))
    return f"{prefix}-{date_part}-{highest + 1:02d}"


def append_to_section(path: Path, heading: str, entry: str) -> None:
    text = path.read_text(encoding="utf-8")
    marker = f"## {heading}"
    if marker not in text:
        text = text.rstrip() + f"\n\n{marker}\n"

    start = text.index(marker)
    next_heading = text.find("\n## ", start + len(marker))
    insert_at = len(text) if next_heading == -1 else next_heading
    before = text[:insert_at].rstrip()
    after = text[insert_at:]
    path.write_text(before + "\n\n" + entry.strip() + "\n" + after, encoding="utf-8")


def command_init(args: argparse.Namespace) -> None:
    root = resolve_root(args.root)
    ideas_path, graveyard_path = ensure_docs(root)
    print(f"ideas: {ideas_path}")
    print(f"graveyard: {graveyard_path}")


def command_add_idea(args: argparse.Namespace) -> None:
    root = resolve_root(args.root)
    ideas_path, _ = ensure_docs(root)
    idea_id = next_id(ideas_path, "IDEA")
    entry = f"""### {idea_id}: {args.title}
- Status: {args.status}
- Source: {args.source}
- Summary: {args.summary}
- Next test: {args.next_test}
- Links: {args.links}
- Last touched: {today_iso()}"""
    append_to_section(ideas_path, "Active Ideas", entry)
    print(idea_id)


def command_bury_hypothesis(args: argparse.Namespace) -> None:
    root = resolve_root(args.root)
    _, graveyard_path = ensure_docs(root)
    hyp_id = next_id(graveyard_path, "HYP")
    entry = f"""### {hyp_id}: {args.claim}
- Status: {args.status}
- Tested: {today_iso()}
- Test: {args.test}
- Evidence: {args.evidence}
- Decision: {args.decision}
- Linked ideas: {args.linked_ideas}"""
    append_to_section(graveyard_path, "Tested Hypotheses", entry)
    print(hyp_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create memory docs if missing.")
    init_parser.add_argument("--root", default=".", help="Project root. Defaults to cwd.")
    init_parser.set_defaults(func=command_init)

    idea_parser = subparsers.add_parser("add-idea", help="Append an untested idea.")
    idea_parser.add_argument("--root", default=".", help="Project root. Defaults to cwd.")
    idea_parser.add_argument("--title", required=True)
    idea_parser.add_argument("--summary", required=True)
    idea_parser.add_argument("--status", default="Active", choices=["Active", "Parked", "Testing"])
    idea_parser.add_argument("--source", default="Current Codex session")
    idea_parser.add_argument("--next-test", default="Define the smallest useful validation step.")
    idea_parser.add_argument("--links", default="None")
    idea_parser.set_defaults(func=command_add_idea)

    hyp_parser = subparsers.add_parser("bury-hypothesis", help="Append a tested hypothesis.")
    hyp_parser.add_argument("--root", default=".", help="Project root. Defaults to cwd.")
    hyp_parser.add_argument("--claim", required=True)
    hyp_parser.add_argument("--status", required=True, choices=["Confirmed", "Refuted", "Inconclusive", "Superseded"])
    hyp_parser.add_argument("--test", required=True)
    hyp_parser.add_argument("--evidence", required=True)
    hyp_parser.add_argument("--decision", required=True)
    hyp_parser.add_argument("--linked-ideas", default="None")
    hyp_parser.set_defaults(func=command_bury_hypothesis)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
