---
name: brain-workspace
description: Maintain a brain-like Codex workspace for AI-system design, multi-agent orchestration, shared memory experiments, and project learning. Use when the user asks for Brain Workspace, a Codex mod/plugin, shared AI memory, associative/biological memory workflows, hypothesis tracking, idea capture, or automatic updates to `memory/ideas.md` and `memory/hypothesis-graveyard.md`.
---

# Brain Workspace

## Overview

Use this skill to keep a project-level cognitive workspace: ideas stay active until tested, tested hypotheses move into the graveyard with evidence, and each substantive task leaves the memory docs more useful than it found them.

## Operating Loop

1. Find the project root. Prefer the git root; if none exists, use the current working directory.
2. Ensure the memory docs exist by running `scripts/brain_memory.py init --root <project-root>`.
3. Read `memory/ideas.md` and `memory/hypothesis-graveyard.md` before substantive work. Pull in only entries relevant to the current task.
4. Work normally, using parallel tool calls when independent reads, searches, checks, or tests can run at the same time.
5. Before the final response, update the memory docs when the task generated an idea, testable claim, test result, contradiction, or reusable lesson.

Do not add noise. Skip doc edits when the task produces no meaningful project memory.

## Memory Model

- `memory/ideas.md` is working memory for untested ideas, candidate hypotheses, open questions, and promising architecture directions.
- `memory/hypothesis-graveyard.md` is long-term evidence memory for tested hypotheses. "Graveyard" does not mean failed; it means the claim is no longer floating around untested.
- Treat a hypothesis as a testable claim with an observable result. If it has not been tested, keep it in ideas. If it has been tested, record it in the graveyard as `Confirmed`, `Refuted`, `Inconclusive`, or `Superseded`.
- Keep entries concise and provenance-rich: include source, date, evidence, decision, and next test when useful.

## Update Rules

Use the helper script for append-only updates:

```bash
python <skill-dir>/scripts/brain_memory.py init --root <project-root>
python <skill-dir>/scripts/brain_memory.py add-idea --root <project-root> --title "..." --summary "..." --next-test "..."
python <skill-dir>/scripts/brain_memory.py bury-hypothesis --root <project-root> --claim "..." --status Confirmed --test "..." --evidence "..." --decision "..."
```

When an existing entry needs to be moved, corrected, or linked, edit the Markdown directly rather than appending a duplicate.

## Brain-Inspired Heuristics

- Prefer activation over exhaustive recall: retrieve what is relevant to the current goal, recently useful, repeatedly useful, or strongly connected to the task.
- Strengthen associations when two ideas are useful together by cross-linking their IDs.
- Let stale or low-utility ideas decay by moving them lower, marking them parked, or converting them into tested hypotheses.
- When stuck, raise novelty: search adjacent ideas, inspect contradicted hypotheses, or propose a small test.
- Keep shared memory governed: do not store secrets, credentials, private user data, or unverified claims as durable facts.

## Expected Entry Shapes

Idea entries should use this shape:

```markdown
### IDEA-YYYYMMDD-NN: Short title
- Status: Active | Parked | Testing
- Source: Where this came from
- Summary: One to three sentences
- Next test: Smallest useful validation step
- Links: Related IDs or files
- Last touched: YYYY-MM-DD
```

Graveyard entries should use this shape:

```markdown
### HYP-YYYYMMDD-NN: Short claim
- Status: Confirmed | Refuted | Inconclusive | Superseded
- Tested: YYYY-MM-DD
- Test: What was tried
- Evidence: What happened
- Decision: What to believe or do next
- Linked ideas: Related IDEA IDs
```
