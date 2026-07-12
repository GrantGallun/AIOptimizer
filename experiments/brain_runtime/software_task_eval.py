#!/usr/bin/env python3
"""Isolated executable software tasks for context-compiler evaluation."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments.brain_runtime.context_organization_eval import (
    V2_ARMS,
    local_rewrite_fn,
    render_five_arms,
)
from experiments.local_worker.ollama_client import DEFAULT_MODEL, OllamaClient
from gateway.context_compiler import ConversationCompiler

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    "required": ["path", "content"],
    "additionalProperties": False,
}
FORBIDDEN_CALLS = {"eval", "exec", "open", "compile", "__import__"}
SYSTEM = (
    "Implement the requested Python file using only the supplied project context. Return the "
    "declared JSON object. Do not modify tests, invent requirements, or access external resources."
)


def _conversation(stub: str, requirement: str, query: str, *, position: int) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": "Only target.py may be changed."}]
    for index in range(18):
        if index == position:
            content = f"Authoritative implementation requirement:\n{requirement}\n\nCurrent target.py:\n{stub}"
        else:
            content = f"Unrelated project note {index}: documentation cleanup remains pending."
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": content})
    messages.append({"role": "user", "content": query})
    return messages


def make_tasks() -> list[dict[str, Any]]:
    specs = [
        (
            "relevance-router", 10,
            "def choose_mode(scores, threshold=0.5):\n    raise NotImplementedError\n",
            "Implement choose_mode: return 'raw' for an empty sequence; otherwise return "
            "'attention' exactly when max(scores) >= threshold, else 'raw'. Do not mutate scores.",
            "Implement choose_mode in target.py according to the recorded relevance-routing contract.",
            """import unittest\nfrom target import choose_mode\nclass T(unittest.TestCase):\n def test_modes(self):\n  self.assertEqual(choose_mode([], .5), 'raw')\n  self.assertEqual(choose_mode([.1,.49], .5), 'raw')\n  self.assertEqual(choose_mode([.1,.5], .5), 'attention')\n def test_no_mutation(self):\n  x=[.7,.1]; choose_mode(x); self.assertEqual(x,[.7,.1])\n""",
        ),
        (
            "stable-ranking", 12,
            "def rank_records(records):\n    raise NotImplementedError\n",
            "Implement rank_records(records), where each record has id and score. Return a new list "
            "of IDs sorted by descending numeric score, then ascending string ID for exact ties. "
            "Do not mutate records.",
            "Implement deterministic rank_records in target.py using the project's stable tie rule.",
            """import unittest\nfrom target import rank_records\nclass T(unittest.TestCase):\n def test_order(self):\n  x=[{'id':'b','score':1},{'id':'a','score':1},{'id':'c','score':2}]\n  self.assertEqual(rank_records(x),['c','a','b']); self.assertEqual(x[0]['id'],'b')\n""",
        ),
        (
            "authority-guard", 8,
            "def may_bind(authority, requested_binding):\n    raise NotImplementedError\n",
            "Implement may_bind(authority, requested_binding). Return False whenever authority is "
            "'inferred'; otherwise return bool(requested_binding). Unknown authorities raise ValueError.",
            "Implement may_bind in target.py from the typed context authority contract.",
            """import unittest\nfrom target import may_bind\nclass T(unittest.TestCase):\n def test_contract(self):\n  self.assertFalse(may_bind('inferred',True)); self.assertFalse(may_bind('source',False)); self.assertTrue(may_bind('source',True)); self.assertTrue(may_bind('derived',1))\n  with self.assertRaises(ValueError): may_bind('other',True)\n""",
        ),
        (
            "budget-fit", 14,
            "def fit_items(items, budget):\n    raise NotImplementedError\n",
            "Implement fit_items(items,budget). Each item is (id,size). Iterate in input order and "
            "return IDs whose non-negative integer sizes fit cumulatively; skip an item that does not "
            "fit and continue. Negative budget or size raises ValueError. Do not mutate items.",
            "Implement fit_items in target.py according to the deterministic context-budget policy.",
            """import unittest\nfrom target import fit_items\nclass T(unittest.TestCase):\n def test_fit(self): self.assertEqual(fit_items([('a',4),('b',8),('c',3)],7),['a','c'])\n def test_errors(self):\n  with self.assertRaises(ValueError): fit_items([], -1)\n  with self.assertRaises(ValueError): fit_items([('a',-1)], 2)\n""",
        ),
    ]
    tasks = []
    for task_id, position, stub, requirement, query, tests in specs:
        messages = _conversation(stub, requirement, query, position=position)
        tasks.append({
            "id": task_id, "messages": messages, "query": query,
            "expected": "target.py", "expected_source": f"T{position + 2:04d}",
            "forbidden": [], "budget_chars": int(len(json.dumps(messages)) * .55),
            "stub": stub, "tests": tests, "allowed_path": "target.py",
        })
    return tasks


def validate_source(source: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return False, f"syntax: {error.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            return False, f"forbidden AST node: {type(node).__name__}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return False, f"forbidden call: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "spawn", "unlink"}:
            return False, f"forbidden attribute: {node.attr}"
    return True, "ok"


def evaluate_artifact(text: str, task: Mapping[str, Any]) -> dict[str, Any]:
    try:
        artifact = json.loads(text)
    except json.JSONDecodeError as error:
        return {"valid_output": False, "allowed_path": False, "safe_ast": False,
                "tests_passed": False, "detail": f"json: {error.msg}"}
    if not isinstance(artifact, dict) or set(artifact) != {"path", "content"}:
        return {"valid_output": False, "allowed_path": False, "safe_ast": False,
                "tests_passed": False, "detail": "wrong output keys"}
    path_ok = artifact["path"] == task["allowed_path"]
    if not path_ok or not isinstance(artifact["content"], str):
        return {"valid_output": True, "allowed_path": path_ok, "safe_ast": False,
                "tests_passed": False, "detail": "path or content rejected"}
    safe, detail = validate_source(artifact["content"])
    if not safe:
        return {"valid_output": True, "allowed_path": True, "safe_ast": False,
                "tests_passed": False, "detail": detail}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "target.py").write_text(artifact["content"], encoding="utf-8")
        tests = root / "tests"; tests.mkdir()
        (tests / "test_target.py").write_text(str(task["tests"]), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            cwd=root, capture_output=True, text=True, timeout=10,
        )
    return {"valid_output": True, "allowed_path": True, "safe_ast": True,
            "tests_passed": result.returncode == 0,
            "detail": (result.stdout + result.stderr)[-500:]}


def run(
    tasks: Sequence[Mapping[str, Any]], client: OllamaClient, *, model: str = DEFAULT_MODEL,
    compiler: ConversationCompiler | None = None, include_llm_rewrite: bool = True,
) -> dict[str, Any]:
    compiler = compiler or ConversationCompiler()
    rewrite = local_rewrite_fn(client, model) if include_llm_rewrite else None
    rows = []; started = time.perf_counter()
    for task in tasks:
        contexts, diagnostics = render_five_arms(task, compiler, rewrite_fn=rewrite)
        for arm in V2_ARMS:
            generation = client.generate_with_metrics(
                f"Project context:\n{contexts[arm]}\n\nTask (verbatim):\n{task['query']}",
                model=model, system=SYSTEM, temperature=0.0, max_tokens=384,
                format=OUTPUT_SCHEMA,
            )
            verdict = evaluate_artifact(generation.text, task)
            rewrite_metrics = diagnostics[arm].get("rewrite_metrics", {})
            rows.append({
                "task": task["id"], "arm": arm, "response": generation.text,
                "active_request_verbatim": task["query"] in contexts[arm],
                "requirement_present": str(task["expected_source"]) in contexts[arm],
                "record_ids": diagnostics[arm]["record_ids"],
                "context_chars": diagnostics[arm]["context_chars"],
                "preprocess_seconds": diagnostics[arm]["preprocess_seconds"],
                "rewrite_prompt_tokens": int(rewrite_metrics.get("prompt_tokens", 0)),
                "rewrite_completion_tokens": int(rewrite_metrics.get("completion_tokens", 0)),
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_duration_ns": generation.total_duration_ns,
                **verdict,
            })
    summaries = []
    for arm in V2_ARMS:
        selected = [row for row in rows if row["arm"] == arm]; n = len(selected)
        summaries.append({
            "arm": arm, "tasks": n,
            "success_rate": sum(row["tests_passed"] for row in selected) / n,
            "requirement_retention": sum(row["requirement_present"] for row in selected) / n,
            "valid_output_rate": sum(row["valid_output"] for row in selected) / n,
            "safe_ast_rate": sum(row["safe_ast"] for row in selected) / n,
            "prompt_tokens": sum(row["prompt_tokens"] for row in selected),
            "completion_tokens": sum(row["completion_tokens"] for row in selected),
            "rewrite_prompt_tokens": sum(row["rewrite_prompt_tokens"] for row in selected),
            "rewrite_completion_tokens": sum(row["rewrite_completion_tokens"] for row in selected),
            "model_duration_seconds": round(sum(row["total_duration_ns"] for row in selected)/1e9,3),
            "preprocess_seconds": round(sum(row["preprocess_seconds"] for row in selected),3),
        })
    return {"benchmark":"software-context-v1","model":model,"arms":list(V2_ARMS),
            "rows":rows,"summaries":summaries,"elapsed_seconds":round(time.perf_counter()-started,3)}
