#!/usr/bin/env python3
"""Prototype behavior-vector steering harness.

The `toy` backend is deterministic and dependency-light. It proves the benchmark
plumbing: compute contrastive vectors, apply them, score outputs, and write a
machine-readable result.

The `hf` backend is optional and requires a locally available causal language
model plus torch/transformers. It does not download models unless
`--allow-download` is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_PAIRS = DATA_DIR / "behavior_pairs.jsonl"
DEFAULT_TASKS = DATA_DIR / "eval_tasks.jsonl"
DEFAULT_TRAIN_PAIRS = DATA_DIR / "steering_pairs_train.jsonl"
DEFAULT_DEV_TASKS = DATA_DIR / "eval_dev.jsonl"
DEFAULT_HIDDEN_TASKS = DATA_DIR / "eval_hidden.jsonl"
DEFAULT_ADVERSARIAL_TASKS = DATA_DIR / "eval_adversarial.jsonl"


TOY_DIMS = {
    "evidence": 0,
    "clarify": 1,
    "concise": 2,
}

TOY_POSITIVE_TERMS = {
    "evidence": {"evidence", "test", "result", "verify", "inspect", "observed", "source", "run"},
    "clarify": {"ask", "question", "clarifying", "uncertainty", "requirement", "missing", "detail"},
    "concise": {"brief", "short", "direct", "necessary", "concrete", "scannable", "result", "done"},
}

TOY_NEGATIVE_TERMS = {
    "evidence": {"guess", "assume", "probably", "skip"},
    "clarify": {"assume", "proceed", "silently", "hide"},
    "concise": {"verbose", "meandering", "unrelated", "background", "repeated"},
}

TOY_GOOD_RESPONSES = {
    "evidence": "Verify the evidence first: inspect the state, run a small test, report the observed result, then decide.",
    "clarify": "Ask one clarifying question, name the uncertainty, and request the missing requirement before acting.",
    "concise": "Done. Give a short, direct, concrete result with only necessary details.",
}

TOY_BAD_RESPONSES = {
    "evidence": "Guess from memory, assume the likely answer, and probably skip the test.",
    "clarify": "Proceed silently, assume the convenient path, and hide the missing detail.",
    "concise": "Write a verbose, meandering update with unrelated background and repeated caveats.",
}


@dataclass
class BehaviorVector:
    behavior: str
    values: list[float]

    @property
    def norm(self) -> float:
        return math.sqrt(sum(value * value for value in self.values))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_noise(token: str, dims: int) -> list[float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    values = []
    for index in range(dims):
        byte = digest[index % len(digest)]
        values.append((byte / 255.0 - 0.5) * 0.05)
    return values


def add_vectors(left: list[float], right: Iterable[float]) -> list[float]:
    return [a + b for a, b in zip(left, right)]


def sub_vectors(left: list[float], right: Iterable[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def scale_vector(values: list[float], scale: float) -> list[float]:
    return [value * scale for value in values]


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class ToyBehaviorModel:
    """A tiny interpretable model with behavior axes.

    This is intentionally not a language model. It gives us a deterministic
    harness where contrastive vectors have an observable causal effect.
    """

    def __init__(self, dims: int = 12) -> None:
        if dims <= max(TOY_DIMS.values()):
            raise ValueError("dims is too small for toy behavior axes")
        self.dims = dims

    def encode(self, text: str) -> list[float]:
        text_lower = text.lower()
        vector = [0.0] * self.dims
        for token in text_lower.replace(".", " ").replace(",", " ").replace(":", " ").split():
            vector = add_vectors(vector, stable_noise(token, self.dims))

        for behavior, dim in TOY_DIMS.items():
            pos_hits = sum(1 for term in TOY_POSITIVE_TERMS[behavior] if term in text_lower)
            neg_hits = sum(1 for term in TOY_NEGATIVE_TERMS[behavior] if term in text_lower)
            vector[dim] += pos_hits * 1.2
            vector[dim] -= neg_hits * 1.2
        return vector

    def direction(self, behavior: str) -> list[float]:
        vector = [0.0] * self.dims
        vector[TOY_DIMS[behavior]] = 1.0
        return vector

    def generate(self, prompt: str, behavior: str, steering: BehaviorVector | None = None, coefficient: float = 1.0) -> str:
        activation = self.encode(prompt)
        if steering is not None:
            activation = add_vectors(activation, scale_vector(steering.values, coefficient))
        score = dot(activation, self.direction(behavior))
        return TOY_GOOD_RESPONSES[behavior] if score > 0.75 else TOY_BAD_RESPONSES[behavior]


def compute_toy_vectors(pairs: list[dict[str, Any]], model: ToyBehaviorModel) -> dict[str, BehaviorVector]:
    grouped: dict[str, list[list[float]]] = {}
    for row in pairs:
        behavior = row["behavior"]
        diff = sub_vectors(model.encode(row["positive"]), model.encode(row["negative"]))
        grouped.setdefault(behavior, []).append(diff)

    vectors: dict[str, BehaviorVector] = {}
    for behavior, diffs in grouped.items():
        average = [sum(diff[dim] for diff in diffs) / len(diffs) for dim in range(model.dims)]
        vectors[behavior] = BehaviorVector(behavior=behavior, values=average)
    return vectors


def lexical_score(output: str, positive_terms: list[str], negative_terms: list[str]) -> dict[str, Any]:
    text = output.lower()
    positive_hits = sorted({term for term in positive_terms if term.lower() in text})
    negative_hits = sorted({term for term in negative_terms if term.lower() in text})
    return {
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "score": len(positive_hits) - len(negative_hits),
        "success": len(positive_hits) >= 2 and not negative_hits,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    baseline_success = sum(1 for row in rows if row["baseline"]["metrics"]["success"])
    steered_success = sum(1 for row in rows if row["steered"]["metrics"]["success"])
    summary = {
        "tasks": total,
        "baseline_success": baseline_success,
        "steered_success": steered_success,
        "baseline_success_rate": baseline_success / total if total else 0.0,
        "steered_success_rate": steered_success / total if total else 0.0,
        "absolute_success_lift": (steered_success - baseline_success) / total if total else 0.0,
    }
    if rows and "baseline_preference" in rows[0]:
        baseline_pref_success = sum(1 for row in rows if row["baseline_preference"]["success"])
        steered_pref_success = sum(1 for row in rows if row["steered_preference"]["success"])
        baseline_avg_margin = sum(row["baseline_preference"]["margin"] for row in rows) / total
        steered_avg_margin = sum(row["steered_preference"]["margin"] for row in rows) / total
        summary.update(
            {
                "baseline_preference_success": baseline_pref_success,
                "steered_preference_success": steered_pref_success,
                "baseline_preference_success_rate": baseline_pref_success / total,
                "steered_preference_success_rate": steered_pref_success / total,
                "preference_success_lift": (steered_pref_success - baseline_pref_success) / total,
                "baseline_average_margin": baseline_avg_margin,
                "steered_average_margin": steered_avg_margin,
                "average_margin_lift": steered_avg_margin - baseline_avg_margin,
            }
        )
    return summary


def run_toy(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    pairs = read_jsonl(Path(args.pairs))
    tasks = read_jsonl(Path(args.tasks))
    model = ToyBehaviorModel(dims=args.dims)
    vectors = compute_toy_vectors(pairs, model)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        behavior = task["behavior"]
        vector = vectors[behavior]
        baseline_output = model.generate(task["prompt"], behavior)
        steered_output = model.generate(task["prompt"], behavior, steering=vector, coefficient=args.coefficient)
        rows.append(
            {
                "id": task["id"],
                "behavior": behavior,
                "prompt": task["prompt"],
                "vector_norm": vector.norm,
                "baseline": {
                    "output": baseline_output,
                    "metrics": lexical_score(baseline_output, task["positive_terms"], task["negative_terms"]),
                },
                "steered": {
                    "output": steered_output,
                    "metrics": lexical_score(steered_output, task["positive_terms"], task["negative_terms"]),
                },
            }
        )

    payload = {
        "backend": "toy",
        "config": {
            "pairs": str(Path(args.pairs)),
            "tasks": str(Path(args.tasks)),
            "dims": args.dims,
            "coefficient": args.coefficient,
        },
        "summary": summarize(rows),
        "rows": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(Path(args.out), payload)
    return payload


def find_transformer_blocks(model: Any) -> list[Any]:
    candidates = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        current = model
        try:
            for part in path:
                current = getattr(current, part)
            if len(current) > 0:
                return list(current)
        except AttributeError:
            continue
    raise RuntimeError("Could not locate transformer blocks for activation hooks.")


def normalize_index(index: int, length: int) -> int:
    return index if index >= 0 else length + index


def parse_number_list(raw: str, caster: Any) -> list[Any]:
    values: list[Any] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(caster(part))
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_string_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def score_preference_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    baseline_pref_success = sum(1 for row in rows if row["baseline_preference"]["success"])
    steered_pref_success = sum(1 for row in rows if row["steered_preference"]["success"])
    baseline_avg_margin = sum(row["baseline_preference"]["margin"] for row in rows) / total if total else 0.0
    steered_avg_margin = sum(row["steered_preference"]["margin"] for row in rows) / total if total else 0.0
    return {
        "tasks": total,
        "baseline_preference_success": baseline_pref_success,
        "steered_preference_success": steered_pref_success,
        "baseline_preference_success_rate": baseline_pref_success / total if total else 0.0,
        "steered_preference_success_rate": steered_pref_success / total if total else 0.0,
        "preference_success_lift": (steered_pref_success - baseline_pref_success) / total if total else 0.0,
        "baseline_average_margin": baseline_avg_margin,
        "steered_average_margin": steered_avg_margin,
        "average_margin_lift": steered_avg_margin - baseline_avg_margin,
    }


def run_hf(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("The hf backend requires torch and transformers.") from exc

    pairs = read_jsonl(Path(args.pairs))
    tasks = read_jsonl(Path(args.tasks))
    local_files_only = not args.allow_download

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=local_files_only,
        torch_dtype="auto",
        device_map=args.device_map,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = find_transformer_blocks(model)
    block_index = normalize_index(args.layer, len(blocks))
    if block_index < 0 or block_index >= len(blocks):
        raise ValueError(f"Layer index {args.layer} resolved to {block_index}, outside 0..{len(blocks)-1}.")

    device = next(model.parameters()).device

    def supports_chat_template() -> bool:
        return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")

    def format_user_prompt(text: str) -> str:
        if args.chat_template == "off" or not supports_chat_template():
            return text
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def format_behavior_example(text: str) -> str:
        if args.chat_template == "off" or not supports_chat_template():
            return text
        messages = [
            {"role": "user", "content": "Describe the response style the assistant should follow."},
            {"role": "assistant", "content": text},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def activation_for(text: str) -> Any:
        inputs = tokenizer(format_behavior_example(text), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_index = block_index + 1
        hidden = outputs.hidden_states[hidden_index][0]
        return hidden.mean(dim=0)

    vectors: dict[str, Any] = {}
    for behavior in sorted({row["behavior"] for row in pairs}):
        diffs = []
        for row in pairs:
            if row["behavior"] != behavior:
                continue
            diffs.append(activation_for(row["positive"]) - activation_for(row["negative"]))
        vectors[behavior] = torch.stack(diffs).mean(dim=0)

    def install_hook(vector: Any | None) -> Any | None:
        if vector is None:
            return None
        delta = (args.coefficient * vector).to(device)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0] + delta.view(1, 1, -1).to(output[0].dtype)
                return (hidden,) + output[1:]
            return output + delta.view(1, 1, -1).to(output.dtype)

        return blocks[block_index].register_forward_hook(hook)

    def generate(prompt: str, vector: Any | None = None) -> str:
        handle = install_hook(vector)
        try:
            formatted_prompt = format_user_prompt(prompt)
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, inputs["input_ids"].shape[1] :]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        finally:
            if handle is not None:
                handle.remove()

    def continuation_logprob(prompt: str, continuation: str, vector: Any | None = None) -> float:
        handle = None
        handle = install_hook(vector)
        try:
            formatted_prompt = format_user_prompt(prompt)
            prompt_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
            full_ids = tokenizer(formatted_prompt + continuation, return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] <= prompt_ids.shape[1]:
                return float("-inf")
            with torch.no_grad():
                logits = model(input_ids=full_ids).logits
                log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = full_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            continuation_mask = torch.zeros_like(target_ids, dtype=torch.bool)
            continuation_mask[:, max(prompt_ids.shape[1] - 1, 0) :] = True
            selected = token_log_probs[continuation_mask]
            return float(selected.mean().detach().cpu()) if selected.numel() else float("-inf")
        finally:
            if handle is not None:
                handle.remove()

    def preference(prompt: str, positive_answer: str, negative_answer: str, vector: Any | None = None) -> dict[str, Any]:
        positive_logprob = continuation_logprob(prompt, positive_answer, vector)
        negative_logprob = continuation_logprob(prompt, negative_answer, vector)
        margin = positive_logprob - negative_logprob
        return {
            "positive_logprob": positive_logprob,
            "negative_logprob": negative_logprob,
            "margin": margin,
            "success": margin > 0,
        }

    rows: list[dict[str, Any]] = []
    for task in tasks:
        behavior = task["behavior"]
        baseline_output = generate(task["prompt"])
        steered_output = generate(task["prompt"], vectors[behavior])
        rows.append(
            {
                "id": task["id"],
                "behavior": behavior,
                "prompt": task["prompt"],
                "vector_norm": float(vectors[behavior].norm().detach().cpu()),
                "baseline": {
                    "output": baseline_output,
                    "metrics": lexical_score(baseline_output, task["positive_terms"], task["negative_terms"]),
                },
                "steered": {
                    "output": steered_output,
                    "metrics": lexical_score(steered_output, task["positive_terms"], task["negative_terms"]),
                },
                "baseline_preference": preference(task["prompt"], task["positive_answer"], task["negative_answer"]),
                "steered_preference": preference(task["prompt"], task["positive_answer"], task["negative_answer"], vectors[behavior]),
            }
        )

    payload = {
        "backend": "hf",
        "config": {
            "model": args.model,
            "layer": args.layer,
            "resolved_layer": block_index,
            "coefficient": args.coefficient,
            "max_new_tokens": args.max_new_tokens,
            "allow_download": args.allow_download,
            "chat_template": args.chat_template,
        },
        "summary": summarize(rows),
        "rows": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(Path(args.out), payload)
    return payload


def run_hf_sweep(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("The hf-sweep backend requires torch and transformers.") from exc

    pairs = read_jsonl(Path(args.pairs))
    tasks = read_jsonl(Path(args.tasks))
    coefficients = parse_number_list(args.coefficients, float)
    local_files_only = not args.allow_download

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=local_files_only,
        torch_dtype="auto",
        device_map=args.device_map,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = find_transformer_blocks(model)
    if args.layers == "all":
        layer_indices = list(range(len(blocks)))
    else:
        layer_indices = [normalize_index(index, len(blocks)) for index in parse_number_list(args.layers, int)]
    for layer_index in layer_indices:
        if layer_index < 0 or layer_index >= len(blocks):
            raise ValueError(f"Layer index {layer_index} outside 0..{len(blocks)-1}.")

    device = next(model.parameters()).device

    def supports_chat_template() -> bool:
        return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")

    def format_user_prompt(text: str) -> str:
        if args.chat_template == "off" or not supports_chat_template():
            return text
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def format_behavior_example(text: str) -> str:
        if args.chat_template == "off" or not supports_chat_template():
            return text
        messages = [
            {"role": "user", "content": "Describe the response style the assistant should follow."},
            {"role": "assistant", "content": text},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def activations_for_layers(text: str) -> dict[int, Any]:
        inputs = tokenizer(format_behavior_example(text), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return {layer: outputs.hidden_states[layer + 1][0].mean(dim=0) for layer in layer_indices}

    vectors_by_layer: dict[int, dict[str, Any]] = {}
    for layer in layer_indices:
        vectors_by_layer[layer] = {}

    behaviors = sorted({row["behavior"] for row in pairs})
    diffs_by_layer_behavior: dict[int, dict[str, list[Any]]] = {
        layer: {behavior: [] for behavior in behaviors} for layer in layer_indices
    }
    for row in pairs:
        positive = activations_for_layers(row["positive"])
        negative = activations_for_layers(row["negative"])
        for layer in layer_indices:
            diffs_by_layer_behavior[layer][row["behavior"]].append(positive[layer] - negative[layer])

    for layer in layer_indices:
        for behavior in behaviors:
            vectors_by_layer[layer][behavior] = torch.stack(diffs_by_layer_behavior[layer][behavior]).mean(dim=0)

    def install_hook(layer: int, vector: Any | None, coefficient: float) -> Any | None:
        if vector is None or coefficient == 0:
            return None
        delta = (coefficient * vector).to(device)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0] + delta.view(1, 1, -1).to(output[0].dtype)
                return (hidden,) + output[1:]
            return output + delta.view(1, 1, -1).to(output.dtype)

        return blocks[layer].register_forward_hook(hook)

    def continuation_logprob(prompt: str, continuation: str, layer: int | None = None, vector: Any | None = None, coefficient: float = 0.0) -> float:
        handle = install_hook(layer, vector, coefficient) if layer is not None else None
        try:
            formatted_prompt = format_user_prompt(prompt)
            prompt_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
            full_ids = tokenizer(formatted_prompt + continuation, return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] <= prompt_ids.shape[1]:
                return float("-inf")
            with torch.no_grad():
                logits = model(input_ids=full_ids).logits
                log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = full_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            continuation_mask = torch.zeros_like(target_ids, dtype=torch.bool)
            continuation_mask[:, max(prompt_ids.shape[1] - 1, 0) :] = True
            selected = token_log_probs[continuation_mask]
            return float(selected.mean().detach().cpu()) if selected.numel() else float("-inf")
        finally:
            if handle is not None:
                handle.remove()

    baseline_preferences: dict[str, dict[str, Any]] = {}
    for task in tasks:
        positive_logprob = continuation_logprob(task["prompt"], task["positive_answer"])
        negative_logprob = continuation_logprob(task["prompt"], task["negative_answer"])
        margin = positive_logprob - negative_logprob
        baseline_preferences[task["id"]] = {
            "positive_logprob": positive_logprob,
            "negative_logprob": negative_logprob,
            "margin": margin,
            "success": margin > 0,
        }

    sweep_rows: list[dict[str, Any]] = []
    for layer in layer_indices:
        for coefficient in coefficients:
            task_rows: list[dict[str, Any]] = []
            for task in tasks:
                vector = vectors_by_layer[layer][task["behavior"]]
                positive_logprob = continuation_logprob(
                    task["prompt"],
                    task["positive_answer"],
                    layer=layer,
                    vector=vector,
                    coefficient=coefficient,
                )
                negative_logprob = continuation_logprob(
                    task["prompt"],
                    task["negative_answer"],
                    layer=layer,
                    vector=vector,
                    coefficient=coefficient,
                )
                margin = positive_logprob - negative_logprob
                task_rows.append(
                    {
                        "id": task["id"],
                        "behavior": task["behavior"],
                        "baseline_preference": baseline_preferences[task["id"]],
                        "steered_preference": {
                            "positive_logprob": positive_logprob,
                            "negative_logprob": negative_logprob,
                            "margin": margin,
                            "success": margin > 0,
                        },
                        "vector_norm": float(vector.norm().detach().cpu()),
                    }
                )
            summary = score_preference_summary(task_rows)
            sweep_rows.append(
                {
                    "layer": layer,
                    "coefficient": coefficient,
                    "summary": summary,
                    "rows": task_rows if args.include_rows else [],
                }
            )

    ranked = sorted(
        sweep_rows,
        key=lambda row: (
            row["summary"]["steered_preference_success"],
            row["summary"]["average_margin_lift"],
            row["summary"]["steered_average_margin"],
        ),
        reverse=True,
    )
    payload = {
        "backend": "hf-sweep",
        "config": {
            "model": args.model,
            "layers": layer_indices,
            "coefficients": coefficients,
            "allow_download": args.allow_download,
            "chat_template": args.chat_template,
            "include_rows": args.include_rows,
        },
        "best": ranked[0] if ranked else None,
        "ranked": ranked[: args.top_k],
        "sweep": sweep_rows,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(Path(args.out), payload)
    return payload


def run_hf_split_eval(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("The hf-split-eval backend requires torch and transformers.") from exc

    train_pairs = read_jsonl(Path(args.train_pairs))
    dev_tasks = read_jsonl(Path(args.dev_tasks))
    hidden_tasks = read_jsonl(Path(args.hidden_tasks))
    adversarial_tasks = read_jsonl(Path(args.adversarial_tasks))
    coefficients = parse_number_list(args.coefficients, float)
    vector_modes = parse_string_list(args.vector_modes)
    valid_vector_modes = {"raw", "orthogonalized"}
    invalid_vector_modes = sorted(set(vector_modes) - valid_vector_modes)
    if invalid_vector_modes:
        raise ValueError(f"Unknown vector mode(s): {', '.join(invalid_vector_modes)}.")
    local_files_only = not args.allow_download

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=local_files_only,
        torch_dtype="auto",
        device_map=args.device_map,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    blocks = find_transformer_blocks(model)
    if args.layers == "all":
        layer_indices = list(range(len(blocks)))
    else:
        layer_indices = [normalize_index(index, len(blocks)) for index in parse_number_list(args.layers, int)]
    for layer_index in layer_indices:
        if layer_index < 0 or layer_index >= len(blocks):
            raise ValueError(f"Layer index {layer_index} outside 0..{len(blocks)-1}.")

    device = next(model.parameters()).device
    behaviors = sorted({row["behavior"] for row in train_pairs})
    next_behavior = {behavior: behaviors[(index + 1) % len(behaviors)] for index, behavior in enumerate(behaviors)}

    def supports_chat_template() -> bool:
        return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")

    def format_user_prompt(text: str) -> str:
        if args.chat_template == "off" or not supports_chat_template():
            return text
        messages = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def format_behavior_example(text: str) -> str:
        if args.chat_template == "off" or not supports_chat_template():
            return text
        messages = [
            {"role": "user", "content": "Describe the response style the assistant should follow."},
            {"role": "assistant", "content": text},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def activations_for_layers(text: str) -> dict[int, Any]:
        inputs = tokenizer(format_behavior_example(text), return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return {layer: outputs.hidden_states[layer + 1][0].mean(dim=0) for layer in layer_indices}

    target_diffs: dict[int, dict[str, list[Any]]] = {
        layer: {behavior: [] for behavior in behaviors} for layer in layer_indices
    }
    shuffled_diffs: dict[int, dict[str, list[Any]]] = {
        layer: {behavior: [] for behavior in behaviors} for layer in layer_indices
    }
    for row in train_pairs:
        positive = activations_for_layers(row["positive"])
        negative = activations_for_layers(row["negative"])
        shuffled_behavior = next_behavior[row["behavior"]]
        for layer in layer_indices:
            diff = positive[layer] - negative[layer]
            target_diffs[layer][row["behavior"]].append(diff)
            shuffled_diffs[layer][shuffled_behavior].append(diff)

    target_vectors: dict[int, dict[str, Any]] = {layer: {} for layer in layer_indices}
    shuffled_vectors: dict[int, dict[str, Any]] = {layer: {} for layer in layer_indices}
    for layer in layer_indices:
        for behavior in behaviors:
            target_vectors[layer][behavior] = torch.stack(target_diffs[layer][behavior]).mean(dim=0)
            shuffled_vectors[layer][behavior] = torch.stack(shuffled_diffs[layer][behavior]).mean(dim=0)

    def orthogonalize_vector(vector: Any, basis: Any) -> Any:
        vector_float = vector.float()
        basis_float = basis.to(vector.device).float()
        denom = torch.dot(basis_float.flatten(), basis_float.flatten())
        if float(denom.detach().cpu()) <= 1e-12:
            return vector.clone()
        projection_scale = torch.dot(vector_float.flatten(), basis_float.flatten()) / denom
        projected = vector_float - projection_scale * basis_float
        if args.preserve_orthogonalized_norm:
            original_norm = vector_float.norm()
            projected_norm = projected.norm()
            if float(projected_norm.detach().cpu()) > 1e-12:
                projected = projected * (original_norm / projected_norm)
        return projected.to(dtype=vector.dtype)

    shared_vectors: dict[int, Any] = {}
    orthogonalized_vectors: dict[int, dict[str, Any]] = {layer: {} for layer in layer_indices}
    orthogonalized_shuffled_vectors: dict[int, dict[str, Any]] = {layer: {} for layer in layer_indices}
    for layer in layer_indices:
        shared_vectors[layer] = torch.stack([target_vectors[layer][behavior] for behavior in behaviors]).mean(dim=0)
        for behavior in behaviors:
            orthogonalized_vectors[layer][behavior] = orthogonalize_vector(
                target_vectors[layer][behavior],
                shared_vectors[layer],
            )
            orthogonalized_shuffled_vectors[layer][behavior] = orthogonalize_vector(
                shuffled_vectors[layer][behavior],
                shared_vectors[layer],
            )

    vector_sets = {
        "raw": target_vectors,
        "orthogonalized": orthogonalized_vectors,
    }
    shuffled_vector_sets = {
        "raw": shuffled_vectors,
        "orthogonalized": orthogonalized_shuffled_vectors,
    }

    def random_like(vector: Any, key: str) -> Any:
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % (2**31)
        generator = torch.Generator(device=vector.device)
        generator.manual_seed(seed)
        noise = torch.randn(vector.shape, dtype=vector.dtype, device=vector.device, generator=generator)
        noise_norm = noise.norm()
        vector_norm = vector.norm()
        if float(noise_norm.detach().cpu()) == 0.0:
            return noise
        return noise * (vector_norm / noise_norm)

    def install_hook(layer: int, vector: Any | None, coefficient: float) -> Any | None:
        if vector is None or coefficient == 0:
            return None
        delta = (coefficient * vector).to(device)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0] + delta.view(1, 1, -1).to(output[0].dtype)
                return (hidden,) + output[1:]
            return output + delta.view(1, 1, -1).to(output.dtype)

        return blocks[layer].register_forward_hook(hook)

    def continuation_logprob(prompt: str, continuation: str, layer: int | None = None, vector: Any | None = None, coefficient: float = 0.0) -> float:
        handle = install_hook(layer, vector, coefficient) if layer is not None else None
        try:
            formatted_prompt = format_user_prompt(prompt)
            prompt_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
            full_ids = tokenizer(formatted_prompt + continuation, return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] <= prompt_ids.shape[1]:
                return float("-inf")
            with torch.no_grad():
                logits = model(input_ids=full_ids).logits
                log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = full_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            continuation_mask = torch.zeros_like(target_ids, dtype=torch.bool)
            continuation_mask[:, max(prompt_ids.shape[1] - 1, 0) :] = True
            selected = token_log_probs[continuation_mask]
            return float(selected.mean().detach().cpu()) if selected.numel() else float("-inf")
        finally:
            if handle is not None:
                handle.remove()

    def preference(task: dict[str, Any], layer: int | None = None, vector: Any | None = None, coefficient: float = 0.0) -> dict[str, Any]:
        positive_logprob = continuation_logprob(task["prompt"], task["positive_answer"], layer, vector, coefficient)
        negative_logprob = continuation_logprob(task["prompt"], task["negative_answer"], layer, vector, coefficient)
        margin = positive_logprob - negative_logprob
        return {
            "positive_logprob": positive_logprob,
            "negative_logprob": negative_logprob,
            "margin": margin,
            "success": margin > 0,
        }

    def summarize_variant(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
        projected = [
            {
                "baseline_preference": row["baseline_preference"],
                "steered_preference": row["variants"][variant],
            }
            for row in rows
        ]
        return score_preference_summary(projected)

    def evaluate_target(tasks: list[dict[str, Any]], layer: int, coefficient: float, vector_mode: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for task in tasks:
            behavior = task["behavior"]
            vector = vector_sets[vector_mode][layer][behavior]
            rows.append(
                {
                    "id": task["id"],
                    "behavior": behavior,
                    "baseline_preference": preference(task),
                    "variants": {
                        "target": preference(task, layer, vector, coefficient),
                    },
                }
            )
        return rows

    dev_sweep: list[dict[str, Any]] = []
    for vector_mode in vector_modes:
        for layer in layer_indices:
            for coefficient in coefficients:
                rows = evaluate_target(dev_tasks, layer, coefficient, vector_mode)
                summary = summarize_variant(rows, "target")
                dev_sweep.append(
                    {
                        "vector_mode": vector_mode,
                        "layer": layer,
                        "coefficient": coefficient,
                        "summary": summary,
                        "rows": rows if args.include_rows else [],
                    }
                )

    ranked_dev = sorted(
        dev_sweep,
        key=lambda row: (
            row["summary"]["steered_preference_success"],
            row["summary"]["average_margin_lift"],
            row["summary"]["steered_average_margin"],
        ),
        reverse=True,
    )
    selected = ranked_dev[0]
    selected_vector_mode = selected["vector_mode"]
    selected_layer = selected["layer"]
    selected_coefficient = selected["coefficient"]

    def control_vectors(layer: int, behavior: str, vector_mode: str) -> dict[str, Any]:
        target = vector_sets[vector_mode][layer][behavior]
        wrong = vector_sets[vector_mode][layer][next_behavior[behavior]]
        return {
            "target": target,
            "random": random_like(target, f"{vector_mode}:{layer}:{behavior}:random"),
            "reversed": -target,
            "shuffled_label": shuffled_vector_sets[vector_mode][layer][behavior],
            "wrong_behavior": wrong,
        }

    def evaluate_split(name: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for task in tasks:
            behavior = task["behavior"]
            baseline = preference(task)
            variants: dict[str, Any] = {}
            for variant, vector in control_vectors(selected_layer, behavior, selected_vector_mode).items():
                variants[variant] = preference(task, selected_layer, vector, selected_coefficient)
            rows.append(
                {
                    "id": task["id"],
                    "behavior": behavior,
                    "baseline_preference": baseline,
                    "variants": variants,
                }
            )
        summaries = {variant: summarize_variant(rows, variant) for variant in rows[0]["variants"]}
        control_names = [name for name in summaries if name != "target"]
        best_control = max(
            control_names,
            key=lambda control: (
                summaries[control]["steered_preference_success"],
                summaries[control]["average_margin_lift"],
            ),
        )
        target = summaries["target"]
        return {
            "name": name,
            "summaries": summaries,
            "best_control": {
                "name": best_control,
                "summary": summaries[best_control],
            },
            "target_beats_best_control": (
                target["steered_preference_success"] > summaries[best_control]["steered_preference_success"]
                or (
                    target["steered_preference_success"] == summaries[best_control]["steered_preference_success"]
                    and target["average_margin_lift"] > summaries[best_control]["average_margin_lift"]
                )
            ),
            "rows": rows if args.include_rows else [],
        }

    split_results = {
        "dev": evaluate_split("dev", dev_tasks),
        "hidden": evaluate_split("hidden", hidden_tasks),
        "adversarial": evaluate_split("adversarial", adversarial_tasks),
    }

    hidden_target = split_results["hidden"]["summaries"]["target"]
    hidden_best_control = split_results["hidden"]["best_control"]["summary"]
    adversarial_target = split_results["adversarial"]["summaries"]["target"]
    adversarial_best_control = split_results["adversarial"]["best_control"]["summary"]
    gate_checks = {
        "hidden_success_lift": hidden_target["preference_success_lift"] >= args.min_hidden_success_lift,
        "hidden_margin_lift": hidden_target["average_margin_lift"] >= args.min_hidden_margin_lift,
        "hidden_beats_controls": split_results["hidden"]["target_beats_best_control"],
        "adversarial_nonnegative_margin_lift": adversarial_target["average_margin_lift"] >= 0,
        "adversarial_nonnegative_success_lift": adversarial_target["preference_success_lift"] >= 0,
        "adversarial_beats_controls": split_results["adversarial"]["target_beats_best_control"],
    }
    gate = {
        "passed": all(gate_checks.values()),
        "checks": gate_checks,
        "thresholds": {
            "min_hidden_success_lift": args.min_hidden_success_lift,
            "min_hidden_margin_lift": args.min_hidden_margin_lift,
        },
        "hidden_best_control_margin_lift": hidden_best_control["average_margin_lift"],
        "adversarial_best_control_margin_lift": adversarial_best_control["average_margin_lift"],
    }

    payload = {
        "backend": "hf-split-eval",
        "config": {
            "model": args.model,
            "train_pairs": str(Path(args.train_pairs)),
            "dev_tasks": str(Path(args.dev_tasks)),
            "hidden_tasks": str(Path(args.hidden_tasks)),
            "adversarial_tasks": str(Path(args.adversarial_tasks)),
            "layers": layer_indices,
            "coefficients": coefficients,
            "vector_modes": vector_modes,
            "preserve_orthogonalized_norm": args.preserve_orthogonalized_norm,
            "allow_download": args.allow_download,
            "chat_template": args.chat_template,
            "include_rows": args.include_rows,
        },
        "selection": {
            "selected_on": "dev",
            "vector_mode": selected_vector_mode,
            "layer": selected_layer,
            "coefficient": selected_coefficient,
            "summary": selected["summary"],
        },
        "ranked_dev": ranked_dev[: args.top_k],
        "splits": split_results,
        "gate": gate,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(Path(args.out), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="backend", required=True)

    toy = subparsers.add_parser("toy", help="Run deterministic steering-plumbing benchmark.")
    toy.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    toy.add_argument("--tasks", default=str(DEFAULT_TASKS))
    toy.add_argument("--out", default="results/activation_steering/toy_run.json")
    toy.add_argument("--dims", type=int, default=12)
    toy.add_argument("--coefficient", type=float, default=1.0)
    toy.set_defaults(func=run_toy)

    hf = subparsers.add_parser("hf", help="Run optional activation steering on a local HF causal LM.")
    hf.add_argument("--model", required=True, help="Local path or Hugging Face model name.")
    hf.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    hf.add_argument("--tasks", default=str(DEFAULT_TASKS))
    hf.add_argument("--out", default="results/activation_steering/hf_run.json")
    hf.add_argument("--layer", type=int, default=-1)
    hf.add_argument("--coefficient", type=float, default=1.0)
    hf.add_argument("--max-new-tokens", type=int, default=64)
    hf.add_argument("--device-map", default="auto")
    hf.add_argument("--allow-download", action="store_true")
    hf.add_argument("--chat-template", choices=["auto", "off"], default="auto")
    hf.set_defaults(func=run_hf)

    sweep = subparsers.add_parser("hf-sweep", help="Sweep HF activation steering layers and coefficients.")
    sweep.add_argument("--model", required=True, help="Local path or Hugging Face model name.")
    sweep.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    sweep.add_argument("--tasks", default=str(DEFAULT_TASKS))
    sweep.add_argument("--out", default="results/activation_steering/hf_sweep.json")
    sweep.add_argument("--layers", default="all", help="Comma-separated layers, negative indexes, or 'all'.")
    sweep.add_argument("--coefficients", default="0,1,2,5,8")
    sweep.add_argument("--device-map", default="auto")
    sweep.add_argument("--allow-download", action="store_true")
    sweep.add_argument("--chat-template", choices=["auto", "off"], default="auto")
    sweep.add_argument("--include-rows", action="store_true", help="Include per-task rows for every sweep point.")
    sweep.add_argument("--top-k", type=int, default=10)
    sweep.set_defaults(func=run_hf_sweep)

    split_eval = subparsers.add_parser("hf-split-eval", help="Tune on dev, then evaluate hidden/adversarial with controls.")
    split_eval.add_argument("--model", required=True, help="Local path or Hugging Face model name.")
    split_eval.add_argument("--train-pairs", default=str(DEFAULT_TRAIN_PAIRS))
    split_eval.add_argument("--dev-tasks", default=str(DEFAULT_DEV_TASKS))
    split_eval.add_argument("--hidden-tasks", default=str(DEFAULT_HIDDEN_TASKS))
    split_eval.add_argument("--adversarial-tasks", default=str(DEFAULT_ADVERSARIAL_TASKS))
    split_eval.add_argument("--out", default="results/activation_steering/hf_split_eval.json")
    split_eval.add_argument("--layers", default="8,12,16,20,24,28,29")
    split_eval.add_argument("--coefficients", default="0,1,2,5,8,12")
    split_eval.add_argument(
        "--vector-modes",
        default="raw,orthogonalized",
        help="Comma-separated vector modes to tune on dev: raw, orthogonalized.",
    )
    split_eval.add_argument(
        "--no-preserve-orthogonalized-norm",
        action="store_false",
        dest="preserve_orthogonalized_norm",
        help="Do not rescale orthogonalized vectors back to the raw vector norm.",
    )
    split_eval.add_argument("--device-map", default="auto")
    split_eval.add_argument("--allow-download", action="store_true")
    split_eval.add_argument("--chat-template", choices=["auto", "off"], default="auto")
    split_eval.add_argument("--include-rows", action="store_true", help="Include per-task rows for each split.")
    split_eval.add_argument("--top-k", type=int, default=10)
    split_eval.add_argument("--min-hidden-success-lift", type=float, default=0.15)
    split_eval.add_argument("--min-hidden-margin-lift", type=float, default=0.5)
    split_eval.set_defaults(preserve_orthogonalized_norm=True)
    split_eval.set_defaults(func=run_hf_split_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = args.func(args)
    if "summary" in payload:
        printable = payload["summary"]
    elif payload.get("best"):
        printable = {"best": payload["best"], "elapsed_seconds": payload["elapsed_seconds"]}
    elif payload.get("gate"):
        printable = {
            "selection": payload["selection"],
            "hidden_target": payload["splits"]["hidden"]["summaries"]["target"],
            "hidden_best_control": payload["splits"]["hidden"]["best_control"],
            "adversarial_target": payload["splits"]["adversarial"]["summaries"]["target"],
            "gate": payload["gate"],
            "elapsed_seconds": payload["elapsed_seconds"],
        }
    else:
        printable = payload
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
