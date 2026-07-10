#!/usr/bin/env python3
"""Memory-as-steering vs in-context memory harness.

This thread tests IDEA-20260709-03 / the pre-registration in `PREREGISTRATION.md`:
does the best channel for delivering a retrieved memory depend on the memory *type*?

Two channels deliver the SAME memory content:
- `in_context`: paste the memory text into the prompt.
- `activation`: inject a residual-stream steering vector derived from the memory text.

Two memory types:
- `procedural`: how to act (verify, clarify, be concise).
- `factual`: a specific project fact (the config delimiter is a pipe).

Pre-registered prediction (H-AM-01): a crossover interaction -- activation wins for
procedural memory, in-context wins for factual memory.

Backends:
- `toy`: deterministic, dependency-light. Proves the harness computes channels, the
  type x channel interaction, and the gate correctly. Its crossover is BUILT IN by the
  toy's mechanics (a behavior is a continuous axis a vector can move; a fact is a discrete
  token only text can supply). It is NOT evidence that the effect holds in a real LM.
- `hf`: optional real causal LM (torch + transformers). This is the confirmatory backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

# Allow direct-script invocation (python experiments/.../memory_harness.py) as well as
# module invocation by putting the repo root on the path before the package import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Reuse the steering thread's IO + toy vocabulary so both threads stay consistent.
from experiments.activation_steering.steering_harness import (
    TOY_DIMS,
    TOY_NEGATIVE_TERMS,
    TOY_POSITIVE_TERMS,
    add_vectors,
    find_transformer_blocks,
    normalize_index,
    parse_number_list,
    read_jsonl,
    scale_vector,
    stable_noise,
    sub_vectors,
    write_json,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_MEMORIES = DATA_DIR / "memory_items.jsonl"
DEFAULT_DEV = DATA_DIR / "tasks_dev.jsonl"
DEFAULT_HIDDEN = DATA_DIR / "tasks_hidden.jsonl"
DEFAULT_ADVERSARIAL = DATA_DIR / "tasks_adversarial.jsonl"

NEUTRAL_REFERENCE = "Working notes: respond to the user."

MEMORY_TYPES = ("procedural", "factual")
# 'none' is the baseline every lift is measured against.
CHANNELS = (
    "none",
    "in_context",
    "in_context_wrong",
    "activation",
    "activation_random",
    "activation_wrong",
)


def load_memories(path: Path) -> dict[str, dict[str, Any]]:
    memories = {row["id"]: row for row in read_jsonl(path)}
    if not memories:
        raise ValueError(f"No memory items found in {path}.")
    return memories


def wrong_memory_id(memory_id: str, memories: dict[str, dict[str, Any]]) -> str:
    """Pick a different memory of the same type -- a fair same-type control."""
    this = memories[memory_id]
    same_type = [mid for mid, m in memories.items() if m["type"] == this["type"] and mid != memory_id]
    if not same_type:
        raise ValueError(f"Need at least two {this['type']} memories to build a wrong-memory control.")
    # Deterministic pick so runs are reproducible.
    return sorted(same_type)[0]


# --------------------------------------------------------------------------------------
# Aggregation + gate (shared by toy and hf)
# --------------------------------------------------------------------------------------

def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per (type, channel): mean margin lift over baseline and preference success rate."""
    result: dict[str, Any] = {}
    for mtype in MEMORY_TYPES:
        typed = [r for r in rows if r["type"] == mtype]
        per_channel: dict[str, Any] = {}
        for channel in CHANNELS:
            if not typed:
                per_channel[channel] = {"tasks": 0, "success_rate": 0.0, "margin_lift": 0.0}
                continue
            lifts = [r["channels"][channel]["margin"] - r["channels"]["none"]["margin"] for r in typed]
            successes = sum(1 for r in typed if r["channels"][channel]["success"])
            per_channel[channel] = {
                "tasks": len(typed),
                "success_rate": successes / len(typed),
                "margin_lift": sum(lifts) / len(lifts),
            }
        result[mtype] = per_channel
    return result


def interaction_and_gate(agg: dict[str, Any], min_gap: float = 0.5) -> dict[str, Any]:
    proc = agg["procedural"]
    fact = agg["factual"]
    adv_proc = proc["activation"]["margin_lift"] - proc["in_context"]["margin_lift"]
    adv_fact = fact["activation"]["margin_lift"] - fact["in_context"]["margin_lift"]
    gap = adv_proc - adv_fact

    checks = {
        "interaction_gap_met": gap >= min_gap,
        "activation_wins_procedural": adv_proc > 0,
        "in_context_wins_factual": adv_fact < 0,
        "procedural_activation_beats_random": (
            proc["activation"]["margin_lift"] > proc["activation_random"]["margin_lift"]
        ),
        "procedural_activation_beats_wrong": (
            proc["activation"]["margin_lift"] > proc["activation_wrong"]["margin_lift"]
        ),
        "factual_in_context_beats_wrong": (
            fact["in_context"]["margin_lift"] > fact["in_context_wrong"]["margin_lift"]
        ),
    }
    if not checks["interaction_gap_met"]:
        verdict = "Refuted"
    elif checks["activation_wins_procedural"] and checks["in_context_wins_factual"]:
        verdict = "Confirmed" if all(checks.values()) else "Partial: crossover but a control failed"
    else:
        verdict = "Partial: interaction without crossover"
    return {
        "adv_proc": adv_proc,
        "adv_fact": adv_fact,
        "interaction_gap": gap,
        "min_gap": min_gap,
        "checks": checks,
        "passed": all(checks.values()),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------------------
# Toy backend
# --------------------------------------------------------------------------------------

class ToyMemoryModel:
    """Deterministic model that makes the pre-registered mechanism observable.

    Assumption A (behavior is a continuous axis): both the memory *text* and the memory
    *vector* push a behavior axis, but the distilled vector pushes it harder per unit, so
    activation >= in_context for procedural memory.

    Assumption B (a fact is a discrete token): the correct factual answer is only reachable
    when the fact string is present in the *context text*, which a mean-pooled vector cannot
    supply, so in_context > activation for factual memory.

    These assumptions are the whole point of the toy: it proves the harness measures the
    interaction, not that the interaction is real in an LM. Only the `hf` backend can do that.
    """

    def __init__(self, dims: int = 12, coefficient: float = 1.5) -> None:
        self.dims = dims
        self.coefficient = coefficient

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

    def memory_vector(self, memory_text: str) -> list[float]:
        return sub_vectors(self.encode(memory_text), self.encode(NEUTRAL_REFERENCE))

    def random_vector(self, key: str) -> list[float]:
        # Reuse stable_noise so the toy stays dependency-free and reproducible.
        raw = stable_noise(key, self.dims)
        return scale_vector(raw, 20.0)  # comparable scale, but not axis-aligned

    def _context_vector(self, channel: str, prompt: str, mem: dict[str, Any], wrong: dict[str, Any]) -> list[float]:
        base = self.encode(prompt)
        if channel in ("none",):
            return base
        if channel == "in_context":
            return self.encode(f"{mem['text']} {prompt}")
        if channel == "in_context_wrong":
            return self.encode(f"{wrong['text']} {prompt}")
        if channel == "activation":
            return add_vectors(base, scale_vector(self.memory_vector(mem["text"]), self.coefficient))
        if channel == "activation_random":
            return add_vectors(base, scale_vector(self.random_vector(mem["id"]), self.coefficient))
        if channel == "activation_wrong":
            return add_vectors(base, scale_vector(self.memory_vector(wrong["text"]), self.coefficient))
        raise ValueError(f"Unknown channel {channel}.")

    def _context_text(self, channel: str, prompt: str, mem: dict[str, Any], wrong: dict[str, Any]) -> str:
        if channel == "in_context":
            return f"{mem['text']} {prompt}"
        if channel == "in_context_wrong":
            return f"{wrong['text']} {prompt}"
        # Activation channels never add tokens to the context text.
        return prompt

    def preference(self, task: dict[str, Any], channel: str, mem: dict[str, Any], wrong: dict[str, Any]) -> dict[str, Any]:
        if task["type"] == "procedural":
            axis = TOY_DIMS[mem["behavior"]]
            ctx = self._context_vector(channel, task["prompt"], mem, wrong)
            # Positive answer aligns with the behavior axis; negative opposes it.
            behavior_signal = ctx[axis]
            margin = behavior_signal - 0.5  # baseline prompt alone stays under the bar
        else:  # factual
            ctx_text = self._context_text(channel, task["prompt"], mem, wrong).lower()
            available = mem["fact_value"] is not None and str(mem["fact_value"]).lower() in ctx_text
            wrong_present = wrong["fact_value"] is not None and str(wrong["fact_value"]).lower() in ctx_text
            margin = (2.0 if available else -0.5) - (0.5 if wrong_present else 0.0)
        return {"margin": margin, "success": margin > 0}


def run_toy(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    memories = load_memories(Path(args.memories))
    tasks = read_jsonl(Path(args.tasks))
    model = ToyMemoryModel(dims=args.dims, coefficient=args.coefficient)

    rows: list[dict[str, Any]] = []
    for task in tasks:
        mem = memories[task["memory_id"]]
        wrong = memories[wrong_memory_id(task["memory_id"], memories)]
        channels = {ch: model.preference(task, ch, mem, wrong) for ch in CHANNELS}
        rows.append(
            {
                "id": task["id"],
                "type": task["type"],
                "memory_id": task["memory_id"],
                "behavior": mem.get("behavior"),
                "channels": channels,
            }
        )

    agg = aggregate(rows)
    gate = interaction_and_gate(agg, min_gap=args.min_gap)
    payload = {
        "backend": "toy",
        "config": {"memories": str(Path(args.memories)), "tasks": str(Path(args.tasks)), "dims": args.dims, "coefficient": args.coefficient},
        "aggregate": agg,
        "interaction": gate,
        "rows": rows if args.include_rows else [],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(Path(args.out), payload)
    return payload


# --------------------------------------------------------------------------------------
# HF backend (confirmatory)
# --------------------------------------------------------------------------------------

def run_hf(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("The hf backend requires torch and transformers.") from exc

    memories = load_memories(Path(args.memories))
    tasks = read_jsonl(Path(args.tasks))
    local_files_only = not args.allow_download

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=local_files_only, torch_dtype="auto", device_map=args.device_map
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

    def format_prompt(text: str, system: str | None = None) -> str:
        if args.chat_template == "off" or not supports_chat_template():
            return f"{system}\n\n{text}" if system else text
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def mean_hidden(text: str) -> Any:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[block_index + 1][0].mean(dim=0)

    # memory_vector = unit( mean_hidden(memory.text) - mean_hidden(NEUTRAL) ), per pre-registration.
    neutral_hidden = mean_hidden(NEUTRAL_REFERENCE)
    memory_vectors: dict[str, Any] = {}
    for mid, mem in memories.items():
        raw = mean_hidden(mem["text"]) - neutral_hidden
        norm = raw.norm()
        memory_vectors[mid] = raw / norm if float(norm.detach().cpu()) > 1e-8 else raw

    def random_like(vector: Any, key: str) -> Any:
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % (2**31)
        generator = torch.Generator(device=vector.device)
        generator.manual_seed(seed)
        noise = torch.randn(vector.shape, dtype=vector.dtype, device=vector.device, generator=generator)
        norm = noise.norm()
        return noise / norm if float(norm.detach().cpu()) > 1e-8 else noise

    def install_hook(vector: Any | None) -> Any | None:
        if vector is None:
            return None
        delta = (args.coefficient * vector).to(device)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                return (output[0] + delta.view(1, 1, -1).to(output[0].dtype),) + output[1:]
            return output + delta.view(1, 1, -1).to(output.dtype)

        return blocks[block_index].register_forward_hook(hook)

    def continuation_logprob(context: str, continuation: str, vector: Any | None) -> float:
        handle = install_hook(vector)
        try:
            ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
            full_ids = tokenizer(context + continuation, return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] <= ctx_ids.shape[1]:
                return float("-inf")
            with torch.no_grad():
                logits = model(input_ids=full_ids).logits
                log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = full_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            mask = torch.zeros_like(target_ids, dtype=torch.bool)
            mask[:, max(ctx_ids.shape[1] - 1, 0):] = True
            selected = token_log_probs[mask]
            return float(selected.mean().detach().cpu()) if selected.numel() else float("-inf")
        finally:
            if handle is not None:
                handle.remove()

    def channel_inputs(channel: str, task: dict[str, Any], mem: dict[str, Any], wrong: dict[str, Any]) -> tuple[str, Any | None]:
        """Return (context_string, steering_vector) for a channel."""
        if channel == "none":
            return format_prompt(task["prompt"]), None
        if channel == "in_context":
            return format_prompt(task["prompt"], system=mem["text"]), None
        if channel == "in_context_wrong":
            return format_prompt(task["prompt"], system=wrong["text"]), None
        if channel == "activation":
            return format_prompt(task["prompt"]), memory_vectors[mem["id"]]
        if channel == "activation_random":
            return format_prompt(task["prompt"]), random_like(memory_vectors[mem["id"]], f"rand:{mem['id']}")
        if channel == "activation_wrong":
            return format_prompt(task["prompt"]), memory_vectors[wrong["id"]]
        raise ValueError(f"Unknown channel {channel}.")

    def preference(channel: str, task: dict[str, Any], mem: dict[str, Any], wrong: dict[str, Any]) -> dict[str, Any]:
        context, vector = channel_inputs(channel, task, mem, wrong)
        pos = continuation_logprob(context, " " + task["positive_answer"], vector)
        neg = continuation_logprob(context, " " + task["negative_answer"], vector)
        margin = pos - neg
        return {"margin": margin, "success": margin > 0}

    rows: list[dict[str, Any]] = []
    for task in tasks:
        mem = memories[task["memory_id"]]
        wrong = memories[wrong_memory_id(task["memory_id"], memories)]
        channels = {ch: preference(ch, task, mem, wrong) for ch in CHANNELS}
        rows.append(
            {
                "id": task["id"],
                "type": task["type"],
                "memory_id": task["memory_id"],
                "behavior": mem.get("behavior"),
                "channels": channels,
            }
        )

    agg = aggregate(rows)
    gate = interaction_and_gate(agg, min_gap=args.min_gap)
    payload = {
        "backend": "hf",
        "config": {
            "model": args.model,
            "layer": args.layer,
            "resolved_layer": block_index,
            "coefficient": args.coefficient,
            "memories": str(Path(args.memories)),
            "tasks": str(Path(args.tasks)),
            "allow_download": args.allow_download,
            "chat_template": args.chat_template,
        },
        "aggregate": agg,
        "interaction": gate,
        "rows": rows if args.include_rows else [],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(Path(args.out), payload)
    return payload


def rank_sweep(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank dev-sweep points by the pre-registered objective (biggest crossover first).

    Pure function so the ranking logic is unit-testable without a model. The dev objective
    is the interaction gap, then how much activation wins procedural, then raw procedural
    activation lift as a tie-break.
    """
    return sorted(
        points,
        key=lambda p: (
            p["interaction"]["interaction_gap"],
            p["interaction"]["adv_proc"],
            p["aggregate"]["procedural"]["activation"]["margin_lift"],
        ),
        reverse=True,
    )


def run_hf_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Tune layer/coefficient on the given tasks (use dev!) by the pre-registered objective."""
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("The hf-sweep backend requires torch and transformers.") from exc

    memories = load_memories(Path(args.memories))
    tasks = read_jsonl(Path(args.tasks))
    coefficients = parse_number_list(args.coefficients, float)
    local_files_only = not args.allow_download

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=local_files_only, torch_dtype="auto", device_map=args.device_map
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

    def format_prompt(text: str, system: str | None = None) -> str:
        if args.chat_template == "off" or not supports_chat_template():
            return f"{system}\n\n{text}" if system else text
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": text})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def mean_hidden_all_layers(text: str) -> dict[int, Any]:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return {layer: outputs.hidden_states[layer + 1][0].mean(dim=0) for layer in layer_indices}

    # One forward pass per memory text gives every swept layer at once.
    neutral_all = mean_hidden_all_layers(NEUTRAL_REFERENCE)
    memory_vectors: dict[int, dict[str, Any]] = {layer: {} for layer in layer_indices}
    for mid, mem in memories.items():
        text_all = mean_hidden_all_layers(mem["text"])
        for layer in layer_indices:
            raw = text_all[layer] - neutral_all[layer]
            norm = raw.norm()
            memory_vectors[layer][mid] = raw / norm if float(norm.detach().cpu()) > 1e-8 else raw

    def random_like(vector: Any, key: str) -> Any:
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % (2**31)
        generator = torch.Generator(device=vector.device)
        generator.manual_seed(seed)
        noise = torch.randn(vector.shape, dtype=vector.dtype, device=vector.device, generator=generator)
        norm = noise.norm()
        return noise / norm if float(norm.detach().cpu()) > 1e-8 else noise

    def install_hook(layer: int, vector: Any | None, coefficient: float) -> Any | None:
        if vector is None or coefficient == 0:
            return None
        delta = (coefficient * vector).to(device)

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            if isinstance(output, tuple):
                return (output[0] + delta.view(1, 1, -1).to(output[0].dtype),) + output[1:]
            return output + delta.view(1, 1, -1).to(output.dtype)

        return blocks[layer].register_forward_hook(hook)

    def continuation_logprob(context: str, continuation: str, layer: int | None, vector: Any | None, coefficient: float) -> float:
        handle = install_hook(layer, vector, coefficient) if layer is not None else None
        try:
            ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
            full_ids = tokenizer(context + continuation, return_tensors="pt").input_ids.to(device)
            if full_ids.shape[1] <= ctx_ids.shape[1]:
                return float("-inf")
            with torch.no_grad():
                logits = model(input_ids=full_ids).logits
                log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = full_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            mask = torch.zeros_like(target_ids, dtype=torch.bool)
            mask[:, max(ctx_ids.shape[1] - 1, 0):] = True
            selected = token_log_probs[mask]
            return float(selected.mean().detach().cpu()) if selected.numel() else float("-inf")
        finally:
            if handle is not None:
                handle.remove()

    def margin(context: str, task: dict[str, Any], layer: int | None, vector: Any | None, coefficient: float) -> dict[str, Any]:
        pos = continuation_logprob(context, " " + task["positive_answer"], layer, vector, coefficient)
        neg = continuation_logprob(context, " " + task["negative_answer"], layer, vector, coefficient)
        return {"margin": pos - neg, "success": pos - neg > 0}

    # Hook-free channels (none, in_context, in_context_wrong) never depend on layer/coeff -- compute once.
    hook_free: dict[str, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        mem = memories[task["memory_id"]]
        wrong = memories[wrong_memory_id(task["memory_id"], memories)]
        hook_free[task["id"]] = {
            "none": margin(format_prompt(task["prompt"]), task, None, None, 0.0),
            "in_context": margin(format_prompt(task["prompt"], system=mem["text"]), task, None, None, 0.0),
            "in_context_wrong": margin(format_prompt(task["prompt"], system=wrong["text"]), task, None, None, 0.0),
        }

    sweep_points: list[dict[str, Any]] = []
    for layer in layer_indices:
        for coefficient in coefficients:
            rows: list[dict[str, Any]] = []
            for task in tasks:
                mem = memories[task["memory_id"]]
                wrong = memories[wrong_memory_id(task["memory_id"], memories)]
                context = format_prompt(task["prompt"])
                channels = dict(hook_free[task["id"]])
                channels["activation"] = margin(context, task, layer, memory_vectors[layer][mem["id"]], coefficient)
                channels["activation_random"] = margin(
                    context, task, layer, random_like(memory_vectors[layer][mem["id"]], f"rand:{mem['id']}:{layer}"), coefficient
                )
                channels["activation_wrong"] = margin(context, task, layer, memory_vectors[layer][wrong["id"]], coefficient)
                rows.append({"id": task["id"], "type": task["type"], "memory_id": task["memory_id"], "behavior": mem.get("behavior"), "channels": channels})
            agg = aggregate(rows)
            sweep_points.append({"layer": layer, "coefficient": coefficient, "aggregate": agg, "interaction": interaction_and_gate(agg, min_gap=args.min_gap)})

    ranked = rank_sweep(sweep_points)
    payload = {
        "backend": "hf-sweep",
        "config": {
            "model": args.model,
            "layers": layer_indices,
            "coefficients": coefficients,
            "tasks": str(Path(args.tasks)),
            "min_gap": args.min_gap,
            "allow_download": args.allow_download,
            "chat_template": args.chat_template,
        },
        "best": ranked[0] if ranked else None,
        "ranked": ranked[: args.top_k],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    write_json(Path(args.out), payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="backend", required=True)

    toy = subparsers.add_parser("toy", help="Deterministic plumbing + interaction check.")
    toy.add_argument("--memories", default=str(DEFAULT_MEMORIES))
    toy.add_argument("--tasks", default=str(DEFAULT_HIDDEN))
    toy.add_argument("--out", default="results/activation_memory/toy_run.json")
    toy.add_argument("--dims", type=int, default=12)
    toy.add_argument("--coefficient", type=float, default=1.5)
    toy.add_argument("--min-gap", type=float, default=0.5)
    toy.add_argument("--include-rows", action="store_true")
    toy.set_defaults(func=run_toy)

    hf = subparsers.add_parser("hf", help="Confirmatory run on a local HF causal LM.")
    hf.add_argument("--model", required=True, help="Local path or Hugging Face model name.")
    hf.add_argument("--memories", default=str(DEFAULT_MEMORIES))
    hf.add_argument("--tasks", default=str(DEFAULT_HIDDEN), help="Use tasks_dev.jsonl to tune, tasks_hidden.jsonl to confirm.")
    hf.add_argument("--out", default="results/activation_memory/hf_run.json")
    hf.add_argument("--layer", type=int, default=12)
    hf.add_argument("--coefficient", type=float, default=8.0)
    hf.add_argument("--min-gap", type=float, default=0.5)
    hf.add_argument("--device-map", default="auto")
    hf.add_argument("--allow-download", action="store_true")
    hf.add_argument("--chat-template", choices=["auto", "off"], default="auto")
    hf.add_argument("--include-rows", action="store_true")
    hf.set_defaults(func=run_hf)

    sweep = subparsers.add_parser("hf-sweep", help="Tune layer/coefficient on dev by the pre-registered crossover objective.")
    sweep.add_argument("--model", required=True, help="Local path or Hugging Face model name.")
    sweep.add_argument("--memories", default=str(DEFAULT_MEMORIES))
    sweep.add_argument("--tasks", default=str(DEFAULT_DEV), help="Tune on dev; do not sweep on hidden.")
    sweep.add_argument("--out", default="results/activation_memory/hf_sweep.json")
    sweep.add_argument("--layers", default="all", help="Comma-separated layers, negative indexes, or 'all'.")
    sweep.add_argument("--coefficients", default="0,2,4,8,12")
    sweep.add_argument("--min-gap", type=float, default=0.5)
    sweep.add_argument("--device-map", default="auto")
    sweep.add_argument("--allow-download", action="store_true")
    sweep.add_argument("--chat-template", choices=["auto", "off"], default="auto")
    sweep.add_argument("--top-k", type=int, default=10)
    sweep.set_defaults(func=run_hf_sweep)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = args.func(args)
    if payload["backend"] == "hf-sweep":
        printable = {"best": payload["best"], "elapsed_seconds": payload["elapsed_seconds"]}
    else:
        printable = {"aggregate": payload["aggregate"], "interaction": payload["interaction"], "elapsed_seconds": payload["elapsed_seconds"]}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
