# Gateway operations

Start the OpenAI-compatible gateway:

```powershell
python -m aioptimizer --port 8800
```

Route default-constructed research clients through it without changing call sites:

```powershell
$env:AIOPT_GATEWAY = "http://127.0.0.1:8800"
```

Read the request receipts:

```powershell
python -m aioptimizer.report results/gateway/ledger.jsonl
```

Caching requires an explicitly deterministic request (`temperature: 0`) or an
operator-declared deterministic upstream (`upstream_sampling_default: 0` in the
config, or `--upstream-sampling-default 0`). When temperature is omitted, the
gateway defaults to the upstream's sampled behavior and bypasses caching. This
intentionally reduces the default cache hit rate: replaying a sampled response is a
correctness bug, not a cache feature. The tracked development profile enables
privacy-filtered adaptive attention; a bare CLI invocation remains conservative
unless `--attention-context` or a config enables it.

The proxy accepts OpenAI chat (`/v1/chat/completions`), Anthropic messages
(`/v1/messages`), and Ollama chat/generate endpoints. Provider authentication and
version headers are forwarded upstream. JSON and streamed responses are supported;
streamed requests bypass the exact-response cache and are recorded with
`"streamed": true` in the ledger.

The local `/health` response includes `ready` and a build stamp derived once at
server start from Python-file paths, sizes, and modification times. The lazy sidecar
launcher reuses only a matching build. It restarts a stale process only when the
workspace PID file identifies a live process; otherwise it fails open and records
`sidecar_state: "stale_code"` rather than terminating an unknown listener.

When the provider reports usage, receipts normalize OpenAI (`prompt_tokens` /
`completion_tokens`), Anthropic (`input_tokens` / `output_tokens`), and Ollama
(`prompt_eval_count` / `eval_count`) into input, output, and total token counts.
The report shows token totals, shadow-evaluation overhead, and usage coverage;
providers that omit usage remain visible as uncovered upstream requests. When
both optimized and raw shadow arms report usage, it also reports directly
measured input-token savings. `provider_total_tokens_consumed` includes shadow
overhead so quality measurement is never presented as free.

Anthropic cache creation/read tokens and OpenAI cached/cache-write prompt-token
details are normalized separately. Reports expose cache-observation coverage,
hit rate, cache creation/read/write totals, reasoning and prediction-token
details, and effective input totals; cached-token counts are never treated as
ordinary gateway exact-cache hits.

The final provider-visible request is also fingerprinted for stable-prefix
reuse opportunities. This diagnostic is content-free and never changes the
request; `prompt_prefix_candidate_reuse_rate` is an engineering signal, not a
claim that the upstream provider actually served a cache hit. Provider-reported
cache reads/writes remain the authoritative outcome receipts.

## Outcome-linked episode receipts

Codex and Claude Code hooks now generate a stable opaque `episode_id` plus a unique
`turn_id`. Those identifiers travel to the local context endpoint and, when supplied
as `X-AIOptimizer-Episode-ID` / `X-AIOptimizer-Turn-ID`, through provider gateway
receipts without being forwarded upstream. Hook, gateway, and agent-bus event rows use
`aioptimizer.episode-event.v1`; the schema rejects prompt, response, command,
acceptance, transcript, and producer-result text.

Agent-bus tasks receive a content-free episode id automatically. To join a task to an
external hook episode, issue it with `board.py add --episode-id <opaque-id> ...`.
The live loop appends dispatch, execution cost, producer pass/fail, independent-review,
retry, and terminal-outcome events to `agent_bus/episode_events.jsonl`.

Freeze a replay dataset and a development hard-case view with versioned output names:

```powershell
python -m aioptimizer.episodes inspect --workspace .
python -m aioptimizer.episodes build `
  --workspace . `
  --split-salt frozen-v1 `
  --out results/episodes/episode_dataset_v1.json
python -m aioptimizer.episodes hard-cases `
  --dataset results/episodes/episode_dataset_v1.json `
  --split dev `
  --out results/episodes/hard_cases_dev_v1.json
python -m aioptimizer.episodes replay `
  --dataset results/episodes/episode_dataset_v1.json --split dev --hard-only
```

`inspect` discovers the hook, sidecar gateway, main gateway, and agent-bus ledgers
that exist under the workspace. It reports only aggregate source/event counts and
per-episode coverage for context routing, provider usage, latency, acceptance tests,
verification, retries, requirements, terminal outcomes, and cross-source joins. It
never prints episode IDs or content. `build --workspace` uses the same discovery;
explicit `--events` paths remain available for frozen or external streams.

The normal gateway report can combine both views in one operational snapshot:

```powershell
python -m aioptimizer.report results/gateway/ledger.jsonl `
  --episodes-workspace .
```

Episode-event rows share the append-only gateway ledger but are excluded from
request totals, optimization rates, and latency statistics. The combined report
places those request metrics under `gateway` and join/outcome coverage under
`episodes`, preventing telemetry rows from inflating product traffic counts.

Dataset and hard-case writers use exclusive creation and refuse to overwrite an
existing artifact. Split membership is a frozen SHA-256 threshold assignment. These
artifacts are measurement plumbing, not research verdicts, and no adaptive controller
is enabled by this pipeline. Semantic response caching also remains unwired/off.

## Research evidence audit

Future experiment artifacts can carry a machine-readable evidence card and be
checked without producing a research verdict:

```powershell
python -m aioptimizer.evidence path\to\evidence-card.json
```

The audit flags small samples, too few seeds, absent preregistration/hidden
splits/negative controls, single-model or single-task evidence, author-built-only
data, proxy judges, missing independent review, and unversioned artifacts. A
passing metadata audit means the controls were recorded; it does not mean the
hypothesis was confirmed.

Upstream connect/read operations default to a 300-second timeout. Set
`upstream_timeout_seconds` in `aioptimizer.json` or pass
`--upstream-timeout-seconds`; pre-response timeouts return HTTP 504 and are
recorded as such. A stream that stalls after its response has begun is closed and
receipted with `"stream_complete": false`. Shadow timeouts/errors never fail the
primary request; reports count them separately as measurement failures.

## Deterministic requirement receipts

Normal traffic can opt into exact requirement-retention measurement without an
LLM judge. Add a private envelope to the request; the gateway validates and
removes it before middleware, caching, shadowing, or provider forwarding:

```json
{
  "model": "qwen3:8b",
  "messages": [{"role": "user", "content": "Return the deployment status."}],
  "aioptimizer": {
    "requirements": [{
      "id": "status-format",
      "must_include": ["STATUS:"],
      "must_exclude": ["internal-token"],
      "case_sensitive": true
    }]
  }
}
```

Receipts contain only safe IDs and pass counts, never the include/exclude text.
The same provider request may use different contracts on exact-cache hits.
Streaming text is observed incrementally for OpenAI SSE, Anthropic SSE, and
Ollama NDJSON without changing relayed bytes. This measures explicit contracts;
it intentionally does not guess unstated requirements.
For sampled optimized requests, the report compares primary and raw-shadow
contract passes through `measured_requirement_pass_delta`.
