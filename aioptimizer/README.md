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

Sampled requests bypass the cache so self-consistency samples remain independent. The attention-context stage is default-OFF pending t0031.

The proxy accepts OpenAI chat (`/v1/chat/completions`), Anthropic messages
(`/v1/messages`), and Ollama chat/generate endpoints. Provider authentication and
version headers are forwarded upstream. JSON and streamed responses are supported;
streamed requests bypass the exact-response cache and are recorded with
`"streamed": true` in the ledger.

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
