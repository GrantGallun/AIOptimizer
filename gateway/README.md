# Gateway operations

Start the OpenAI-compatible gateway:

```powershell
python -m gateway --port 8800
```

Route default-constructed research clients through it without changing call sites:

```powershell
$env:AIOPT_GATEWAY = "http://127.0.0.1:8800"
```

Read the request receipts:

```powershell
python -m gateway.report results/gateway/ledger.jsonl
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
