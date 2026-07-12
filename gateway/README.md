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
