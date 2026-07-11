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
