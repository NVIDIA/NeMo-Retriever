# retriever service

Long-running ingest service: an HTTP/SSE server that accepts document
uploads and runs the pipeline behind the scenes. Two subcommands:

- `retriever service start` — boot the server.
- `retriever service ingest` — client that uploads files to a running
  server.

If flags below look stale, re-check `retriever service <subcmd> --help`.

## When to use this

- You want a single warm process serving many ingest requests (avoids the
  one-shot CLI startup cost — vLLM load, CUDA-graph capture).
- You want to ingest from a remote machine / orchestrator without copying
  files onto a GPU host every time.
- You want to point [[pipeline]] at a remote pipeline via
  `--run-mode service`.

**Use a different command when:**

- One-shot ingest → [[ingest]] / [[pipeline]].
- Local debugging / no service → [[local]].

## Canonical invocations

Start with a YAML config:

```bash
retriever service start --config deploy/retriever-service.yaml
```

Start with inline flags (overrides any YAML):

```bash
retriever service start \
  --host 0.0.0.0 --port 7670 \
  --gpu-devices 0,1 \
  --nim-api-key "$NVIDIA_API_KEY" \
  --api-token "$NEMO_RETRIEVER_API_TOKEN"
```

Upload files to a running server (SSE streaming progress):

```bash
retriever service ingest --server-url http://localhost:7670 data/pdfs/*.pdf
```

Polling instead of SSE (firewalled environments):

```bash
retriever service ingest --no-sse --poll-interval 5.0 data/pdfs/foo.pdf
```

## Inputs / outputs

- **`start`** — no inputs; serves until killed.
- **`ingest`** — one or more file paths, streamed/polled to completion.
  Prints per-file status.

## Key flags

`service start`:

| Flag | Notes |
|---|---|
| `--config -c` | Path to `retriever-service.yaml`. |
| `--host` / `--port -p` | Bind address. Default per YAML. |
| `--log-level` / `--log-file` | Logging overrides. |
| `--nim-api-key` | NIM bearer (also `$NVIDIA_API_KEY`). |
| `--gpu-devices` | CSV GPU IDs. |
| `--api-token` | Bearer required on every request (also `$NEMO_RETRIEVER_API_TOKEN`). Unset = no auth. |

`service ingest`:

| Flag | Default | Notes |
|---|---|---|
| `--server-url -s` | `http://localhost:7670` | Server base URL. |
| `--sse / --no-sse` | `sse` | Stream progress or poll. |
| `--poll-interval` | `2.0` s | Polling cadence when `--no-sse`. |
| `--concurrency` | `8` | Max concurrent uploads. |
| `--api-token` | from `$NEMO_RETRIEVER_API_TOKEN` | Auto-falls back to the env var; pass the flag only to override. |

## Common failure modes

- **`401 Unauthorized`** — server has `--api-token` set; the client must
  match (`--api-token` or `$NEMO_RETRIEVER_API_TOKEN`).
- **Hangs on first request after boot** — model warmup. First request can
  take 30–60s; subsequent ones are sub-second.
- **`Connection refused`** — server binds `0.0.0.0` but firewall blocks the
  port. Tunnel or open the port.
- **CUDA OOM under concurrency** — drop client `--concurrency`, or reduce
  per-stage actor counts in the server YAML.

## Related

- [[pipeline]] with `--run-mode service` — pipeline CLI that delegates to a
  running service.
- [[ingest]] — local one-shot equivalent.
