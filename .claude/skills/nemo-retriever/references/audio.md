# retriever audio

Audio / video extraction stage: chunk media files, run ASR (Parakeet locally
or a remote Riva/NIM endpoint), and write extraction JSON sidecars in the
same primitives shape as [[pdf]].

If flags below look stale, re-check `retriever audio extract --help`.

## When to use this

- You have audio (`.mp3`, `.wav`) or video files and want ASR transcripts
  fed into the rest of the retriever pipeline.
- You want to verify mount/path layout before kicking off a long ASR run →
  use `retriever audio discover` (no ASR, just lists what would be
  processed).

**Use a different command when:**

- You want full ingest including audio → [[pipeline]] with
  `--input-type audio` or [[ingest]] once it accepts audio inputs.
- You want to benchmark ASR throughput → [[benchmark]] (`audio-extract`).

## Canonical invocations

Dry-run discovery:

```bash
retriever audio discover --input-dir data/audio/
```

Local Parakeet ASR over `*.mp3`/`*.wav` (default globs):

```bash
retriever audio extract --input-dir data/audio/
```

Cloud ASR via NIM env vars:

```bash
export NGC_API_KEY=...
export AUDIO_FUNCTION_ID=...
retriever audio extract --input-dir data/audio/ --use-env-asr
```

Override the gRPC endpoint explicitly:

```bash
retriever audio extract \
  --input-dir data/audio/ \
  --audio-grpc-endpoint riva-asr:50051 \
  --auth-token "$NVIDIA_API_KEY"
```

Process video too, extracting audio first:

```bash
retriever audio extract --input-dir data/media/ --glob "*.mp4" --audio-only
```

## Inputs

- **`--input-dir DIR`** — required, scanned (non-recursive) for files
  matching `--glob`.
- **`--glob`** — comma-separated patterns. Default `*.mp3,*.wav`.

## Outputs

- One `<file>.audio_extraction.json` sidecar per source file (default; toggle
  with `--write-json/--no-write-json`).
- Sidecar shape mirrors PDF primitives (`text`, `source_id`, `metadata`),
  with `metadata.content_metadata.type == "text"` per ASR chunk.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--split-type` | `size` | `size` (bytes), `time` (seconds), or `frame`. |
| `--split-interval` | `450` | Chunk size in the chosen units. |
| `--audio-only` | off | Extract audio track from video first, then chunk. |
| `--video-audio-separate` | off | Emit the extracted MP3 as its own item. |
| `--use-env-asr` | on | Build ASR params from `AUDIO_GRPC_ENDPOINT`/`NGC_API_KEY`/`AUDIO_FUNCTION_ID`. |
| `--audio-grpc-endpoint` | — | Override env; sets remote ASR. Wins over `--use-env-asr`. |
| `--auth-token` | — | Bearer for cloud ASR (also `$NVIDIA_API_KEY`). |
| `--limit` | — | Cap files processed. |

## Common failure modes

- **`No files matched glob`** — default globs are `*.mp3,*.wav`. Pass
  `--glob "*.mp4"` for video, etc.
- **Falls back to local Parakeet unexpectedly** — `--use-env-asr` is on but
  none of `AUDIO_GRPC_ENDPOINT` / `NGC_API_KEY` / `AUDIO_FUNCTION_ID` are
  set. Either set them or pass `--audio-grpc-endpoint`.
- **Local Parakeet OOM on long files** — drop `--split-interval` (smaller
  chunks) or switch to a remote NIM.

## Related

- [[pipeline]] with `--input-type audio` — full ingest including embedding +
  VDB.
- [[benchmark]] `audio-extract` — throughput benchmarks.
