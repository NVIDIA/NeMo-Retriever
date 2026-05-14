# retriever benchmark

Throughput micro-benchmarks for individual Ray actors in the ingest
pipeline. Each subcommand isolates one stage and reports rows/sec.

Subcommands:

| Stage | Subcommand | Actor benchmarked |
|---|---|---|
| Split | `retriever benchmark split run` | `PDFSplitActor` |
| Extract | `retriever benchmark extract run` | `PDFExtractionActor` |
| Page elements | `retriever benchmark page-elements run` | `PageElementDetectionActor` |
| OCR | `retriever benchmark ocr run` | `OCRActor` |
| Audio extract | `retriever benchmark audio-extract run` | `MediaChunkActor + ASRActor` |
| All | `retriever benchmark all run` | runs the above in sequence |

If flags below look stale, re-check `retriever benchmark <stage> run --help`.

## When to use this

- You suspect a specific pipeline stage is the bottleneck and want
  rows/sec numbers under controlled load.
- You're sizing Ray actor counts / GPU fractions for [[pipeline]] / [[ingest]]
  and need empirical numbers per stage.
- You want a regression-style benchmark across machines or releases (pair
  with [[harness]] for orchestration).

**Use a different command when:**

- You want end-to-end ingest, not stage-isolated numbers → [[ingest]] or
  [[pipeline]] with a stopwatch.
- You want recall/QA quality, not throughput → [[recall]] / [[eval]].

## Canonical invocations

Benchmark the page-element detector alone:

```bash
retriever benchmark page-elements run --help   # see options
retriever benchmark page-elements run
```

Benchmark OCR (v2 by default; pair with [[pipeline]]'s `--ocr-version`):

```bash
retriever benchmark ocr run
```

Run all stage benchmarks in sequence and print a summary:

```bash
retriever benchmark all run --num-gpus 0.5 --num-cpus 1.0
```

## Inputs

- All `run` commands take their own flag set (run `--help` on the
  individual subcommand). Common shape: rows count, batch size, GPU/CPU
  fractions per actor, optional remote NIM URL.

## Outputs

- Stdout report with per-actor throughput in rows/sec, plus headers per
  stage (e.g. `=== benchmark: page-elements ===`).

## Key flags (`all run`)

| Flag | Default | Notes |
|---|---|---|
| `--num-gpus` | `1.0` | GPUs reserved per page-elements / OCR actor. |
| `--num-cpus` | `1.0` | CPUs reserved per actor. |
| `--rows-page-elements` etc. | per-stage | Synthetic rows per stage benchmark. |

## Reading the results

- Numbers come from a synthetic Ray Dataset; they're representative of the
  stage in isolation, not of end-to-end throughput.
- To convert to [[pipeline]] tuning: pick the slowest stage's rows/sec,
  divide your target rate by it → number of actors needed.

## Common failure modes

- **Page-elements benchmark stalls** — needs YOLOX weights or a remote
  endpoint. Pass the URL flags or pre-cache weights.
- **Benchmark numbers don't match [[pipeline]]** — micro-benchmarks exclude
  inter-stage queues / batching overhead. Treat as upper bounds.
- **`CUDA OOM`** — drop `--num-gpus` (fractional) or `*-batch-size` per
  stage.

## Related

- [[pipeline]] — apply the actor counts derived from these benchmarks.
- [[harness]] — runs benchmarks across configs/datasets and stores results.
