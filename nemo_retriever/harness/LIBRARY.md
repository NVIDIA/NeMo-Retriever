<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Library Harness Runs

Use library execution to benchmark the current NeMo Retriever checkout directly.
Choose the mode by workload size:

| Mode | Use it for | Execution |
| --- | --- | --- |
| `local` | Small smoke tests such as JP20 | In-process ingest |
| `batch` | Larger corpora and multi-GPU runs | Ray-backed batch ingest |

Both modes write the [same artifacts](README.md#results-and-artifacts).

## Prepare the Host

You need the repository dependencies, NVIDIA drivers, the selected datasets,
and enough storage for artifacts. Run commands from the repository root through
the `nemo_retriever` project.

Dataset mounts vary by machine:

```bash
cp nemo_retriever/harness/dataset_paths.example.yaml \
  /local/path/to/dataset_paths.yaml
${EDITOR:-vi} /local/path/to/dataset_paths.yaml
```

Keep this file outside the repository. It maps registered dataset names to local
corpus paths and any annotations that are not checked in. JP20 uses the
repository's `data/jp20_query_gt.csv` unless `query_file` is explicitly
overridden.

## Run One Benchmark

Start with the checked-in JP20 runfile:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --session-name jp20_check \
  --output-dir /tmp/retriever-harness-jp20-check \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  --dry-run \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

Confirm that `session_summary.json` succeeds and inspect the child
`resolved_benchmark.json`. Then run the same command without `--dry-run`.

If the registry paths already exist on the host, `run` is a shorter single-run
form:

```bash
uv run --project nemo_retriever retriever harness run jp20_beir \
  --output-dir /tmp/retriever-harness-jp20 \
  --require 'files==20' \
  --require 'pages==1940'
```

## Run in Batch

Checked-in runfiles for BO767, FinanceBench, Earnings, and ViDoRe already select
`batch`. Run one of them exactly as you ran JP20:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --session-name bo767_beir \
  --output-dir /local/path/to/retriever-artifacts/bo767-beir \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  nemo_retriever/harness/runfiles/bo767_beir.json
```

Pass several runfiles to create one sequential session:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --session-name library_beir \
  --output-dir /local/path/to/retriever-artifacts/library-beir \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  nemo_retriever/harness/runfiles/jp20_beir.json \
  nemo_retriever/harness/runfiles/bo767_beir.json \
  nemo_retriever/harness/runfiles/earnings_beir.json \
  nemo_retriever/harness/runfiles/financebench_beir.json
```

Children run sequentially in fresh processes. Ray and model resources are
released between datasets, while the parent writes one
`session_summary.json`.

## Run ViDoRe

ViDoRe queries, qrels, and corpus metadata require Hugging Face access. Check it
before GPU work:

```bash
export HF_TOKEN=...
uv run --project nemo_retriever retriever harness check-vidore-access
```

Then run one domain with its checked-in runfile:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --session-name vidore_v3_computer_science \
  --output-dir /local/path/to/retriever-artifacts/vidore-v3-computer-science \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  nemo_retriever/harness/runfiles/vidore_v3_computer_science_beir.json
```

The [nightly launcher](../../ops/retriever-nightly/README.md) runs all eight
ViDoRe domains plus the four library benchmarks.

## Apply a One-Off Override

Use repeated `--set KEY=VALUE` options for an experiment:

```bash
uv run --project nemo_retriever retriever harness run jp20_beir \
  --set query.top_k=20 \
  --set query.rerank=true
```

Supported namespaces are `dataset.*`, `ingest.*`, `query.*`, and
`evaluation.*`. Unknown keys fail before execution. Use `show --json` and
`run --dry-run --json` to inspect the resolved configuration.

For reproducible recurring changes, put the values in a reviewed runfile or
benchmark definition rather than a shell command.

## Evaluate Agentic Retrieval

Agentic BEIR evaluation uses the same retrieval graph as `retriever query
--agentic` after ingest:

```bash
uv run --project nemo_retriever retriever harness run jp20_beir \
  --set query.agentic=true
```

The default agent LLM runs locally. To use an OpenAI-compatible endpoint:

```bash
uv run --project nemo_retriever retriever harness run jp20_beir \
  --set query.agentic=true \
  --set query.agentic_llm_model=custom-remote-model \
  --set query.agentic_invoke_url=http://localhost:9000/v1/chat/completions
```

Use `query.agentic_backend_top_k`, `query.agentic_react_max_steps`,
`query.agentic_num_concurrent`, and the local vLLM resource overrides only when
the experiment requires them. The resolved benchmark records every value.

## Run on a Schedule

The harness itself is one-shot. Use the
[nightly launcher](../../ops/retriever-nightly/README.md) for the checked-in
twelve-benchmark suite, Git selection, Slack reporting, and a transparent daily
`tmux` loop.
