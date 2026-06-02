# agent_eval

An agent-agnostic harness to evaluate how well a coding agent (**claude** or **codex**)
answers questions over a document corpus — **with** the nemo-retriever skill vs. a
**baseline** that has no retriever — and to score the results (recall@k + LLM judge).

Three stages, three scripts:

```
extract_queries.py   manifest.json         ->  queries.json          (answer-free)
run_agent_eval.py    queries.json          ->  <save-root>/<run_id>/ (per-question artifacts)
build_report.py      run dir(s) + manifest ->  report.md/json + comparison.md
```

- **`run_agent_eval.py` is standalone** (stdlib + the local modules only — no
  `nemo_retriever` import). Copy `agent_eval/` to a clean machine to run a baseline
  with no retriever installed.
- **`build_report.py` may import `nemo_retriever`** (reuses `recall_at_k` + `LLMJudge`),
  so run it where the codebase exists.

---

## 1. `extract_queries.py` — manifest → answer-free queries

Pulls the agent-facing prompt out of an eval manifest into a `queries.json` containing
**only** `{query_id, prompt, domain}` per query. No answer, no gold pages, no
`category`/`scoring_mode` (so a `refusal` label can't tip off the agent). Gold is
recovered later by `build_report.py`, which re-reads the same manifest.

Handles both manifest shapes: the corpus-level manifest's top-level `prompt`, and the
scenario manifest's `scenario_prompt_candidates.candidates[].prompt`. An entry is
included iff it has a `primary_eval_id` and `prompt_export_status == "exported"`.

```bash
python3 extract_queries.py \
  --manifest /raid/retriever-sdg-v3/runs/.../agent_corpus_level_manifest.json \
  --out      /raid/retriever-sdg-v3/runs/.../queries.json
```

| arg | default | meaning |
|---|---|---|
| `--manifest` | *(required)* | Path to the eval manifest JSON. |
| `--out` | `queries.json` | Output path for the answer-free queries file. |
| `--categories` | none | Comma-separated category filter, e.g. `query,extract`. Filtering happens here so the label never enters the file. |
| `--domains` | none | Comma-separated domain filter. |
| `--limit` | none | Cap the number of queries (smoke tests). |

Output: `{source_manifest, extracted_at, count, queries:[{query_id, prompt, domain}]}`.

---

## 2. `run_agent_eval.py` — run an agent over the queries

For each query: builds an isolated workdir (corpus symlinked in), runs the agent for a
single turn, and saves a per-question subfolder. The agent must write `./output.json`
with `final_answer` + ranked `selected_chunks` (top-k pages).

**Profiles** (the with/without-retriever axis):
- `baseline` — no skill; `retriever` blocked (PATH-shim + Claude `settings.json` deny +
  empty HF cache). Agent answers from `./pdfs/` with native tools. Emits 0-indexed pages.
- `skill` — the nemo-retriever skill is copied in and the real `retriever` CLI is used.
  The index is built by an **agent setup turn** (`retriever ingest`, measured) unless a
  prebuilt index is supplied. Emits the retriever's raw 1-indexed pages (the report
  normalizes via `page_index_base`).

```bash
# Baseline (no GPU; crank parallelism):
python3 run_agent_eval.py --queries queries.json \
  --agent claude --model claude-opus-4-7 --profile baseline \
  --save-root /tmp/runs --domains vidore_v3_finance_en,vidore_v3_hr,vidore_v3_pharmaceuticals \
  --parallelism 48 --timeout 900

# Skill, fresh indexes, GPU-pinned (one engine per GPU):
python3 run_agent_eval.py --queries queries.json \
  --agent codex --model gpt-5.5 --profile skill \
  --save-root /tmp/runs --domains vidore_v3_finance_en,vidore_v3_hr,vidore_v3_pharmaceuticals \
  --parallelism 8 --gpu-list 0,1,2,3 --timeout 7200 \
  --retriever-bin /raid/nemo_retriever/.venv/bin/retriever

# Skill, reuse indexes from a prior run (skip GPU ingest):
python3 run_agent_eval.py --queries queries.json --agent claude --profile skill \
  --save-root /tmp/runs --prebuilt-index-root /tmp/runs/agenteval_claude_skill_<ts> --gpus 8 --parallelism 8
```

| arg | default | meaning |
|---|---|---|
| `--queries` | *(required)* | The `queries.json` from stage 1. |
| `--agent` | `claude` | `claude` or `codex`. |
| `--model` | `claude-opus-4-7` | Model id (e.g. `gpt-5.5` for codex) — editable. |
| `--profile` | `skill` | `baseline` (no retriever) or `skill`. |
| `--corpus-root` | `/raid/data/vidore_v3` | Domain corpora live at `<corpus-root>/<domain>/*.pdf`. |
| `--domains` | all | Comma-separated domain filter (each must have a corpus dir). |
| `--limit` | none | Cap query count. |
| `--save-root` | `./agent_eval_runs` | Where to write `<run_id>/`. **Keep this OUTSIDE the repo** for baseline (else the skill/`.claude` leaks in — a guard blocks it). |
| `--top-k` | `10` | Chunks the agent should return. |
| `--parallelism` | `4` | Concurrent query turns. Baseline: high (no GPU). Skill: ≈ GPUs (or 2× with pinning). |
| `--timeout` | `2400` | Per-turn wall-clock seconds (raise to ~7200 for full `retriever ingest` setups). |
| `--budget-usd` | `5.0` | Per-trial budget (Claude). |
| `--gpus` | `0` (auto) | GPUs to round-robin query processes across (skill). 0 = auto-detect via `nvidia-smi`. |
| `--gpu-list` | none | Explicit physical GPU ids, e.g. `0,1,2,3`. Overrides `--gpus`; lets two concurrent runs use disjoint halves. Pins setup ingests too. |
| `--skill-src` | `…/skills/nemo-retriever` | Skill dir to copy into skill-profile workdirs. |
| `--retriever-bin` | `…/.venv/bin/retriever` | The `retriever` CLI (skill profile). |
| `--embed-model` | `nvidia/llama-nemotron-embed-1b-v2` | Embed model for subprocess ingest. |
| `--prebuilt-index` | none | Reuse one `lancedb` dir (single-domain). |
| `--prebuilt-index-root` | none | Reuse per-domain indexes from a prior run: `<root>/_setup/skill_<domain>/lancedb`. Skips GPU ingest. |
| `--subprocess-ingest` | off | Build the index via a direct `retriever ingest` subprocess instead of an agent setup turn (no setup tokens). |
| `--skip-setup` | off | Skill: don't ingest (assume index already present). |
| `--allow-unsafe-save-root` | off | Proceed even if the save-root would un-blind a baseline. |
| `--dry-run` | off | Build workdirs + print the agent command; no agent calls. |

**Output** — `<save-root>/<run_id>/`:
- `run_config.json`, `run_metas.json`, `setup_metas.json`
- `_setup/<profile>_<domain>/` — corpus symlinks + built `lancedb/` (+ `setup_meta.json` for agent setups)
- `<query_id>/` — `output.json` (normalized chunks+answer), `agent_output.raw.json`,
  `meta.json` (status/tokens/cost/timing/session), `trace.md`, `agent_log.jsonl`, `prompt.txt`

`run_id` = `agenteval_<agent>_<profile>_<UTC-timestamp>`.

---

## 3. `build_report.py` — score runs into a report

Re-reads the manifest for gold (`relevant_pages`, `answer`), joins by `query_id`, and
writes `report.md`/`report.json` per run + a `comparison.md` across runs. Computes
`success_rate`, `recall@1/5/10` (normalized by each run's `page_index_base`), LLM
**judge**, refusal-correctness, and **retriever usage** (`attempted` /
`succeeded_clean` / `succeeded_engine`) with agent-aware log parsing (claude tool-events
vs codex `response_item` events). Re-runs reuse judge scores from an existing
`report.json` (judge cache), so re-scoring after a code change is fast.

```bash
# Compare two runs (auto-resolves the manifest from each run_config; judge on if NVIDIA_API_KEY set):
python3 build_report.py \
  /tmp/runs/agenteval_claude_baseline_<ts> /tmp/runs/agenteval_codex_baseline_<ts> \
  --manifest /raid/retriever-sdg-v3/runs/.../agent_corpus_level_manifest.json \
  --out /tmp/runs

# Recall + retriever only, no judge (fast):
python3 build_report.py /tmp/runs/agenteval_claude_skill_<ts> --no-judge
```

| arg | default | meaning |
|---|---|---|
| `run_dirs` | *(required, 1+)* | One or more run directories. >1 → also writes `comparison.md`/`json`. |
| `--manifest` | each run's `source_manifest` | Gold manifest. Override if it moved. |
| `--judge` / `--no-judge` | on if `$NVIDIA_API_KEY` set | Enable/disable LLM-as-judge. |
| `--judge-model` | `nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5` | Judge model. |
| `--judge-api-base` | `https://integrate.api.nvidia.com/v1` | Judge endpoint. |
| `--judge-api-key-env` | `NVIDIA_API_KEY` | Env var holding the judge API key. |
| `--out` | parent of first run dir | Where `comparison.md`/`json` are written. |

**Outputs:** `report.md` + `report.json` inside each run dir; `comparison.md` +
`comparison.json` in `--out` when multiple runs are passed.

### Cost & token accounting

Cost and token totals are made apples-to-apples across agents inside `build_report.py`:

- **Claude** emits `total_cost_usd` directly (used as-is) and reports `input_tokens`
  *exclusive* of cached tokens (`cache_read` is separate).
- **Codex** emits **no dollar cost**, so it's derived from tokens via a per-model
  price table (`_PRICING`, $/1M):

  | model | input | cached_input | output |
  |---|---|---|---|
  | `gpt-5.5` | $5.00 | $0.50 | $30.00 |

  `cost = (fresh_input·input + cached·cached_input + output·output) / 1e6`, where
  `fresh_input = input_tokens − cached_input_tokens`. Codex's `output_tokens` already
  includes reasoning tokens, so they aren't added again. **To price a new model, add a
  row to `_PRICING`** (top of `build_report.py`); models not in the table fall back to
  whatever cost the CLI emitted (or `n/a`).

- **Token-total convention:** codex reports `input_tokens` *inclusive* of the cached
  subset (OpenAI style), so the report normalizes `input → input − cache_read`
  (`_norm_tokens`, codex only) **for the displayed totals** — otherwise the sum
  `input + output + cache_read` would double-count cached tokens. Cost is computed from
  the raw tokens *before* this normalization, so it's unaffected. Net: every run's
  `total` = `non-cached-input + output + cache_read` regardless of agent.

This is all report-side — no adapter change, no meta mutation — so it applies to
existing run artifacts on re-report (judge scores reuse the cache, so re-reporting is fast).

---

## End-to-end example

```bash
M=/raid/retriever-sdg-v3/runs/agent_corpus_level_batch_4/agent_corpus_level_manifest.json
DOM=vidore_v3_finance_en,vidore_v3_hr,vidore_v3_pharmaceuticals

# 1. queries
python3 extract_queries.py --manifest $M --out /tmp/queries.json

# 2. runs (baseline both agents in parallel; skill both agents on disjoint GPU halves)
python3 run_agent_eval.py --queries /tmp/queries.json --agent claude --model claude-opus-4-7 \
  --profile baseline --domains $DOM --save-root /tmp/base --parallelism 48 --timeout 900 &
python3 run_agent_eval.py --queries /tmp/queries.json --agent codex --model gpt-5.5 \
  --profile baseline --domains $DOM --save-root /tmp/base --parallelism 48 --timeout 900 &
wait

python3 run_agent_eval.py --queries /tmp/queries.json --agent claude --model claude-opus-4-7 \
  --profile skill --domains $DOM --save-root /tmp/skill --gpu-list 0,1,2,3 --parallelism 8 --timeout 7200 &
python3 run_agent_eval.py --queries /tmp/queries.json --agent codex --model gpt-5.5 \
  --profile skill --domains $DOM --save-root /tmp/skill --gpu-list 4,5,6,7 --parallelism 8 --timeout 7200 &
wait

# 3. reports
python3 build_report.py /tmp/base/agenteval_claude_baseline_* /tmp/base/agenteval_codex_baseline_* --manifest $M --out /tmp/base
python3 build_report.py /tmp/skill/agenteval_claude_skill_*   /tmp/skill/agenteval_codex_skill_*     --manifest $M --out /tmp/skill
```

## Notes / gotchas
- **Baseline save-root must be outside the repo** — otherwise the agent discovers
  `.claude/skills/nemo-retriever` by walking up the tree; a guard hard-blocks this
  (override with `--allow-unsafe-save-root`).
- **Skill GPU contention** — many concurrent query processes default to GPU 0 and OOM;
  use `--gpus`/`--gpu-list` to spread them (≈ 1–2 engines per GPU).
- **codex cost** is derived from tokens × `_PRICING` (the codex CLI emits no
  `total_cost_usd`) — see *Cost & token accounting* above. Update `_PRICING` if rates change.
- **codex `retr_succeeded_clean` is low by design** — codex's ~1s exec-yield backgrounds
  `retriever query`, so it rarely captures a clean exit even though the engine returns
  hits (see `retr_succeeded_engine`).
