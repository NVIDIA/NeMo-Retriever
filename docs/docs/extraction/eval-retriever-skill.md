# Evaluate the Retriever skill with coding agents

Use this page when you integrate NeMo Retriever Library into a coding-agent workflow and you need a durable place to prove the integration works, then to investigate traces when scores lag a Claude or Codex baseline.

`retriever skill-eval` is a development and experimental command. It is not a replacement for measuring extraction quality on your own corpus. For dataset-dependent throughput and operational tuning, refer to [Evaluate on your data](evaluate-on-your-data.md).

This page is also not the in-library agentic retrieval loop (`retriever query --agentic`). That loop searches an existing LanceDB table. Skill evaluation instead asks an external agent CLI to ingest and query through the `/nemo-retriever` skill.

Use this eval before you open or expand partner harness pull requests. Keep the session directory as the evidence that the skill path works, or as the traces that explain why it does not.

## On this page { #on-this-page }

- [Ecosystem notebooks versus this eval](#ecosystem-notebooks-versus-this-eval)
- [What the harness compares](#what-the-harness-compares)
- [Proof of life on a small query set](#proof-of-life-on-a-small-query-set)
- [Investigate traces when a baseline looks better](#investigate-traces-when-a-baseline-looks-better)
- [Worked example: distribution of turns](#worked-example-distribution-of-turns)
- [Related commands](#related-commands)

## Ecosystem notebooks versus this eval { #ecosystem-notebooks-versus-this-eval }

NRL shows up in partner stacks in more than one way. Do not mix those surfaces.

| Surface | What it is | Where to look |
|---|---|---|
| Framework examples | Ingest with NeMo Retriever Library, then run a LangChain or LlamaIndex RAG pipeline over the LanceDB table. | [Multimodal RAG with LangChain](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/langchain_multimodal_rag.ipynb) and [Multimodal RAG with LlamaIndex](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/llama_index_multimodal_rag.ipynb) in the repository [starter kits](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/README.md) |
| Agent harnesses | An external coding agent or harness calls ingest and query through the `/nemo-retriever` skill or the `retriever` CLI. | This page |

The LangChain and LlamaIndex notebooks are examples. They are not a first-class partner-connector layer. An NVIDIA RAG retriever published under LangChain for the Foundation RAG blueprint is a different product surface. Do not treat that connector as NRL skill-eval coverage.

LangChain and LlamaIndex APIs have many integration layers beyond those notebooks. This repository does not yet ship a broader connector matrix. Until a partner adapter lands, prove the CLI and skill path with `retriever skill-eval`, then investigate compact traces when a Claude or Codex baseline looks better.

`--agent` still accepts only `claude` and `codex`. Hermes and other harness traces are not archived into compact traces on this command. Run those evals in their own harness, then apply the same questions (did ingest run, did query run, what is the turn distribution) to whatever session logs that harness writes. Do not treat unpublished Harbor runs as a published baseline.

## What the harness compares { #what-the-harness-compares }

`retriever skill-eval run` loads an agent-eval manifest (a JSON list) and, for each domain, runs a setup turn plus one query turn per exported entry. It scores recall@1, recall@5, and recall@10 against the manifest `relevant_pages` field. When a judge is configured, it also scores each query turn `final_answer` against the manifest `answer` on a 0-5 scale.

The packaged conditions are the following:

| Condition | What the agent can use |
|---|---|
| `c1_base` | Retriever blocked. The agent must answer from `./pdfs/` with native tools. |
| `c2_retriever` | Retriever available. Natural-language prompt. Skill auto-discovery. |
| `c3_retriever_skill` | Retriever available. Explicit `/nemo-retriever` slash prompt. |

`--agent` accepts only `claude` or `codex`. Default models are `claude-opus-4-7` and `gpt-5.5`. Other agent CLIs are not parsed into compact traces. Refer to [Ecosystem notebooks versus this eval](#ecosystem-notebooks-versus-this-eval).

Each (agent, condition, domain) session writes artifacts under a `skilleval_<timestamp>` directory. The default root is `nemo_retriever/artifacts/`. Override it with `--artifacts-root`. Keep that directory as the source of truth. Do not reconstruct metrics from chat posts.

## Proof of life on a small query set { #proof-of-life-on-a-small-query-set }

A full domain times conditions matrix is expensive. For a first integration check, run one domain, one skill-aware condition, and a handful of exported manifest entries. The CLI has no `--limit` flag. You shrink the run by slicing the manifest and passing `--domains` and `--conditions`.

### Prerequisites { #prerequisites }

Complete the following before you run:

1. Install the `retriever` CLI. `skill-eval` is a hidden experimental subcommand. Invoke it by name, for example `retriever skill-eval --help`.
2. Put the `claude` or `codex` CLI on `PATH`.
3. Confirm `.claude/skills/nemo-retriever/SKILL.md` exists in the repository, or set `skill_source_dir` in the config. Conditions `c2_retriever` and `c3_retriever_skill` fail if that file is missing.
4. Copy the packaged config next to your dataset checkout and fill the two required bindings:

```bash
cp nemo_retriever/src/nemo_retriever/tools/skill_eval/configs/skill_eval.yaml \
  ~/path/to/datasets/skill_eval.yaml
```

Set `eval_manifest_path` to the agent-eval manifest JSON. Set `pdf_dirs` so each key matches a manifest `domain` and each value is that domain's PDF directory. The runner symlinks `*.pdf` from that directory into the trial `./pdfs/` folder.

Each exported manifest entry must include a paraphrased prompt under `sdg_prompt_candidates` or `scenario_prompt_candidates`, plus `relevant_pages` and `answer` for scoring. Entries whose `prompt_export_status` is not `exported` are skipped.

5. Optional: set `NVIDIA_API_KEY` so the large language model (LLM) judge can run. Without the key, the run still completes and still reports recall. Judge scores are omitted with a console note. To use the local judge helper instead of the hosted endpoint, start it from the repository root and point `judge.api_base` at `http://localhost:8000/v1`:

```bash
docker compose -f nemo_retriever/dev/compose/judge.compose.yaml up -d judge
```

### Slice a smoke manifest { #slice-a-smoke-manifest }

Copy three to five exported objects from the full JSON list into a new file, for example `skill_eval_smoke.json`. Keep at least one `domain` whose PDFs you have locally. Point `eval_manifest_path` (or `--eval-manifest`) at that file.

### Run the skill-aware condition { #run-the-skill-aware-condition }

The following command is a proof-of-life shape: one agent, one domain, and the explicit skill prompt. Replace the domain name with a key from your `pdf_dirs` map.

```bash
retriever skill-eval run \
  --config ~/path/to/datasets/skill_eval.yaml \
  --eval-manifest ~/path/to/datasets/skill_eval_smoke.json \
  --agent claude \
  --domains your_domain \
  --conditions c3_retriever_skill
```

Repeat with `--agent codex` when you need a second baseline. After the skill path works, add `c1_base` and `c2_retriever` to compare blocked-retriever versus auto-discovery versus slash-command behavior:

```bash
retriever skill-eval run \
  --config ~/path/to/datasets/skill_eval.yaml \
  --eval-manifest ~/path/to/datasets/skill_eval_smoke.json \
  --agent claude \
  --domains your_domain \
  --conditions c1_base,c2_retriever,c3_retriever_skill
```

Default `query_parallelism` is 1. That runs setup plus every query in one agent session. Values greater than 1 copy the completed setup workdir and run query turns in isolated parallel sessions.

### Confirm the integration worked { #confirm-the-integration-worked }

The CLI prints the session directory. Open `session_summary.md` in that directory. Proof of life is the following:

- Setup status is `ok`.
- Query-turn `status` values are `ok`, not `timeout`.
- The overall table has recall@1, recall@5, and recall@10.
- For `c3_retriever_skill`, `retr_attempted` and `retr_succeeded` are greater than zero.
- **Per-query traces** links resolve to `trials/<agent>/<condition>/<domain>/traces/*.md`.

The console also prints per-turn `status`, token counts, and a `trace=` path when a compact trace was written. If judge scores are missing, run `retriever skill-eval rescore <session-dir>` after the judge endpoint is reachable. Use `--force` only when you want to re-judge every query turn.

## Investigate traces when a baseline looks better { #investigate-traces-when-a-baseline-looks-better }

Recall and judge scores tell you that a run underperformed. They do not tell you why. After a smoke run works, compare the skill conditions to the same queries on Claude or Codex without treating chat commentary as evidence.

Work from the session directory in this order.

### 1. Read the session summary { #read-the-session-summary }

`session_summary.md` is the index. Use the overall table to identify which `(agent, condition)` pair lost recall or judge score. Use **Session totals** `query_turns` only as the count of scored eval queries. That number is not the agent-internal turn count.

**Diagnostics** reports `skill_fired_rate` when the harness could detect the skill. A high recall with `retr_attempted` near zero on `c1_base` is expected. The same pattern on `c3_retriever_skill` means the slash prompt did not drive the CLI.

**Tool-use summaries** are optional Claude narrations of the compact trace. They require the `claude` CLI. Disable `summarizer.enabled` in the config if you do not want that extra call.

### 2. Open the compact trace, then the raw log { #open-the-compact-trace }

The **Per-query traces** table links `compact_trace` and `raw_log`. Start with the compact trace. It is a turn-organized excerpt of the archived agent JSONL.

Linear sessions (`query_parallelism` 1) share one trace file for setup plus every query. Isolated sessions write one trace file per trial. Compact traces use this layout:

```text
[Turn 1 - setup]
  user: <setup prompt>
  tool_use Bash: retriever ingest ...
  assistant: <truncated assistant text>

[Turn 2 - query 1]
  user: <eval prompt>
  tool_use Bash: retriever query ...
  assistant: <truncated assistant text>
```

Isolated query sessions label the first heading with `query entry_id=<id> query_id=<id>` instead of `query 1`.

Ask the following while you read the trace:

- Did the agent call `retriever ingest` during setup, or did it only read PDFs?
- Did the query turn call `retriever query`, or did it guess from filenames?
- Did a `retriever` command fail (non-zero exit) even though the agent still wrote an answer?
- Did the agent ignore retrieved chunks and answer from prior context?

If the compact trace is missing or too truncated, open the sibling raw log under `trials/<agent>/<condition>/<domain>/logs/`. Claude logs are copied from `~/.claude/projects/<slug>/<session>.jsonl`. Codex logs are copied from `~/.codex/sessions/`. The harness deletes trial workdirs after each condition. Without those archived logs you cannot reconstruct tool-use signals.

### 3. Read the per-trial JSON { #read-the-per-trial-json }

Each trial is `trials/<agent>/<condition>/<domain>/<trial_id>.json`. Query-turn files include `final_answer`, `ranked_retrieved`, `retriever_attempted`, `retriever_succeeded`, `retriever_first_use_turn`, `skill_fired`, `num_turns`, `judge_score`, and `judge_reasoning`.

Compare `ranked_retrieved` to the manifest `relevant_pages` for that `query_id` when recall is low but the trace shows a successful `retriever query`. That pattern is an indexing or ranking issue, not a missing skill call. Compare `judge_reasoning` when recall is fine but the 0-5 answer score is not.

## Worked example: distribution of turns { #worked-example-distribution-of-turns }

The following is one question the traces can answer and chat posts cannot: what is the distribution of agent turns, and is the mean a useful summary?

**Do not** take an average from a Slack thread and stop there. That number does not tell you whether the mean, the median, or a multi-peaked shape describes the run.

**Do** count turns from the session you just ran.

1. Open `session_summary.md`. The **Session totals** `query_turns` column is how many eval queries were scored. It is not how many user and assistant turns the agent used inside a query.
2. Open a compact trace. Count headings that start with `[Turn `. In a linear session, Turn 1 is setup and the remaining headings are query turns in that same session. In isolated mode, each trace file is one query (plus its own first-turn label).
3. For a per-query distribution, read `num_turns` from every query-turn trial JSON. Skip objects where `is_setup` is true. From the session directory:

```python
import json
from pathlib import Path
from statistics import mean, median

turns = []
for path in Path("trials").rglob("*.json"):
    trial = json.loads(path.read_text(encoding="utf-8"))
    if trial.get("is_setup"):
        continue
    turns.append(int(trial["num_turns"]))

print("n", len(turns))
print("mean", mean(turns) if turns else None)
print("median", median(turns) if turns else None)
print("min", min(turns) if turns else None)
print("max", max(turns) if turns else None)
```

4. Compare mean and median. If they diverge, the mean is a poor central description. Group the same `num_turns` values into a short histogram. Two or three peaks usually mean mixed strategies (for example, one-shot answers versus multi-tool retries), not a single typical turn count.
5. Split the histogram by `condition` and `agent` in the trial JSON. A Codex skill run that uses more turns than Claude with the same recall is a cost and latency finding, not automatically a quality win.

Keep the session directory, `session_summary.md`, the compact traces, and this counting method together. That bundle is the place to look the next time someone asks the same question.

## Related commands { #related-commands }

The following commands overlap in name but not in purpose:

- `retriever skill-eval run` and `retriever skill-eval rescore` are this page.
- `retriever eval` and `retriever benchmark` are separate development commands for corpus and stage measurement. Refer to [Evaluate on your data](evaluate-on-your-data.md).
- The in-repo `agent_eval/` harness is a second development track for Claude versus Codex recall and functional pass/fail. It is not the product CLI. That harness accepts `--limit` when you want a smoke subset without slicing a skill-eval manifest.
- LangChain and LlamaIndex example notebooks live under `examples/` in the repository. They demonstrate framework RAG after ingest. They do not replace this skill-eval workflow.
