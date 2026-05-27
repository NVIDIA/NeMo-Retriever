---
name: nemo-retriever-setup
description: Use when the user asks to install, verify, or orient to NeMo Retriever, when `retriever` is missing, when choosing extras or model/API prerequisites, or before another Retriever workflow can run. Do not use for a specific ingest, query, service, or evaluation task once the CLI works; use that task skill instead.
---

# nemo-retriever-setup

Use this skill to get an agent into a working NeMo Retriever environment before
running task-specific workflows.

## Orientation

1. Verify the intended public entry points:

   ```bash
   retriever --help
   ```

2. If this is a source checkout, the developer fallback is:

   ```bash
   uv run --project nemo_retriever retriever --help
   ```

3. If neither the installed CLI nor the source fallback works, report setup as
   the blocker before attempting ingest/query/service/evaluation.

## References

- `PITFALLS.md`: Python version, missing package, optional extras, system
  dependencies, API keys, and model-cache issues.

## Workflow

1. Confirm Python 3.12. NeMo Retriever requires Python `>=3.12,<3.13`.
2. Choose install shape:
   - Remote NIM inference, no local GPU models: install the base package.
   - Local GPU inference: install the `local` extra and verify CUDA/PyTorch.
   - Audio/video or SVG inputs: add the `multimedia` extra and system `ffmpeg`
     / `ffprobe` when needed.
   - QA generation or judging: add the `llm` extra and configure model keys.
3. Create an isolated environment:

   ```bash
   uv python install 3.12
   uv venv retriever --python 3.12
   source retriever/bin/activate
   uv pip install nemo-retriever
   ```

   For local GPU inference, install the appropriate extra instead:

   ```bash
   uv pip install "nemo-retriever[local]"
   ```

4. Route first-time HuggingFace downloads outside the repo when preparing local
   inference:

   ```bash
   export HF_HOME="$HOME/models/huggingface"
   export HF_HUB_CACHE="$HOME/models/huggingface/hub"
   ```

5. For remote hosted NIMs, configure credentials before ingest/query:

   ```bash
   export NVIDIA_API_KEY=nvapi-...
   ```

6. Re-run the public-surface checks. Once `retriever --help` and the relevant
   subcommand help work, switch to the task skill for ingest, query, service, or
   evaluation.

## Success Checks

- `retriever --help` shows `ingest`, `query`, `service`, `recall`, `eval`, and
  `pipeline` commands.
- `python -c "import nemo_retriever"` succeeds in the same environment.
- The chosen task command's `--help` output is visible before running expensive
  model or data work.

## Evaluation Scenarios

- "Install NeMo Retriever and verify the CLI." Use this skill.
- "`retriever` is not found; what should I do?" Use this skill.
- "Index the PDFs in `data/reports`." Use `nemo-retriever-ingest` once the
  environment is working.
