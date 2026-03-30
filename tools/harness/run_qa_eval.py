#!/usr/bin/env python3
"""
Standalone QA evaluation runner for FileRetriever mode.

Reads all config from environment variables so it can be used directly
inside the minimal Docker image without the full harness CLI/config stack.

Required env vars:
  RETRIEVAL_FILE   Path to the retrieval results JSON
                   e.g. /data/test_retrieval/bo767_sample.json
  NVIDIA_API_KEY   API key for NVIDIA NIM endpoints

Optional env vars:
  GROUND_TRUTH_DIR Directory used by some dataset loaders (ViDoRe, etc.).
                   For csv: datasets, paths are taken from QA_DATASET.
  QA_DATASET       Dataset key or csv: path (default: csv:<repo>/data/bo767_annotations.csv).
                   Use "csv:/path/to/file.csv" for custom CSVs.
  QA_TOP_K         Chunks per query (default: 5)
  QA_MAX_WORKERS   Concurrent API calls (default: 4)
  QA_LIMIT         Only evaluate the first N queries (0 = all, default: 0)
  OUTPUT_FILE      Where to write results JSON (default: /tmp/qa_results.json)
  JUDGE_MODEL      litellm model string for judge
                   (default: nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1)
  JUDGE_API_BASE   Override endpoint for judge model
  GEN_MODEL_NAME   Short label for the generator (default: generator)
  GEN_MODEL        litellm model string for answer generation
                   (default: nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5)
  GEN_API_BASE     Override endpoint for generator model
  GEN_MODELS       Multi-model sweep: comma-separated name:model pairs.
                   Overrides GEN_MODEL/GEN_MODEL_NAME when set.
                   e.g. "nemotron:nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5,qwen3:nvidia_nim/qwen/qwen3-235b-a22b"
  LITELLM_DEBUG    Set to 1 to enable full litellm request/response logging
"""

import json
import os
import sys

# Allow running from repo root: add the harness src/ to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))


def _require(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        print(f"ERROR: {name} environment variable is required but not set", file=sys.stderr)
        sys.exit(1)
    return value


def _print_errors(eval_results: dict, qa_pairs: list) -> None:
    """Print per-query error details so they are visible without the results JSON."""
    per_query = eval_results.get("per_query", [])
    errors_found = False
    for i, qr in enumerate(per_query):
        query_text = qr.get("query", "")[:60]
        for model_name, gen in qr.get("generations", {}).items():
            if gen.get("error"):
                if not errors_found:
                    print("\n--- Generation errors ---")
                    errors_found = True
                print(f"  [query {i}] {query_text!r}")
                print(f"    model={model_name}  error={gen['error']}")
        for model_name, jdg in qr.get("judgements", {}).items():
            if jdg.get("error"):
                if not errors_found:
                    print("\n--- Judge errors ---")
                    errors_found = True
                print(f"  [query {i}] {query_text!r}")
                print(f"    model={model_name}  error={jdg['error']}")
    if not errors_found and per_query:
        print("\nNo per-query errors.")


def _print_multi_tier_summary(eval_results: dict, total_queries: int) -> None:
    """Print the multi-tier evaluation summary to stdout."""
    print("\n" + "=" * 60)
    print("Multi-Tier Results")
    print("=" * 60)

    tier1 = eval_results.get("tier1_retrieval", {})
    aic_rate = tier1.get("answer_in_context_rate", 0)
    aic_count = tier1.get("answer_in_context_count", 0)
    aic_total = tier1.get("total", total_queries)
    print(f"\nTier 1 - Retrieval Quality:")
    print(f"  Answer-in-Context rate:  {aic_rate:.1%} ({aic_count}/{aic_total})")

    tier2 = eval_results.get("tier2_programmatic", {})
    if tier2:
        print(f"\nTier 2 - Programmatic Answer Quality:")
        for name, metrics in tier2.items():
            em = metrics.get("mean_exact_match", 0)
            f1 = metrics.get("mean_token_f1", 0)
            print(f"  {name:20s} exact_match={em:.1%}  token_f1={f1:.3f}")

    by_model = eval_results.get("by_model", {})
    if by_model:
        print(f"\nTier 3 - LLM Judge:")
        for name, stats in by_model.items():
            ms = stats.get("mean_score", 0)
            ml = stats.get("mean_latency_s", 0)
            sc = stats.get("scored_count", 0)
            ec = stats.get("error_count", 0)
            print(f"  {name:20s} mean={ms:.2f}/5  latency={ml:.1f}s  scored={sc}  errors={ec}")
            dist = stats.get("score_distribution", {})
            if dist:
                print(f"  {'':20s} dist: " + "  ".join(f"{k}:{v}" for k, v in sorted(dist.items())))

    fb = eval_results.get("failure_breakdown", {})
    if fb:
        print(f"\nFailure Breakdown:")
        for name, counts in fb.items():
            parts = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            print(f"  {name:20s} {parts}")

    print("=" * 60)


def main() -> int:
    retrieval_file = _require("RETRIEVAL_FILE")
    ground_truth_dir = os.environ.get("GROUND_TRUTH_DIR", os.path.join(_HERE, "data"))
    _repo_root = os.path.normpath(os.path.join(_HERE, "..", ".."))
    qa_dataset = os.environ.get(
        "QA_DATASET",
        "csv:" + os.path.join(_repo_root, "data", "bo767_annotations.csv"),
    )
    qa_top_k = int(os.environ.get("QA_TOP_K", "5"))
    qa_max_workers = int(os.environ.get("QA_MAX_WORKERS", "4"))
    qa_limit = int(os.environ.get("QA_LIMIT", "0"))
    output_file = os.environ.get("OUTPUT_FILE", "/tmp/qa_results.json")
    litellm_debug = os.environ.get("LITELLM_DEBUG", "0").strip() in ("1", "true", "yes")

    judge_model = os.environ.get(
        "JUDGE_MODEL", "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"
    )
    judge_api_base = os.environ.get("JUDGE_API_BASE")

    gen_models_str = os.environ.get("GEN_MODELS", "")
    gen_name = os.environ.get("GEN_MODEL_NAME", "generator")
    gen_model = os.environ.get(
        "GEN_MODEL", "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5"
    )
    gen_api_base = os.environ.get("GEN_API_BASE")

    # Validate API key looks real before spending time loading everything
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print("WARNING: NVIDIA_API_KEY is not set. API calls will fail.", file=sys.stderr)
    elif api_key.startswith("nvapi-") and len(api_key) < 20:
        print("WARNING: NVIDIA_API_KEY looks like a placeholder. Did you paste the real key?",
              file=sys.stderr)

    if litellm_debug:
        import litellm
        litellm._turn_on_debug()

    from nv_ingest_harness.utils.qa.retrievers import FileRetriever
    from nv_ingest_harness.utils.qa.generators import LiteLLMClient
    from nv_ingest_harness.utils.qa.judges import LLMJudge
    from nv_ingest_harness.utils.qa.orchestrator import QAEvalPipeline
    from nv_ingest_harness.utils.qa.ground_truth import get_qa_dataset_loader

    gen_model_pairs: list[tuple[str, str]] = []
    if gen_models_str:
        for entry in gen_models_str.split(","):
            entry = entry.strip()
            if ":" not in entry:
                print(f"ERROR: GEN_MODELS entry '{entry}' must be name:model", file=sys.stderr)
                sys.exit(1)
            name, model = entry.split(":", 1)
            gen_model_pairs.append((name.strip(), model.strip()))
    else:
        gen_model_pairs.append((gen_name, gen_model))

    print("=" * 60)
    print("QA Evaluation (standalone runner)")
    print("=" * 60)
    print(f"Dataset:        {qa_dataset}")
    print(f"Retrieval file: {retrieval_file}")
    print(f"Ground truth:   {ground_truth_dir}")
    print(f"Top-K:          {qa_top_k}")
    print(f"Max workers:    {qa_max_workers}")
    for gn, gm in gen_model_pairs:
        print(f"Generator:      {gn} ({gm})")
    print(f"  api_base:     {gen_api_base}")
    print(f"Judge:          {judge_model}")
    print(f"  api_base:     {judge_api_base}")
    print(f"NVIDIA_API_KEY: {'set (' + api_key[:12] + '...)' if api_key else 'NOT SET'}")
    print("=" * 60)

    loader = get_qa_dataset_loader(qa_dataset)
    qa_pairs = loader(data_dir=ground_truth_dir)
    print(f"\nLoaded {len(qa_pairs)} Q&A pairs from '{qa_dataset}'")

    if qa_limit and qa_limit > 0:
        qa_pairs = qa_pairs[:qa_limit]
        print(f"QA_LIMIT={qa_limit}: evaluating first {len(qa_pairs)} pairs")

    retriever = FileRetriever(file_path=retrieval_file)
    coverage = retriever.check_coverage(qa_pairs)
    if coverage < 0.5:
        print(f"WARNING: retrieval file covers only {coverage:.0%} of queries -- "
              "results will be unreliable. Check that the retrieval JSON was "
              "generated from the same query set.", file=sys.stderr)

    llm_clients = {}
    for gn, gm in gen_model_pairs:
        llm_clients[gn] = LiteLLMClient(
            model=gm,
            api_base=gen_api_base or None,
            api_key=api_key or None,
        )

    judge = LLMJudge(
        model=judge_model,
        api_base=judge_api_base or None,
        api_key=api_key or None,
    )

    pipeline = QAEvalPipeline(
        retriever=retriever,
        llm_clients=llm_clients,
        judge=judge,
        top_k=qa_top_k,
        max_workers=qa_max_workers,
    )

    print(f"\nRunning evaluation ({len(qa_pairs)} queries, {len(llm_clients)} model(s)) ...")
    eval_results = pipeline.evaluate(qa_pairs)

    _print_multi_tier_summary(eval_results, len(qa_pairs))

    # Always print error details so they are visible without the results JSON
    _print_errors(eval_results, qa_pairs)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(
            {
                "dataset": qa_dataset,
                "retrieval_file": retrieval_file,
                "top_k": qa_top_k,
                "qa_results": eval_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults written to {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
