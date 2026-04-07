#!/usr/bin/env python3
"""
Standalone QA evaluation runner for FileRetriever mode.

Two modes of operation:

  1. **Config-driven** (recommended for sweeps)::

       python run_qa_eval.py --config eval_sweep.yaml

     The YAML file defines models, evaluations (generator+judge combos),
     and optional per-combo run counts.  See ``eval_sweep.yaml`` for the
     full schema.

  2. **Env-var mode** (single model, Docker-friendly)::

       export RETRIEVAL_FILE=data/test_retrieval/bo767_retrieval_fullpage.json
       export NVIDIA_API_KEY=nvapi-...
       python run_qa_eval.py

Required env vars (env-var mode only):
  RETRIEVAL_FILE   Path to the retrieval results JSON

API key env vars:
  GEN_API_KEY      API key for the answer-generation model.
  JUDGE_API_KEY    API key for the judge model.
  NVIDIA_API_KEY   Fallback used by both generator and judge when
                   their individual keys are not set.

Optional env vars (env-var mode):
  GROUND_TRUTH_DIR Directory used by some dataset loaders (ViDoRe, etc.).
  QA_DATASET       Dataset key or csv: path (default: csv:<repo>/data/bo767_annotations.csv).
  QA_TOP_K         Chunks per query (default: 5)
  QA_MAX_WORKERS   Concurrent API calls (default: 4)
  QA_LIMIT         Only evaluate the first N queries (0 = all, default: 0)
  OUTPUT_FILE      Where to write results JSON
                   (default: auto-timestamped under data/test_retrieval/)
  GEN_MODEL        litellm model string for answer generation
  GEN_MODEL_NAME   Short label for the generator (default: generator)
  GEN_API_BASE     Override endpoint for generator model
  GEN_MODELS       Multi-model sweep: comma-separated name:model pairs.
  GEN_TEMPERATURE  Sampling temperature for generator (default: 0.0)
  JUDGE_MODEL      litellm model string for judge
  JUDGE_API_BASE   Override endpoint for judge model
  LITELLM_DEBUG    Set to 1 to enable full litellm request/response logging
  MIN_COVERAGE     Abort if retrieval file covers fewer queries than this
                   fraction (0.0-1.0, default: 0.0 = no check).
                   Set to e.g. 0.8 to fail-fast on misaligned data.
"""

import json
import os
import sys
from datetime import datetime

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
    print("\nTier 1 - Retrieval Quality:")
    print(f"  Answer-in-Context rate:  {aic_rate:.1%} ({aic_count}/{aic_total})")

    tier2 = eval_results.get("tier2_programmatic", {})
    if tier2:
        print("\nTier 2 - Programmatic Answer Quality:")
        for name, metrics in tier2.items():
            f1 = metrics.get("mean_token_f1", 0)
            print(f"  {name:20s} token_f1={f1:.3f}")

    by_model = eval_results.get("by_model", {})
    if by_model:
        print("\nTier 3 - LLM Judge:")
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
        print("\nFailure Breakdown:")
        for name, counts in fb.items():
            total = sum(counts.values())
            parts = "  ".join(f"{k}:{v}({v*100/total:.1f}%)" for k, v in sorted(counts.items()))
            print(f"  {name:20s} {parts}")

    print("=" * 60)


def _run_config_sweep(config_path: str) -> int:
    """Run evaluations from a YAML/JSON config file."""
    if os.environ.get("LITELLM_DEBUG", "0").strip() in ("1", "true", "yes"):
        import litellm

        litellm._turn_on_debug()

    from nemo_retriever.evaluation.config import load_eval_config
    from nemo_retriever.evaluation.retrievers import FileRetriever
    from nemo_retriever.evaluation.generators import LiteLLMClient
    from nemo_retriever.evaluation.judges import LLMJudge
    from nemo_retriever.evaluation.orchestrator import QAEvalPipeline
    from nemo_retriever.evaluation.ground_truth import get_qa_dataset_loader

    config = load_eval_config(config_path)

    execution = config.get("execution", {})
    dataset_cfg = config.get("dataset", {})
    retrieval_cfg = config.get("retrieval", {})
    output_cfg = config.get("output", {})
    models = config["models"]
    evaluations = config["evaluations"]
    default_runs = execution.get("runs", 1)
    top_k = execution.get("top_k", 5)
    max_workers = execution.get("max_workers", 8)
    default_timeout = execution.get("timeout", 120.0)

    qa_dataset = dataset_cfg.get("source", "")
    if not qa_dataset:
        _repo_root = os.path.normpath(os.path.join(_HERE, "..", ".."))
        qa_dataset = "csv:" + os.path.join(_repo_root, "data", "bo767_annotations.csv")

    retrieval_file = retrieval_cfg.get("file_path", "")
    if not retrieval_file:
        print("ERROR: retrieval.file_path is required in config", file=sys.stderr)
        return 1

    results_dir = output_cfg.get("results_dir", os.path.join(_HERE, "data", "test_retrieval"))
    qa_limit = execution.get("limit", 0)
    dataset_stem = os.path.splitext(
        os.path.basename(qa_dataset.split(":", 1)[-1] if ":" in qa_dataset else qa_dataset)
    )[0]

    total_runs = sum(e.get("runs", default_runs) for e in evaluations)
    print("=" * 60)
    print("QA Evaluation (config-driven sweep)")
    print("=" * 60)
    print(f"Config:         {config_path}")
    print(f"Dataset:        {qa_dataset}")
    print(f"Retrieval file: {retrieval_file}")
    print(f"Top-K:          {top_k}")
    print(f"Max workers:    {max_workers}")
    print(f"Total runs:     {total_runs} across {len(evaluations)} evaluation(s)")
    print("=" * 60)

    loader = get_qa_dataset_loader(qa_dataset)
    ground_truth_dir = dataset_cfg.get("ground_truth_dir", os.path.join(_HERE, "data"))
    qa_pairs = loader(data_dir=ground_truth_dir)
    print(f"\nLoaded {len(qa_pairs)} Q&A pairs from '{qa_dataset}'")

    if qa_limit and qa_limit > 0:
        qa_pairs = qa_pairs[:qa_limit]
        print(f"limit={qa_limit}: evaluating first {len(qa_pairs)} pairs")

    retriever = FileRetriever(file_path=retrieval_file)
    coverage = retriever.check_coverage(qa_pairs)
    min_coverage = execution.get("min_coverage", 0.0)
    print(f"Coverage:       {coverage:.1%}")
    if coverage < min_coverage:
        print(
            f"ERROR: retrieval file covers only {coverage:.1%} of queries "
            f"(min_coverage={min_coverage:.0%}). Aborting to avoid "
            f"wasting API credits on misaligned data.",
            file=sys.stderr,
        )
        return 1
    if coverage < 0.5:
        print(
            f"WARNING: retrieval file covers only {coverage:.0%} of queries",
            file=sys.stderr,
        )

    results_log: list[tuple[str, str, str]] = []
    run_number = 0

    for eval_cfg in evaluations:
        gen_name = eval_cfg["generator"]
        judge_name = eval_cfg["judge"]
        n_runs = eval_cfg.get("runs", default_runs)

        gen_model_cfg = models[gen_name]
        judge_model_cfg = models[judge_name]

        client = LiteLLMClient(
            model=gen_model_cfg["model"],
            api_base=gen_model_cfg.get("api_base"),
            api_key=gen_model_cfg.get("api_key"),
            temperature=eval_cfg.get("temperature", gen_model_cfg.get("temperature", 0.0)),
            max_tokens=eval_cfg.get("max_tokens", gen_model_cfg.get("max_tokens", 4096)),
            extra_params=gen_model_cfg.get("extra_params"),
            num_retries=gen_model_cfg.get("num_retries", 3),
            timeout=gen_model_cfg.get("timeout", default_timeout),
        )
        judge = LLMJudge(
            model=judge_model_cfg["model"],
            api_base=judge_model_cfg.get("api_base"),
            api_key=judge_model_cfg.get("api_key"),
            extra_params=judge_model_cfg.get("extra_params"),
            timeout=judge_model_cfg.get("timeout", default_timeout),
        )

        for run_idx in range(1, n_runs + 1):
            run_number += 1
            label = f"{gen_name} + {judge_name} (run {run_idx}/{n_runs})"
            print(f"\n{'=' * 60}\n" f"[{run_number}/{total_runs}] {label}\n" f"{'=' * 60}")

            try:
                pipeline = QAEvalPipeline(
                    retriever=retriever,
                    llm_clients={gen_name: client},
                    judge=judge,
                    top_k=top_k,
                    max_workers=max_workers,
                )
                eval_results = pipeline.evaluate(qa_pairs)

                _print_multi_tier_summary(eval_results, len(qa_pairs))
                _print_errors(eval_results, qa_pairs)

                gen_label = gen_name.replace("/", "_").replace(" ", "_")
                judge_label = judge_name.replace("/", "_").replace(" ", "_")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if n_runs > 1:
                    fname = (
                        f"qa_results_{dataset_stem}_{gen_label}_{judge_label}" f"_{run_idx}of{n_runs}_{timestamp}.json"
                    )
                else:
                    fname = f"qa_results_{dataset_stem}_{gen_label}_{judge_label}" f"_{timestamp}.json"
                out_path = os.path.join(results_dir, fname)
                os.makedirs(results_dir, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(
                        {
                            "dataset": qa_dataset,
                            "retrieval_file": retrieval_file,
                            "top_k": top_k,
                            "generator": gen_name,
                            "judge": judge_name,
                            "run": run_idx,
                            "total_runs": n_runs,
                            "qa_results": eval_results,
                        },
                        f,
                        indent=2,
                    )
                print(f"\nResults written to {out_path}")
                results_log.append((label, "PASS", out_path))

            except Exception as exc:
                print(f"\nERROR: {exc}", file=sys.stderr)
                results_log.append((label, "FAIL", str(exc)))

    passed = sum(1 for _, s, _ in results_log if s == "PASS")
    print(f"\n{'=' * 60}")
    print(f"Sweep complete: {passed}/{total_runs} passed")
    print(f"Coverage:       {coverage:.1%}")
    print("=" * 60)
    for label, status, detail in results_log:
        print(f"  {status}: {label} -> {detail}")

    return 0 if passed == total_runs else 1


def _main_env() -> int:
    """Original env-var-driven single-run entry point."""
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
    litellm_debug = os.environ.get("LITELLM_DEBUG", "0").strip() in ("1", "true", "yes")

    judge_model = os.environ.get("JUDGE_MODEL", "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1")
    judge_api_base = os.environ.get("JUDGE_API_BASE")

    gen_models_str = os.environ.get("GEN_MODELS", "")
    gen_name = os.environ.get("GEN_MODEL_NAME", "generator")
    gen_model = os.environ.get("GEN_MODEL", "nvidia_nim/nvidia/llama-3.3-nemotron-super-49b-v1.5")
    gen_api_base = os.environ.get("GEN_API_BASE")
    gen_temperature = float(os.environ.get("GEN_TEMPERATURE", "0.0"))

    def _default_output_path() -> str:
        dataset_stem = os.path.splitext(
            os.path.basename(qa_dataset.split(":", 1)[-1] if ":" in qa_dataset else qa_dataset)
        )[0]
        model_label = gen_name.replace("/", "_").replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_results_{dataset_stem}_{model_label}_{timestamp}.json"
        return os.path.join(_HERE, "data", "test_retrieval", filename)

    output_file = os.environ.get("OUTPUT_FILE", "") or _default_output_path()

    fallback_key = os.environ.get("NVIDIA_API_KEY", "")
    gen_api_key = os.environ.get("GEN_API_KEY", "") or fallback_key
    judge_api_key = os.environ.get("JUDGE_API_KEY", "") or fallback_key
    if not gen_api_key:
        print("WARNING: No API key found for generator (set GEN_API_KEY or NVIDIA_API_KEY).", file=sys.stderr)
    if not judge_api_key:
        print("WARNING: No API key found for judge (set JUDGE_API_KEY or NVIDIA_API_KEY).", file=sys.stderr)

    if litellm_debug:
        import litellm

        litellm._turn_on_debug()

    from nemo_retriever.evaluation.retrievers import FileRetriever
    from nemo_retriever.evaluation.generators import LiteLLMClient
    from nemo_retriever.evaluation.judges import LLMJudge
    from nemo_retriever.evaluation.orchestrator import QAEvalPipeline
    from nemo_retriever.evaluation.ground_truth import get_qa_dataset_loader

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
    print(f"Gen API key:    {'set' if gen_api_key else 'NOT SET'}")
    print(f"Judge API key:  {'set' if judge_api_key else 'NOT SET'}")
    print("=" * 60)

    loader = get_qa_dataset_loader(qa_dataset)
    qa_pairs = loader(data_dir=ground_truth_dir)
    print(f"\nLoaded {len(qa_pairs)} Q&A pairs from '{qa_dataset}'")

    if qa_limit and qa_limit > 0:
        qa_pairs = qa_pairs[:qa_limit]
        print(f"QA_LIMIT={qa_limit}: evaluating first {len(qa_pairs)} pairs")

    retriever = FileRetriever(file_path=retrieval_file)
    coverage = retriever.check_coverage(qa_pairs)
    min_coverage = float(os.environ.get("MIN_COVERAGE", "0.0"))
    print(f"Coverage:       {coverage:.1%}")
    if coverage < min_coverage:
        print(
            f"ERROR: retrieval file covers only {coverage:.1%} of queries "
            f"(MIN_COVERAGE={min_coverage:.0%}). Aborting to avoid "
            f"wasting API credits on misaligned data.",
            file=sys.stderr,
        )
        return 1
    if coverage < 0.5:
        print(
            f"WARNING: retrieval file covers only {coverage:.0%} of queries -- "
            "results will be unreliable. Check that the retrieval JSON was "
            "generated from the same query set.",
            file=sys.stderr,
        )

    llm_clients = {}
    for gn, gm in gen_model_pairs:
        llm_clients[gn] = LiteLLMClient(
            model=gm,
            api_base=gen_api_base or None,
            api_key=gen_api_key or None,
            temperature=gen_temperature,
        )

    judge = LLMJudge(
        model=judge_model,
        api_base=judge_api_base or None,
        api_key=judge_api_key or None,
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


def main() -> int:
    """Dispatch to config-driven sweep or env-var mode."""
    args = sys.argv[1:]
    if "--config" in args:
        idx = args.index("--config")
        if idx + 1 >= len(args):
            print("ERROR: --config requires a path argument", file=sys.stderr)
            return 1
        return _run_config_sweep(args[idx + 1])
    return _main_env()


if __name__ == "__main__":
    sys.exit(main())
