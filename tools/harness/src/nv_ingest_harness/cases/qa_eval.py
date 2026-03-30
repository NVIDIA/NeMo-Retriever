"""
QA evaluation test case - measures LLM answer quality given retrieved context.

Requires an existing VDB collection (run e2e or e2e_qa_eval first when using
TopKRetriever). For FileRetriever, no collection is needed -- point
qa_retriever_config.file_path at a pre-computed retrieval results JSON.

Outputs _test_results.json with per-model mean scores, score distributions,
latency stats, and full per-query detail.
"""

import json
import os
import re

from nv_ingest_harness.utils.qa.generators import LiteLLMClient
from nv_ingest_harness.utils.qa.ground_truth import get_qa_dataset_loader
from nv_ingest_harness.utils.qa.judges import LLMJudge
from nv_ingest_harness.utils.qa.orchestrator import QAEvalPipeline
from nv_ingest_harness.utils.qa.retrievers import FileRetriever, TopKRetriever


def _expand_env_vars(value):
    """Recursively expand ${VAR} references in config values."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _build_retriever(config, collection_name: str, model_name: str):
    """Construct the appropriate retriever from config."""
    qa_retriever = getattr(config, "qa_retriever", "topk")
    qa_retriever_config = getattr(config, "qa_retriever_config", None) or {}

    if qa_retriever == "file":
        file_path = qa_retriever_config.get("file_path")
        if not file_path:
            raise ValueError("qa_retriever=file requires qa_retriever_config.file_path to be set")
        return FileRetriever(file_path=file_path)

    # Default: topk -- only import the heavy harness utils when actually needed.
    from nv_ingest_harness.utils.vdb import get_lancedb_path

    vdb_backend = config.vdb_backend
    table_path = None
    if vdb_backend == "lancedb":
        table_path = get_lancedb_path(config, collection_name)

    return TopKRetriever(
        collection_name=collection_name,
        hostname=config.hostname,
        model_name=model_name,
        sparse=config.sparse,
        gpu_search=config.gpu_search,
        vdb_backend=vdb_backend,
        table_path=table_path,
        hybrid=config.hybrid,
    )


def _build_llm_clients(qa_llm_configs: list) -> dict:
    """Build the name -> LiteLLMClient map from config list."""
    clients = {}
    for entry in qa_llm_configs:
        entry = _expand_env_vars(entry)
        name = entry.get("name") or entry.get("model", "llm")
        clients[name] = LiteLLMClient(
            model=entry["model"],
            api_base=entry.get("api_base"),
            api_key=entry.get("api_key"),
            temperature=entry.get("temperature", 0.0),
            max_tokens=entry.get("max_tokens", 512),
            extra_params=entry.get("extra_params", {}),
        )
    return clients


def _build_judge(qa_judge_config: dict) -> LLMJudge:
    """Build the LLMJudge from config dict."""
    cfg = _expand_env_vars(qa_judge_config)
    return LLMJudge(
        model=cfg.get("model", "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"),
        api_base=cfg.get("api_base"),
        api_key=cfg.get("api_key"),
        extra_params=cfg.get("extra_params", {}),
    )


def main(config=None, log_path: str = "test_results") -> int:
    """
    Main entry point for standalone QA evaluation.

    Args:
        config: TestConfig object with qa_* fields populated.
        log_path: Directory for result artifacts.

    Returns:
        0 on success, non-zero on error.
    """
    if config is None:
        print("ERROR: No configuration provided")
        return 2

    qa_dataset = getattr(config, "qa_dataset", None)
    if not qa_dataset:
        print("ERROR: qa_dataset must be specified in configuration")
        print("Set qa_dataset in test_configs.yaml qa_eval section or via QA_DATASET env var")
        return 1

    qa_llm_configs = getattr(config, "qa_llm_configs", None) or []
    if not qa_llm_configs:
        print("ERROR: qa_llm_configs must be specified with at least one LLM")
        return 1

    qa_judge_config = getattr(config, "qa_judge_config", None) or {}
    if not qa_judge_config:
        print("WARNING: qa_judge_config not set, using default judge model")
        qa_judge_config = {}

    qa_top_k = getattr(config, "qa_top_k", 5)
    qa_max_workers = getattr(config, "qa_max_workers", 8)
    qa_retriever = getattr(config, "qa_retriever", "topk")
    ground_truth_dir = getattr(config, "ground_truth_dir", None)

    # Derive collection name (same logic as recall.py for consistency).
    # Only import the heavy harness utils when actually running topk mode.
    test_name = config.test_name or os.path.basename(config.dataset_dir.rstrip("/"))
    if qa_retriever == "topk":
        from nv_ingest_harness.utils.recall import get_recall_collection_name
        from nv_ingest_harness.utils.interact import embed_info

        collection_name = config.collection_name or get_recall_collection_name(test_name)
        model_name, _ = embed_info()
    else:
        collection_name = config.collection_name or test_name
        model_name = None

    print("=" * 60)
    print("QA Evaluation Configuration")
    print("=" * 60)
    print(f"Dataset:        {qa_dataset}")
    print(f"Retriever:      {qa_retriever}")
    print(f"Collection:     {collection_name}")
    print(f"VDB Backend:    {config.vdb_backend}")
    print(f"Top-K:          {qa_top_k}")
    print(f"Max Workers:    {qa_max_workers}")
    print(f"LLMs:           {[e.get('name', e.get('model', '?')) for e in qa_llm_configs]}")
    print(f"Judge:          {qa_judge_config.get('model', '(default)')}")
    print("=" * 60)

    try:
        # Load ground truth -- all loaders accept data_dir uniformly
        loader = get_qa_dataset_loader(qa_dataset)
        qa_pairs = loader(data_dir=ground_truth_dir)

        print(f"\nLoaded {len(qa_pairs)} Q&A pairs from '{qa_dataset}'")

        # Build pipeline components
        retriever = _build_retriever(config, collection_name, model_name)
        if hasattr(retriever, "check_coverage"):
            coverage = retriever.check_coverage(qa_pairs)
            if coverage < 0.5:
                print(
                    f"WARNING: retrieval file covers only {coverage:.0%} of queries -- "
                    "results will be unreliable. Check that the retrieval JSON was "
                    "generated from the same query set."
                )

        llm_clients = _build_llm_clients(qa_llm_configs)
        judge = _build_judge(qa_judge_config)

        pipeline = QAEvalPipeline(
            retriever=retriever,
            llm_clients=llm_clients,
            judge=judge,
            top_k=qa_top_k,
            max_workers=qa_max_workers,
        )

        print(f"\nRunning QA evaluation ({len(qa_pairs)} queries, {len(llm_clients)} LLMs)...")
        eval_results = pipeline.evaluate(qa_pairs)

        # Print summary
        print("\n" + "=" * 60)
        print("QA Evaluation Results")
        print("=" * 60)
        by_model = eval_results.get("by_model", {})
        for name, stats in by_model.items():
            print(f"\nModel: {name}")
            print(f"  Mean Score:    {stats['mean_score']:.3f} / 5.0")
            print(f"  Mean Latency:  {stats['mean_latency_s']:.2f}s")
            print(f"  Scored:        {stats['scored_count']} / {len(qa_pairs)}")
            print(f"  Errors:        {stats['error_count']}")
            dist = stats.get("score_distribution", {})
            print("  Distribution:  " + "  ".join(f"{k}:{v}" for k, v in sorted(dist.items())))

        # Write results
        os.makedirs(log_path, exist_ok=True)
        results_file = os.path.join(log_path, "_test_results.json")
        test_results = {
            "test_type": "qa_eval",
            "dataset": qa_dataset,
            "test_name": test_name,
            "collection_name": collection_name,
            "retriever": qa_retriever,
            "top_k": qa_top_k,
            "qa_results": eval_results,
        }
        with open(results_file, "w") as f:
            json.dump(test_results, f, indent=2)

        print("\n" + "=" * 60)
        print("QA Evaluation Complete")
        print("=" * 60)
        return 0

    except Exception as exc:
        print(f"ERROR: QA evaluation failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
