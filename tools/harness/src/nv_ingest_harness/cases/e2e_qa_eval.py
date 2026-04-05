"""
E2E QA evaluation test case - fresh ingestion followed by QA evaluation.

Calls e2e.py to handle ingestion and collection creation, then qa_eval.py
to evaluate LLM answer quality against the newly ingested collection.

Use this to measure end-to-end: extract_tables=true vs false, or different
ingestion settings, and see how they affect downstream answer accuracy.
"""

import json
import os

from nv_ingest_harness.cases.e2e import main as e2e_main
from nv_ingest_harness.cases.qa_eval import main as qa_eval_main
from nv_ingest_harness.utils.recall import get_recall_collection_name


def main(config=None, log_path: str = "test_results") -> int:
    """
    Main entry point for E2E QA evaluation.

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

    test_name = config.test_name or os.path.basename(config.dataset_dir.rstrip("/"))
    collection_name = get_recall_collection_name(test_name)

    # Ensure TopKRetriever uses the collection created by this e2e run
    original_collection_name = config.collection_name
    config.collection_name = collection_name

    print("=" * 60)
    print("E2E QA Evaluation Configuration")
    print("=" * 60)
    print(f"Dataset:     {config.dataset_dir}")
    print(f"Test Name:   {test_name}")
    print(f"Collection:  {collection_name}")
    print(f"QA Dataset:  {qa_dataset}")
    print("=" * 60)

    # Step 1: Ingestion
    print("\n" + "=" * 60)
    print("Step 1: Running Ingestion (via e2e)")
    print("=" * 60)

    e2e_rc = e2e_main(config=config, log_path=log_path)
    if e2e_rc != 0:
        print(f"ERROR: Ingestion failed with exit code: {e2e_rc}")
        config.collection_name = original_collection_name
        return e2e_rc

    # Load e2e results before qa_eval overwrites _test_results.json
    results_file = os.path.join(log_path, "_test_results.json")
    e2e_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                e2e_data = json.load(f)
            e2e_results = {
                "test_config": e2e_data.get("test_config", {}),
                "results": e2e_data.get("results", {}),
            }
        except (json.JSONDecodeError, IOError):
            pass

    # Step 2: QA evaluation
    print("\n" + "=" * 60)
    print("Step 2: Running QA Evaluation (via qa_eval)")
    print("=" * 60)

    qa_rc = qa_eval_main(config=config, log_path=log_path)
    if qa_rc != 0:
        print(f"Warning: QA evaluation returned non-zero exit code: {qa_rc}")

    # Restore original collection_name
    config.collection_name = original_collection_name

    # Load QA results and build combined output
    qa_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                qa_data = json.load(f)
            qa_results = qa_data.get("qa_results", {})
        except (json.JSONDecodeError, IOError):
            pass

    combined = {
        "test_type": "e2e_qa_eval",
        "test_config": {
            "test_name": test_name,
            "collection_name": collection_name,
            "qa_dataset": qa_dataset,
        },
        "ingestion_results": e2e_results.get("results", {}),
        "qa_results": qa_results,
    }

    # Carry over relevant ingestion config fields
    for key in ["api_version", "dataset_dir", "hostname", "model_name", "dense_dim", "sparse", "gpu_search"]:
        if key in e2e_results.get("test_config", {}):
            combined["test_config"][key] = e2e_results["test_config"][key]

    with open(results_file, "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{test_name} e2e_qa_eval Summary")
    print("=" * 60)

    by_model = qa_results.get("by_model", {})
    for name, stats in by_model.items():
        print(
            f"  {name}: mean_score={stats.get('mean_score', 0):.3f} " f"latency={stats.get('mean_latency_s', 0):.2f}s"
        )

    return 0 if qa_rc == 0 else qa_rc


if __name__ == "__main__":
    raise SystemExit(main())
