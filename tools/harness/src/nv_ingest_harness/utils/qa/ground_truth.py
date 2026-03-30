"""
Ground truth dataset loaders for the QA evaluation pipeline.

Loaders return a uniform list of dicts with at least "query" and "answer" keys.
Additional metadata columns are preserved for debugging and analysis.

Supported datasets:
  - bo767_infographic: 369 infographic Q&A pairs from the Digital Corpora bo767 corpus.
  - vidore/<dataset_id>: Any ViDoRe v3 dataset hosted on HuggingFace, loaded via
                         the `datasets` library (e.g., "vidore/vidore_v3_finance_en").
  - csv:/path/to/file.csv: Any CSV with "query" and "answer" columns. All other
                            columns are preserved as metadata.

Registry:
  get_qa_dataset_loader("bo767_infographic")  -> load_infographic_qa
  get_qa_dataset_loader("vidore/...")          -> load_vidore_v3_qa
  get_qa_dataset_loader("csv:/path/to.csv")   -> load_generic_csv (partial)

To add a new dataset, implement a loader function with signature:
    def my_loader(data_dir: Optional[str] = None) -> list[dict]
and register it in get_qa_dataset_loader().
"""

from __future__ import annotations

import csv
import os
from typing import Callable, Optional


def load_infographic_qa(data_dir: Optional[str] = None) -> list[dict]:
    """
    Load bo767 infographic Q&A pairs from the Digital Corpora CSV.

    The CSV (digital_corpora_infographic_query_answer.csv) contains 369 rows
    with human-authored questions and answers for infographic pages from the
    bo767 PDF corpus.

    Columns: modality, query, answer, pdf, page

    Args:
        data_dir: Directory containing the CSV file.
                  Defaults to the repo's tools/harness/data/ directory.

    Returns:
        List of dicts with keys: query, answer, pdf, page, modality.
    """
    if data_dir is None:
        # Lazy import: get_repo_root lives in the heavy harness utils; only needed
        # when data_dir is not explicitly provided (e.g. inside Docker it always is).
        from nv_ingest_harness.utils.cases import get_repo_root

        data_dir = os.path.join(get_repo_root(), "tools", "harness", "data")

    csv_path = os.path.join(data_dir, "digital_corpora_infographic_query_answer.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Infographic QA CSV not found at {csv_path}. "
            "Expected at tools/harness/data/digital_corpora_infographic_query_answer.csv"
        )

    required = {"query", "answer"}
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Infographic QA CSV missing required columns: {missing}")
        for row in reader:
            query = row.get("query", "").strip()
            answer = row.get("answer", "").strip()
            if query and answer:
                records.append({**row, "query": query, "answer": answer})

    return records


def load_vidore_v3_qa(dataset_name: str, cache_dir: Optional[str] = None) -> list[dict]:
    """
    Load Q&A pairs from a ViDoRe v3 dataset hosted on HuggingFace.

    ViDoRe v3 datasets include human-verified reference answers in the
    "queries" split alongside the query text. This loader extracts those
    pairs in the same format as load_infographic_qa.

    The qrels split (relevance judgements) is intentionally not loaded here.
    It is only needed for building a FileRetriever JSON pre-seeded with
    ground-truth pages, which is a separate offline step.

    Args:
        dataset_name: HuggingFace dataset identifier, e.g.
                      "vidore/vidore_v3_finance_en".
        cache_dir: Local directory for the HuggingFace datasets cache.
                   Defaults to the HuggingFace default (~/.cache/huggingface).

    Returns:
        List of dicts with keys: query, answer, query_id (and any other
        columns present in the queries split).

    Raises:
        ImportError: If the `datasets` library is not installed.
        ValueError: If the dataset's queries split lacks "query" or "answer" columns.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required for ViDoRe v3 loading. "
            "Install it: pip install datasets>=2.19.0"
        ) from exc

    load_kwargs: dict = {"split": "queries"}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    ds = load_dataset(dataset_name, **load_kwargs)

    column_names = ds.column_names
    required = ["query", "answer"]
    missing = [c for c in required if c not in column_names]
    if missing:
        raise ValueError(
            f"ViDoRe v3 dataset '{dataset_name}' queries split is missing columns: {missing}. "
            f"Available columns: {column_names}"
        )

    records = []
    skipped = 0
    for row in ds:
        query = str(row["query"]).strip()
        answer = str(row["answer"]).strip()
        if not query or not answer:
            skipped += 1
            continue
        record = {"query": query, "answer": answer}
        for col in column_names:
            if col not in record:
                record[col] = row[col]
        records.append(record)

    if skipped:
        print(f"  [ViDoRe loader] Skipped {skipped} rows with empty query or answer")

    return records


def load_generic_csv(csv_path: str) -> list[dict]:
    """
    Load Q&A pairs from any CSV file with 'query' and 'answer' columns.

    All other columns are preserved as metadata in each record. This allows
    researchers to bring their own dataset without writing code -- just point
    the harness at a CSV.

    Args:
        csv_path: Absolute or relative path to the CSV file.

    Returns:
        List of dicts with at least keys: query, answer.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    required = {"query", "answer"}
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {missing}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            query = row.get("query", "").strip()
            answer = row.get("answer", "").strip()
            if query and answer:
                records.append({**row, "query": query, "answer": answer})

    return records


def get_qa_dataset_loader(dataset_name: str) -> Callable[[Optional[str]], list[dict]]:
    """
    Return the loader function for a given dataset name.

    Contract: all returned callables have signature (data_dir: Optional[str] = None)
    so callers can always invoke loader(data_dir=...) uniformly without knowing the
    dataset type. This keeps the caller free of dataset-specific branching.

    Built-in mappings:
      "bo767_infographic"  -> load_infographic_qa(data_dir)
      "vidore/<anything>"  -> load_vidore_v3_qa(dataset_name) -- data_dir accepted but
                              maps to HuggingFace cache_dir if provided
      "csv:/path/to.csv"   -> load_generic_csv(path) -- data_dir is ignored

    Args:
        dataset_name: Dataset identifier string. Use "csv:/path/to/file.csv" for
                      custom datasets with query and answer columns.

    Returns:
        Callable with signature (data_dir: Optional[str] = None) -> list[dict].

    Raises:
        ValueError: If the dataset name is not recognized.
    """
    lower = dataset_name.lower()

    if lower == "bo767_infographic":
        return load_infographic_qa

    if lower.startswith("vidore/"):
        _ds_name = dataset_name

        def _vidore_loader(data_dir: Optional[str] = None) -> list[dict]:
            return load_vidore_v3_qa(_ds_name, cache_dir=data_dir)

        return _vidore_loader

    if dataset_name.startswith("csv:"):
        _csv_path = dataset_name[4:]

        def _csv_loader(data_dir: Optional[str] = None) -> list[dict]:
            return load_generic_csv(_csv_path)

        return _csv_loader

    raise ValueError(
        f"Unknown QA dataset: '{dataset_name}'. "
        "Built-in datasets: 'bo767_infographic', 'vidore/<hf_dataset_id>', "
        "'csv:/path/to/file.csv'. "
        "To add a new dataset, implement a loader in utils/qa/ground_truth.py "
        "and register it in get_qa_dataset_loader()."
    )
