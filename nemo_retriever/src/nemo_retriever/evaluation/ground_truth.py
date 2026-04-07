# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ground truth dataset loaders for the QA evaluation pipeline.

Loaders return a uniform list of dicts with at least "query" and "answer" keys.

Supported datasets:
  - bo767_infographic: 369 infographic Q&A pairs from the Digital Corpora bo767 corpus.
  - vidore/<dataset_id>: Any ViDoRe v3 dataset hosted on HuggingFace.
  - csv:/path/to/file.csv: Any CSV with "query" and "answer" columns.
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def load_infographic_qa(data_dir: str) -> list[dict]:
    """Load bo767 infographic Q&A pairs from the Digital Corpora CSV.

    Parameters
    ----------
    data_dir : str
        Directory containing ``digital_corpora_infographic_query_answer.csv``.
        Required -- no default path assumed.
    """
    csv_path = os.path.join(data_dir, "digital_corpora_infographic_query_answer.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Infographic QA CSV not found at {csv_path}. "
            "Expected at <data_dir>/digital_corpora_infographic_query_answer.csv"
        )

    required = {"query", "answer"}
    records: list[dict] = []
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
    """Load Q&A pairs from a ViDoRe v3 dataset hosted on HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' library is required for ViDoRe v3 loading. " "Install it: pip install datasets>=2.19.0"
        ) from exc

    load_kwargs: dict = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    try:
        ds = load_dataset(dataset_name, "queries", split="test", **load_kwargs)
    except (ValueError, TypeError):
        ds = load_dataset(dataset_name, split="queries", **load_kwargs)

    column_names = ds.column_names
    required = ["query", "answer"]
    missing = [c for c in required if c not in column_names]
    if missing:
        raise ValueError(
            f"ViDoRe v3 dataset '{dataset_name}' queries split is missing columns: {missing}. "
            f"Available columns: {column_names}"
        )

    records: list[dict] = []
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
        logger.warning("ViDoRe loader: skipped %d rows with empty query or answer", skipped)

    return records


def load_generic_csv(csv_path: str) -> list[dict]:
    """Load Q&A pairs from any CSV file with 'query' and 'answer' columns."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    required = {"query", "answer"}
    records: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV {csv_path} missing required columns: {missing}. " f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            query = row.get("query", "").strip()
            answer = row.get("answer", "").strip()
            if query and answer:
                records.append({**row, "query": query, "answer": answer})

    return records


def get_qa_dataset_loader(dataset_name: str) -> Callable[[Optional[str]], list[dict]]:
    """Return the loader function for a given dataset name.

    Built-in mappings:
      "bo767_infographic"  -> load_infographic_qa(data_dir)
      "vidore/<anything>"  -> load_vidore_v3_qa(dataset_name)
      "csv:/path/to.csv"   -> load_generic_csv(path)
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
    )
