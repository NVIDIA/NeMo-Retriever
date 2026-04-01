# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RayDataExecutor stage-by-stage diagnostic for the graph pipeline.

Run with::

    source /opt/retriever_runtime/bin/activate
    CUDA_VISIBLE_DEVICES=3,4,5,6,7 NEMO_RETRIEVER_HF_CACHE_DIR=./hf_cache/ \
        python nemo_retriever/src/nemo_retriever/examples/test_raydata_graph_stages.py /raid/data/jp20/
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from nemo_retriever.graph import Graph, RayDataExecutor

logger = logging.getLogger(__name__)


def _resolve_input_pdfs(input_path: Path, sample_size: int = 2) -> list[str]:
    if input_path.is_file():
        return [str(input_path)]
    if not input_path.is_dir():
        raise ValueError(f"Path does not exist: {input_path}")

    file_patterns = sorted(str(path) for path in input_path.glob("*.pdf"))
    if not file_patterns:
        raise ValueError(f"No PDF files found in {input_path}")
    return file_patterns[:sample_size]


def _to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records)


def _summarize_series(df: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in df.columns:
        return {"present": False}

    values = df[column].tolist()
    non_null = sum(value is not None for value in values)
    truthy = sum(bool(value) for value in values)
    sample = values[0] if values else None
    return {
        "present": True,
        "non_null": int(non_null),
        "truthy": int(truthy),
        "sample_type": type(sample).__name__ if sample is not None else None,
    }


def _print_stage_summary(stage_name: str, df: pd.DataFrame, extra: dict[str, Any] | None = None) -> None:
    summary = {
        "stage": stage_name,
        "rows": int(len(df.index)),
        "columns": sorted(str(col) for col in df.columns.tolist()),
    }
    if extra:
        summary.update(extra)
    print(json.dumps(summary, default=str, indent=2))


def _require_columns(df: pd.DataFrame, stage_name: str, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise AssertionError(f"{stage_name}: missing required columns: {missing}")


def _validate_split(df: pd.DataFrame) -> dict[str, Any]:
    _require_columns(df, "PDFSplitActor", ["path", "bytes", "page_number", "source_id"])
    if df.empty:
        raise AssertionError("PDFSplitActor: produced no rows")
    return {"page_number": _summarize_series(df, "page_number")}


def _validate_extract(df: pd.DataFrame) -> dict[str, Any]:
    _require_columns(df, "PDFExtractionActor", ["text", "metadata", "page_image", "page_number"])
    if df.empty:
        raise AssertionError("PDFExtractionActor: produced no rows")
    metadata = [value for value in df["metadata"].tolist() if isinstance(value, dict)]
    if not metadata:
        raise AssertionError("PDFExtractionActor: metadata column did not contain dict payloads")
    errors = sum(meta.get("error") is not None for meta in metadata)
    non_empty_text = sum(bool((value or "").strip()) for value in df["text"].tolist())
    return {
        "non_empty_text_rows": int(non_empty_text),
        "metadata_errors": int(errors),
    }


def _validate_page_elements(df: pd.DataFrame) -> dict[str, Any]:
    _require_columns(
        df,
        "PageElementDetectionActor",
        ["page_elements_v3", "page_elements_v3_num_detections", "page_elements_v3_counts_by_label"],
    )
    detections = df["page_elements_v3_num_detections"].fillna(0).astype(int)
    return {
        "rows_with_detections": int((detections > 0).sum()),
        "max_detections_per_row": int(detections.max()) if not detections.empty else 0,
    }


def _validate_ocr(df: pd.DataFrame) -> dict[str, Any]:
    _require_columns(df, "OCRActor", ["text", "ocr_v1", "table", "chart", "infographic"])
    ocr_meta = [value for value in df["ocr_v1"].tolist() if isinstance(value, dict)]
    if not ocr_meta:
        raise AssertionError("OCRActor: ocr_v1 column did not contain dict payloads")
    errors = sum(meta.get("error") is not None for meta in ocr_meta)
    non_empty_text = sum(bool((value or "").strip()) for value in df["text"].tolist())
    return {
        "non_empty_text_rows": int(non_empty_text),
        "ocr_errors": int(errors),
    }


def _validate_explode(df: pd.DataFrame) -> dict[str, Any]:
    _require_columns(df, "ExplodeContentActor", ["text", "_embed_modality", "source_id"])
    if df.empty:
        raise AssertionError("ExplodeContentActor: produced no rows")
    non_empty_text = sum(bool((value or "").strip()) for value in df["text"].tolist())
    return {
        "non_empty_text_rows": int(non_empty_text),
        "embed_modality": _summarize_series(df, "_embed_modality"),
    }


def _validate_embed(df: pd.DataFrame) -> dict[str, Any]:
    _require_columns(
        df,
        "_BatchEmbedActor",
        [
            "text_embeddings_1b_v2",
            "text_embeddings_1b_v2_dim",
            "text_embeddings_1b_v2_has_embedding",
            "_contains_embeddings",
        ],
    )
    has_embedding = df["text_embeddings_1b_v2_has_embedding"].fillna(False).astype(bool)
    contains_embeddings = df["_contains_embeddings"].fillna(False).astype(bool)
    embedded_rows = int(has_embedding.sum())
    return {
        "rows_with_payload_embedding": embedded_rows,
        "rows_with_contains_embeddings": int(contains_embeddings.sum()),
    }


def _node_overrides() -> dict[str, dict[str, Any]]:
    return {
        "PDFSplitActor": {"batch_size": 1, "num_cpus": 1},
        "PDFExtractionActor": {"batch_size": 4, "num_cpus": 1},
        "PageElementDetectionActor": {"batch_size": 8, "num_gpus": 0.5},
        "OCRActor": {"batch_size": 8, "num_gpus": 0.5},
        "ExplodeContentActor": {"batch_size": 256, "num_cpus": 1, "num_gpus": 0},
        "_BatchEmbedActor": {"batch_size": 256, "num_gpus": 0.5},
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if len(sys.argv) < 2:
        print(
            "Usage: python nemo_retriever/src/nemo_retriever/examples/test_raydata_graph_stages.py <input-dir-or-file>",
            file=sys.stderr,
        )
        return 2

    input_path = Path(sys.argv[1]).expanduser()
    file_patterns = _resolve_input_pdfs(input_path)

    from nemo_retriever.pdf.split import PDFSplitActor
    from nemo_retriever.pdf.extract import PDFExtractionActor
    from nemo_retriever.page_elements.page_elements import PageElementDetectionActor
    from nemo_retriever.ocr.ocr import OCRActor
    from nemo_retriever.operators.content import ExplodeContentActor
    from nemo_retriever.params import EmbedParams
    from nemo_retriever.operators.embedding import _BatchEmbedActor

    extract_kwargs: dict[str, Any] = {
        "method": "pdfium",
        "dpi": 300,
        "extract_text": True,
        "extract_tables": True,
        "extract_charts": True,
        "extract_page_as_image": True,
    }
    ocr_kwargs: dict[str, Any] = {
        "extract_text": True,
        "extract_tables": True,
        "extract_charts": True,
    }
    explode_kwargs: dict[str, Any] = {"modality": "text"}
    embed_params = EmbedParams(model_name="nvidia/llama-nemotron-embed-1b-v2")

    stages: list[tuple[str, Any, Callable[[pd.DataFrame], dict[str, Any]]]] = [
        ("PDFSplitActor", Graph() >> PDFSplitActor(), _validate_split),
        ("PDFExtractionActor", PDFSplitActor() >> PDFExtractionActor(**extract_kwargs), _validate_extract),
        (
            "PageElementDetectionActor",
            PDFSplitActor() >> PDFExtractionActor(**extract_kwargs) >> PageElementDetectionActor(),
            _validate_page_elements,
        ),
        (
            "OCRActor",
            PDFSplitActor() >> PDFExtractionActor(**extract_kwargs) >> PageElementDetectionActor() >> OCRActor(**ocr_kwargs),
            _validate_ocr,
        ),
        (
            "ExplodeContentActor",
            PDFSplitActor()
            >> PDFExtractionActor(**extract_kwargs)
            >> PageElementDetectionActor()
            >> OCRActor(**ocr_kwargs)
            >> ExplodeContentActor(**explode_kwargs),
            _validate_explode,
        ),
        (
            "_BatchEmbedActor",
            PDFSplitActor()
            >> PDFExtractionActor(**extract_kwargs)
            >> PageElementDetectionActor()
            >> OCRActor(**ocr_kwargs)
            >> ExplodeContentActor(**explode_kwargs)
            >> _BatchEmbedActor(params=embed_params),
            _validate_embed,
        ),
    ]

    import ray

    exit_code = 0
    for stage_name, graph, validator in stages:
        logger.info("Running RayDataExecutor diagnostic through %s", stage_name)
        if ray.is_initialized():
            ray.shutdown()

        executor = RayDataExecutor(graph, batch_size=1, node_overrides=_node_overrides())
        records = executor.ingest(file_patterns).take_all()
        df = _to_dataframe(records)
        extra = validator(df)
        _print_stage_summary(stage_name, df, extra)

        if stage_name == "_BatchEmbedActor" and int(extra["rows_with_payload_embedding"]) <= 0:
            raise AssertionError("_BatchEmbedActor: no rows reported text_embeddings_1b_v2_has_embedding=True")

    if ray.is_initialized():
        ray.shutdown()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
