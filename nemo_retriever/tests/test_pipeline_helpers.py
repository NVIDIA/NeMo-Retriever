# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the private helpers in :mod:`nemo_retriever.pipeline`.

These tests target the non-trivial pieces of logic that the ``retriever
pipeline run`` command relies on:

* ``_resolve_file_patterns``    — file/dir/glob resolution and input-type map.
* ``_build_extract_params``     — translation from CLI flags to ``ExtractParams``
                                  and the nested ``BatchTuningParams``.
* ``_build_embed_params``       — translation from CLI flags to ``EmbedParams``.
* ``_collect_results``          — Ray ``take_all`` vs. in-process DataFrame
                                  handling + ``_count_input_units`` fallback.
* ``_ensure_lancedb_table``     — idempotent LanceDB table creation.

They also exercise the lazy attribute access in
``nemo_retriever.pipeline.__init__`` so the package-level ``app`` / ``run``
exports stay wired to ``__main__``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest
import typer

import nemo_retriever.pipeline as pipeline_pkg
from nemo_retriever.params import EmbedParams, ExtractParams
from nemo_retriever.pipeline.__main__ import (
    _build_embed_params,
    _build_extract_params,
    _collect_results,
    _count_input_units,
    _ensure_lancedb_table,
    _resolve_file_patterns,
)


# =============================================================================
# Package-level lazy exports (pipeline/__init__.py)
# =============================================================================


class TestPipelinePackageExports:
    """The ``nemo_retriever.pipeline`` package forwards ``app`` and ``run``."""

    def test_app_is_forwarded_from_main(self):
        from nemo_retriever.pipeline.__main__ import app as main_app

        assert pipeline_pkg.app is main_app

    def test_run_is_forwarded_from_main(self):
        from nemo_retriever.pipeline.__main__ import run as main_run

        assert pipeline_pkg.run is main_run

    def test_unknown_attribute_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
            _ = pipeline_pkg.does_not_exist  # type: ignore[attr-defined]

    def test_dunder_all(self):
        assert set(pipeline_pkg.__all__) == {"app", "run"}


# =============================================================================
# _resolve_file_patterns
# =============================================================================


class TestResolveFilePatterns:
    """File / directory / input-type resolution."""

    # --- single-file short-circuit ----------------------------------------
    def test_single_existing_file_returned_verbatim(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")

        # input_type does not matter when the path is an existing file.
        result = _resolve_file_patterns(pdf, "pdf")
        assert result == [str(pdf)]

    def test_single_file_ignores_input_type_mismatch(self, tmp_path: Path) -> None:
        """A single file path is returned as-is, even if the extension doesn't match."""
        f = tmp_path / "arbitrary.pdf"
        f.write_bytes(b"not really a pdf")

        # We still get [str(file)] back even though input_type is wildly wrong.
        result = _resolve_file_patterns(f, "audio")
        assert result == [str(f)]

    # --- directory: per-input-type extension maps --------------------------
    @pytest.mark.parametrize(
        "input_type,files,expected_patterns",
        [
            ("pdf", ["a.pdf"], ["*.pdf"]),
            ("txt", ["notes.txt"], ["*.txt"]),
            ("html", ["page.html"], ["*.html"]),
            ("image", ["a.jpg"], ["*.jpg"]),
            ("image", ["b.jpeg"], ["*.jpeg"]),
            ("image", ["c.png"], ["*.png"]),
            ("image", ["d.tiff"], ["*.tiff"]),
            ("image", ["e.bmp"], ["*.bmp"]),
            ("audio", ["clip.mp3"], ["*.mp3"]),
            ("audio", ["clip.wav"], ["*.wav"]),
            ("audio", ["clip.m4a"], ["*.m4a"]),
        ],
    )
    def test_directory_match_by_input_type(
        self,
        tmp_path: Path,
        input_type: str,
        files: list[str],
        expected_patterns: list[str],
    ) -> None:
        for name in files:
            (tmp_path / name).write_bytes(b"x")

        result = _resolve_file_patterns(tmp_path, input_type)

        # Every returned pattern should match at least one expected suffix.
        assert result == [str(tmp_path / p) for p in expected_patterns]

    def test_doc_input_type_matches_both_docx_and_pptx(self, tmp_path: Path) -> None:
        (tmp_path / "a.docx").write_bytes(b"x")
        (tmp_path / "b.pptx").write_bytes(b"x")

        result = _resolve_file_patterns(tmp_path, "doc")

        assert result == [str(tmp_path / "*.docx"), str(tmp_path / "*.pptx")]

    def test_doc_input_type_only_one_family_present(self, tmp_path: Path) -> None:
        """When only docx files are present, we still only get the docx glob back."""
        (tmp_path / "a.docx").write_bytes(b"x")

        result = _resolve_file_patterns(tmp_path, "doc")

        # Only the matching glob survives the non-empty filter.
        assert result == [str(tmp_path / "*.docx")]

    def test_image_input_type_multiple_extensions_filtered(self, tmp_path: Path) -> None:
        """Only extensions that actually match get returned; empty globs are filtered out."""
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.png").write_bytes(b"x")

        result = _resolve_file_patterns(tmp_path, "image")

        assert set(result) == {str(tmp_path / "*.jpg"), str(tmp_path / "*.png")}
        # No unmatched patterns slip through.
        assert not any(p.endswith("*.tiff") for p in result)
        assert not any(p.endswith("*.bmp") for p in result)
        assert not any(p.endswith("*.jpeg") for p in result)

    # --- error paths -------------------------------------------------------
    def test_nonexistent_path_raises_bad_parameter(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(typer.BadParameter, match="Path does not exist"):
            _resolve_file_patterns(missing, "pdf")

    def test_unsupported_input_type_raises_bad_parameter(self, tmp_path: Path) -> None:
        with pytest.raises(typer.BadParameter, match="Unsupported --input-type"):
            _resolve_file_patterns(tmp_path, "parquet")

    def test_directory_with_no_matches_raises_bad_parameter(self, tmp_path: Path) -> None:
        # Directory exists but contains nothing matching the pdf glob.
        (tmp_path / "sidecar.json").write_bytes(b"{}")

        with pytest.raises(typer.BadParameter, match="No files found"):
            _resolve_file_patterns(tmp_path, "pdf")

    def test_accepts_string_path_not_just_pathlib(self, tmp_path: Path) -> None:
        (tmp_path / "sample.txt").write_bytes(b"x")

        # Upstream callers sometimes pass a ``str``; the helper coerces to Path.
        result = _resolve_file_patterns(str(tmp_path), "txt")  # type: ignore[arg-type]

        assert result == [str(Path(tmp_path) / "*.txt")]


# =============================================================================
# _build_extract_params
# =============================================================================


_EXTRACT_BASE = dict(
    method="pdfium",
    dpi=300,
    extract_text=True,
    extract_tables=True,
    extract_charts=True,
    extract_infographics=False,
    extract_page_as_image=True,
    use_graphic_elements=False,
    use_table_structure=False,
    table_output_format=None,
    extract_remote_api_key=None,
    page_elements_invoke_url=None,
    ocr_invoke_url=None,
    graphic_elements_invoke_url=None,
    table_structure_invoke_url=None,
    pdf_split_batch_size=1,
    pdf_extract_batch_size=0,
    pdf_extract_tasks=0,
    pdf_extract_cpus_per_task=0.0,
    page_elements_actors=0,
    page_elements_batch_size=0,
    page_elements_cpus_per_actor=0.0,
    page_elements_gpus_per_actor=None,
    ocr_actors=0,
    ocr_batch_size=0,
    ocr_cpus_per_actor=0.0,
    ocr_gpus_per_actor=None,
    nemotron_parse_actors=0,
    nemotron_parse_batch_size=0,
    nemotron_parse_gpus_per_actor=None,
)


class TestBuildExtractParams:
    """Translation from CLI flags to ``ExtractParams`` + ``BatchTuningParams``."""

    def test_returns_extract_params(self):
        params = _build_extract_params(**_EXTRACT_BASE)
        assert isinstance(params, ExtractParams)
        assert params.method == "pdfium"
        assert params.dpi == 300
        assert params.extract_text is True
        assert params.extract_tables is True
        assert params.extract_charts is True
        assert params.extract_infographics is False
        assert params.extract_page_as_image is True
        assert params.use_graphic_elements is False
        assert params.use_table_structure is False

    def test_dpi_coerced_to_int(self):
        # CLI passes an int, but the helper explicitly coerces — cover the branch.
        overrides = {**_EXTRACT_BASE, "dpi": 150.7}  # type: ignore[dict-item]
        params = _build_extract_params(**overrides)
        assert params.dpi == 150
        assert isinstance(params.dpi, int)

    def test_zero_batch_sizes_are_dropped(self):
        """``... or None`` should drop CLI ``0`` defaults so pydantic defaults win."""
        params = _build_extract_params(**_EXTRACT_BASE)

        tuning = params.batch_tuning
        # The BatchTuningParams defaults (non-None) should remain.
        assert tuning.pdf_extract_batch_size == 4  # pydantic default, not 0
        assert tuning.page_elements_batch_size == 24
        # Workers stay None (CLI 0 -> None -> pydantic Optional default None).
        assert tuning.pdf_extract_workers is None
        assert tuning.page_elements_workers is None
        assert tuning.ocr_workers is None

    def test_pdf_split_batch_size_is_always_applied(self):
        # pdf_split_batch_size lacks ``or None`` so ``1`` gets through verbatim.
        params = _build_extract_params(**_EXTRACT_BASE)
        assert params.batch_tuning.pdf_split_batch_size == 1

        overrides = {**_EXTRACT_BASE, "pdf_split_batch_size": 8}
        params = _build_extract_params(**overrides)
        assert params.batch_tuning.pdf_split_batch_size == 8

    def test_explicit_worker_counts_flow_through(self):
        overrides = {
            **_EXTRACT_BASE,
            "pdf_extract_tasks": 3,
            "pdf_extract_batch_size": 16,
            "pdf_extract_cpus_per_task": 2.5,
            "page_elements_actors": 4,
            "page_elements_batch_size": 32,
            "page_elements_cpus_per_actor": 0.75,
            "ocr_actors": 2,
            "ocr_batch_size": 12,
            "ocr_cpus_per_actor": 1.25,
            "nemotron_parse_actors": 1,
            "nemotron_parse_batch_size": 6,
        }
        params = _build_extract_params(**overrides)

        tuning = params.batch_tuning
        assert tuning.pdf_extract_workers == 3
        assert tuning.pdf_extract_batch_size == 16
        assert tuning.pdf_extract_num_cpus == 2.5
        assert tuning.page_elements_workers == 4
        assert tuning.page_elements_batch_size == 32
        assert tuning.page_elements_cpus_per_actor == 0.75
        assert tuning.ocr_workers == 2
        assert tuning.ocr_inference_batch_size == 12
        assert tuning.ocr_cpus_per_actor == 1.25
        assert tuning.nemotron_parse_workers == 1
        assert tuning.nemotron_parse_batch_size == 6

    # --- GPU gating: remote URL forces gpu_* = 0.0 -----------------------
    def test_page_elements_invoke_url_forces_zero_gpu(self):
        overrides = {
            **_EXTRACT_BASE,
            "page_elements_invoke_url": "http://pe.example/v1",
            "page_elements_gpus_per_actor": 0.5,  # would be 0.5 without URL
        }
        params = _build_extract_params(**overrides)
        assert params.batch_tuning.gpu_page_elements == 0.0
        assert params.page_elements_invoke_url == "http://pe.example/v1"

    def test_ocr_invoke_url_forces_zero_gpu(self):
        overrides = {
            **_EXTRACT_BASE,
            "ocr_invoke_url": "http://ocr.example/v1",
            "ocr_gpus_per_actor": 0.75,
        }
        params = _build_extract_params(**overrides)
        assert params.batch_tuning.gpu_ocr == 0.0
        assert params.ocr_invoke_url == "http://ocr.example/v1"

    def test_local_gpu_values_flow_through_when_no_remote_url(self):
        overrides = {
            **_EXTRACT_BASE,
            "page_elements_gpus_per_actor": 0.25,
            "ocr_gpus_per_actor": 0.5,
            "nemotron_parse_gpus_per_actor": 0.33,
        }
        params = _build_extract_params(**overrides)
        assert params.batch_tuning.gpu_page_elements == 0.25
        assert params.batch_tuning.gpu_ocr == 0.5
        assert params.batch_tuning.gpu_nemotron_parse == 0.33

    def test_gpu_none_values_dropped_when_no_remote_url(self):
        """With no remote URL and gpu=None, the key is dropped and pydantic default applies."""
        overrides = {
            **_EXTRACT_BASE,
            "page_elements_gpus_per_actor": None,
            "ocr_gpus_per_actor": None,
        }
        params = _build_extract_params(**overrides)
        # BatchTuningParams declares these as Optional[float] with default None.
        assert params.batch_tuning.gpu_page_elements is None
        assert params.batch_tuning.gpu_ocr is None

    # --- api_key + endpoints pass through --------------------------------
    def test_api_key_and_endpoints_pass_through(self):
        overrides = {
            **_EXTRACT_BASE,
            "extract_remote_api_key": "nvapi-secret",
            "page_elements_invoke_url": "http://pe/v1",
            "ocr_invoke_url": "http://ocr/v1",
            "graphic_elements_invoke_url": "http://ge/v1",
            "table_structure_invoke_url": "http://ts/v1",
            "table_output_format": "markdown",
        }
        params = _build_extract_params(**overrides)
        assert params.api_key == "nvapi-secret"
        assert params.page_elements_invoke_url == "http://pe/v1"
        assert params.ocr_invoke_url == "http://ocr/v1"
        assert params.graphic_elements_invoke_url == "http://ge/v1"
        assert params.table_structure_invoke_url == "http://ts/v1"
        # The model validator auto-enables feature flags when URLs are set.
        assert params.use_graphic_elements is True
        assert params.use_table_structure is True
        assert params.table_output_format == "markdown"

    def test_none_values_are_filtered_before_pydantic(self):
        """Keys whose CLI value is ``None`` must be omitted, not forwarded as ``None``."""
        overrides = {
            **_EXTRACT_BASE,
            "table_output_format": None,
            "extract_remote_api_key": None,
        }
        params = _build_extract_params(**overrides)
        # ``table_output_format=None`` gets dropped and the model validator fills
        # it in based on ``use_table_structure`` (False → "pseudo_markdown").
        assert params.table_output_format == "pseudo_markdown"
        # api_key is truly None since it wasn't forced.
        assert params.api_key is None


# =============================================================================
# _build_embed_params
# =============================================================================


_EMBED_BASE = dict(
    embed_model_name="nvidia/test-embed",
    embed_invoke_url=None,
    embed_remote_api_key=None,
    embed_modality="text",
    text_elements_modality=None,
    structured_elements_modality=None,
    embed_granularity="element",
    embed_actors=0,
    embed_batch_size=0,
    embed_cpus_per_actor=0.0,
    embed_gpus_per_actor=None,
)


class TestBuildEmbedParams:
    """Translation from CLI flags to ``EmbedParams`` + ``BatchTuningParams``."""

    def test_returns_embed_params(self):
        params = _build_embed_params(**_EMBED_BASE)
        assert isinstance(params, EmbedParams)
        assert params.model_name == "nvidia/test-embed"
        assert params.embed_modality == "text"
        assert params.embed_granularity == "element"

    def test_zero_batch_size_dropped_to_pydantic_default(self):
        params = _build_embed_params(**_EMBED_BASE)
        # ``embed_batch_size or None`` drops 0; pydantic default (256) wins.
        assert params.batch_tuning.embed_batch_size == 256
        # Same for embed_workers (Optional, default None).
        assert params.batch_tuning.embed_workers is None
        # inference_batch_size comes from the same ``or None`` pattern;
        # pydantic default is 32.
        assert params.inference_batch_size == 32

    def test_explicit_batch_params_flow_through(self):
        overrides = {
            **_EMBED_BASE,
            "embed_actors": 3,
            "embed_batch_size": 64,
            "embed_cpus_per_actor": 1.5,
        }
        params = _build_embed_params(**overrides)
        assert params.batch_tuning.embed_workers == 3
        assert params.batch_tuning.embed_batch_size == 64
        assert params.batch_tuning.embed_cpus_per_actor == 1.5
        # ``inference_batch_size`` is also pinned from ``embed_batch_size``.
        assert params.inference_batch_size == 64

    def test_embed_invoke_url_forces_zero_gpu(self):
        overrides = {
            **_EMBED_BASE,
            "embed_invoke_url": "http://embed.example/v1",
            "embed_gpus_per_actor": 0.5,
            "embed_remote_api_key": "nvapi-xyz",
        }
        params = _build_embed_params(**overrides)
        assert params.batch_tuning.gpu_embed == 0.0
        assert params.embed_invoke_url == "http://embed.example/v1"
        assert params.api_key == "nvapi-xyz"

    def test_local_embed_gpu_value_flows_through(self):
        overrides = {**_EMBED_BASE, "embed_gpus_per_actor": 0.25}
        params = _build_embed_params(**overrides)
        assert params.batch_tuning.gpu_embed == 0.25

    def test_local_embed_gpu_none_dropped(self):
        overrides = {**_EMBED_BASE, "embed_gpus_per_actor": None}
        params = _build_embed_params(**overrides)
        assert params.batch_tuning.gpu_embed is None

    def test_modality_overrides_pass_through(self):
        overrides = {
            **_EMBED_BASE,
            "embed_modality": "text_image",
            "text_elements_modality": "text",
            "structured_elements_modality": "image",
            "embed_granularity": "element",
        }
        params = _build_embed_params(**overrides)
        assert params.embed_modality == "text_image"
        assert params.text_elements_modality == "text"
        assert params.structured_elements_modality == "image"

    def test_invalid_modality_raises_pydantic_validation_error(self):
        overrides = {**_EMBED_BASE, "embed_modality": "image_text"}
        with pytest.raises(Exception, match="text_image"):
            _build_embed_params(**overrides)


# =============================================================================
# _collect_results  (and _count_input_units)
# =============================================================================


class _FakeRayDataset:
    """Minimal stand-in for a Ray Data dataset produced by ``ingestor.ingest()``."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.take_all_calls = 0

    def take_all(self) -> list[dict[str, Any]]:
        self.take_all_calls += 1
        return list(self._records)


class TestCollectResults:
    """Ray (batch) and pandas (inprocess) result materialization."""

    def test_batch_mode_calls_take_all_and_builds_dataframe(self):
        rows = [
            {"source_id": "a", "text": "hello"},
            {"source_id": "a", "text": "world"},
            {"source_id": "b", "text": "!"},
        ]
        fake = _FakeRayDataset(rows)

        records, df, download_time, num_units = _collect_results("batch", fake)

        assert fake.take_all_calls == 1
        assert records == rows
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["source_id", "text"]
        assert len(df) == 3
        # ``source_id`` has two distinct values → that is the unit count.
        assert num_units == 2
        # Elapsed time is measured with ``perf_counter``; it must be non-negative.
        assert download_time >= 0.0

    def test_batch_mode_handles_empty_result(self):
        fake = _FakeRayDataset([])
        records, df, download_time, num_units = _collect_results("batch", fake)
        assert records == []
        assert df.empty
        # Empty DataFrame has no columns → falls through to len(df.index) == 0.
        assert num_units == 0
        assert download_time >= 0.0

    def test_inprocess_mode_accepts_dataframe_directly(self):
        rows = [
            {"source_id": "a", "text": "x"},
            {"source_id": "b", "text": "y"},
        ]
        df_in = pd.DataFrame(rows)

        records, df_out, download_time, num_units = _collect_results("inprocess", df_in)

        # The DataFrame is passed through unchanged (same object).
        assert df_out is df_in
        assert records == rows
        # inprocess mode never incurs Ray download time.
        assert download_time == 0.0
        assert num_units == 2


class TestCountInputUnits:
    """``_count_input_units`` fallback chain: source_id -> source_path -> len."""

    def test_prefers_source_id(self):
        df = pd.DataFrame(
            {
                "source_id": ["a", "a", "b"],
                "source_path": ["/p1", "/p1", "/p2"],
                "text": ["x", "y", "z"],
            }
        )
        assert _count_input_units(df) == 2

    def test_falls_back_to_source_path(self):
        df = pd.DataFrame(
            {
                "source_path": ["/p1", "/p2", "/p2", "/p3"],
                "text": ["x", "y", "z", "w"],
            }
        )
        assert _count_input_units(df) == 3

    def test_falls_back_to_len_when_no_source_columns(self):
        df = pd.DataFrame({"text": ["x", "y", "z", "w"]})
        assert _count_input_units(df) == 4

    def test_empty_dataframe_without_columns(self):
        df = pd.DataFrame()
        assert _count_input_units(df) == 0


# =============================================================================
# _ensure_lancedb_table
# =============================================================================


class TestEnsureLancedbTable:
    """Idempotent creation of the LanceDB table used by ``pipeline run``."""

    def test_creates_uri_directory_when_missing(self, tmp_path: Path):
        uri = tmp_path / "new_lancedb"
        assert not uri.exists()

        _ensure_lancedb_table(str(uri), "nv-ingest")

        assert uri.exists() and uri.is_dir()

    def test_creates_table_with_lancedb_schema(self, tmp_path: Path):
        import lancedb
        from nemo_retriever.vector_store.lancedb_utils import lancedb_schema

        uri = tmp_path / "lancedb"
        _ensure_lancedb_table(str(uri), "nv-ingest")

        db = lancedb.connect(str(uri))
        # Table exists and is empty.
        tbl = db.open_table("nv-ingest")
        assert tbl.count_rows() == 0

        # The schema matches the canonical lancedb_schema() pa schema.
        expected = lancedb_schema()
        actual = tbl.schema
        assert actual.names == expected.names
        for name in expected.names:
            assert actual.field(name).type == expected.field(name).type

    def test_is_idempotent_when_table_already_exists(self, tmp_path: Path):
        import lancedb

        uri = tmp_path / "lancedb"
        _ensure_lancedb_table(str(uri), "nv-ingest")

        # Seed one row so we can confirm the table was not recreated/emptied.
        db = lancedb.connect(str(uri))
        tbl = db.open_table("nv-ingest")
        schema = tbl.schema
        payload = {f.name: [None] for f in schema}
        payload["page_number"] = [1]
        payload["text"] = ["seed"]
        row = pa.table(payload, schema=schema)
        tbl.add(row)
        assert tbl.count_rows() == 1

        # Second call must be a no-op — the existing data must survive.
        _ensure_lancedb_table(str(uri), "nv-ingest")
        tbl = lancedb.connect(str(uri)).open_table("nv-ingest")
        assert tbl.count_rows() == 1

    def test_respects_custom_table_name(self, tmp_path: Path):
        import lancedb

        uri = tmp_path / "lancedb"
        _ensure_lancedb_table(str(uri), "custom-name")

        db = lancedb.connect(str(uri))
        # Prefer the non-deprecated ``list_tables`` API; fall back to
        # ``table_names`` on older LanceDB releases.  Both return either a flat
        # sequence of names or a pydantic-style response with a ``tables`` attr.
        raw = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
        names = getattr(raw, "tables", None) or list(raw)
        assert "custom-name" in names
