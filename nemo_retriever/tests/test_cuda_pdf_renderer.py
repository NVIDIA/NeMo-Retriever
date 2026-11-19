# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in batching, cleanup and fail-closed behavior without CUDA hardware."""

import io
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import pytest
from nemo_retriever.common.params import ExtractParams
from nemo_retriever.graph.graph_pipeline_registry import get_node_kwargs
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.extract.pdf import cuda_renderer as adapter
from nemo_retriever.operators.extract.pdf import extract
from pydantic import ValidationError


@pytest.fixture
def pages():
    document = pdfium.PdfDocument.new()
    try:
        document.new_page(72, 100).close()
        stream = io.BytesIO()
        document.save(stream)
    finally:
        document.close()
    return pd.DataFrame({"bytes": [stream.getvalue()], "page_number": [1]})


@pytest.fixture
def backend(monkeypatch):
    calls, bitmaps = [], []

    class Bitmap:
        mode = "RGB"

        def __init__(self):
            self.closed = False

        def to_numpy(self):
            return np.full((32, 24, 3), 255, dtype=np.uint8)

        def close(self):
            self.closed = True

    def render_batch(packets, scales, rotations):
        calls.append((packets, scales, rotations))
        result = [Bitmap() for _ in packets]
        bitmaps.extend(result)
        return result

    monkeypatch.setitem(
        sys.modules, "cuda_pdf_renderer", SimpleNamespace(render_batch=render_batch)
    )

    def no_cpu(*args, **kwargs):
        pytest.fail("CUDA selection attempted CPU rasterization")

    monkeypatch.setattr(pdfium.PdfPage, "render", no_cpu)
    return calls, bitmaps


def run(pages, **kwargs):
    return extract.pdf_extraction(
        pages,
        extract_page_as_image=True,
        image_format="png",
        pdf_renderer="cuda",
        **kwargs,
    )


def test_default_and_graph_wiring():
    assert ExtractParams().pdf_renderer == "pdfium"
    with pytest.raises(ValidationError):
        ExtractParams(pdf_renderer="unknown")
    graph = build_graph(
        extraction_mode="pdf", extract_params=ExtractParams(pdf_renderer="cuda")
    )
    assert get_node_kwargs(graph, "PDFExtractionActor")["pdf_renderer"] == "cuda"


def test_batch_order_cleanup_and_duplicate_packets(pages, backend):
    data = pd.concat(
        [pages.assign(page_number=i) for i in (3, 1, 2)], ignore_index=True
    )
    result = run(data)
    calls, bitmaps = backend
    assert result.page_number.tolist() == [3, 1, 2]
    assert len(calls) == 1 and len(calls[0][0]) == 3
    assert all(bitmap.closed for bitmap in bitmaps)
    assert adapter._CURRENT_BATCH.get() is None


def test_abi_limit_subdivides_source_batch(pages, backend):
    result = run(pd.concat([pages] * 257, ignore_index=True))
    assert len(result) == 257
    assert [len(call[0]) for call in backend[0]] == [256, 1]
    assert all(bitmap.closed for bitmap in backend[1])


def test_image_crop_callback_and_rotation(pages, backend, monkeypatch):
    def crops(page, **kwargs):
        bitmap = kwargs["render_page"](page, scale=1.0, rotation=page.get_rotation())
        assert not bitmap.closed
        return []

    monkeypatch.setattr(extract, "extract_image_like_objects_from_pdfium_page", crops)
    assert len(run(pages, extract_images=True)) == 1
    assert len(backend[0][0][0]) == 2
    assert all(bitmap.closed for bitmap in backend[1])


def test_swallowed_crop_render_failure_still_stops_batch(pages, backend, monkeypatch):
    def crops(page, **kwargs):
        kwargs["render_page"](page, scale=99)

    monkeypatch.setattr(extract, "extract_image_like_objects_from_pdfium_page", crops)
    with pytest.raises(adapter.PdfRendererError):
        run(pages, extract_images=True)
    assert all(bitmap.closed for bitmap in backend[1])
    assert adapter._CURRENT_BATCH.get() is None


def test_native_failure_is_not_an_error_row_or_cpu_fallback(
    pages, backend, monkeypatch
):
    def fail(*args):
        raise RuntimeError("native render failed")

    monkeypatch.setattr(sys.modules["cuda_pdf_renderer"], "render_batch", fail)
    with pytest.raises(adapter.PdfRendererError, match="CPU fallback is disabled"):
        extract.PDFExtractionCPUActor(extract_page_as_image=True, pdf_renderer="cuda")(
            pages
        )


def test_incomplete_batch_closes_returned_bitmaps(pages, backend, monkeypatch):
    original = sys.modules["cuda_pdf_renderer"].render_batch
    monkeypatch.setattr(
        sys.modules["cuda_pdf_renderer"],
        "render_batch",
        lambda *args: original(*args)[:-1],
    )
    with pytest.raises(adapter.PdfRendererError, match="incomplete"):
        run(pages)


def test_text_only_does_not_load_external_backend(pages, monkeypatch):
    monkeypatch.setitem(sys.modules, "cuda_pdf_renderer", None)
    result = extract.pdf_extraction(
        pages, extract_text=True, method="pdfium", pdf_renderer="cuda"
    )
    assert len(result) == 1
    with pytest.raises(adapter.PdfRendererError, match="separately built"):
        run(pages)


@pytest.mark.parametrize(
    "command", [["ingest"], ["ingest", "local"], ["ingest", "batch"]]
)
def test_cli_renderer_reaches_ingest_plan(command, tmp_path, monkeypatch):
    import importlib

    from nemo_retriever.cli.ingest import graph_commands
    from nemo_retriever.ingest.plan import (
        IngestPlanRequest,
        IngestSourceOptions,
        resolve_ingest_plan,
    )
    from typer.testing import CliRunner

    main = importlib.import_module("nemo_retriever.cli.main")
    document = tmp_path / "one.pdf"
    document.write_bytes(b"pdf")
    captured = []

    def capture(values, *, run_mode):
        options = graph_commands._build_extract_options(
            values,
            batch=graph_commands._build_extract_batch_options(
                values, enabled=run_mode == "batch"
            ),
        )
        plan = resolve_ingest_plan(
            IngestPlanRequest(
                source=IngestSourceOptions(documents=[str(document)]), extract=options
            )
        )
        captured.append(plan.extract_params.pdf_renderer)

    monkeypatch.setattr(
        graph_commands, "_run_graph_ingest_from_parsed_options", capture
    )
    result = CliRunner().invoke(
        main.app, [*command, str(document), "--pdf-renderer", "cuda"]
    )
    assert result.exit_code == 0, result.output
    assert captured == ["cuda"]
    result = CliRunner().invoke(
        main.app, [*command, str(document), "--pdf-renderer", "unknown"]
    )
    assert result.exit_code != 0
