# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte parity with the original render and detector channel round-trips."""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from nemo_retriever import create_ingestor
from nemo_retriever.common.modality.page_elements import shared
from nemo_retriever.common.modality.page_postprocessing import prepare_detector_input
from nemo_retriever.graph.graph_pipeline_registry import get_node_kwargs
from nemo_retriever.graph.ingestor_runtime import build_graph
from nemo_retriever.operators.extract.pdf import extract
from nemo_retriever.operators.extract.pdf.split import _split_pdf_to_single_page_bytes


def _legacy_encode(bitmap, image_format):
    arr = extract.convert_bitmap_to_corrected_numpy(bitmap)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if extract.cv2 is not None:
        cv2 = extract.cv2
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if image_format == "jpeg":
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        assert ok
        payload = buf.tobytes()
    else:
        image = Image.fromarray(arr)
        buf = BytesIO()
        if image_format == "jpeg":
            image.convert("RGB").save(buf, format="JPEG", quality=100)
        else:
            image.save(buf, format="PNG", compress_level=3)
        payload = buf.getvalue()
    return base64.b64encode(payload).decode("ascii")


def _legacy_model_input(payload):
    with Image.open(BytesIO(base64.b64decode(payload))) as image:
        arr = np.array(image.convert("RGB"))
    bgr = arr[:, :, ::-1].copy()
    rgb = bgr[:, :, ::-1].copy()
    return torch.from_numpy(prepare_detector_input(rgb, (1024, 1024)))


@pytest.mark.parametrize("pdf_name", ["test-page-form.pdf", "woods_frost.pdf", "multimodal_test.pdf"])
@pytest.mark.parametrize("image_format", ["jpeg", "png"])
@pytest.mark.parametrize("use_cv2", [True, False])
def test_render_and_model_input_are_bit_identical(pdf_name, image_format, use_cv2, monkeypatch):
    if not use_cv2:
        monkeypatch.setattr(extract, "cv2", None)
    path = Path(__file__).resolve().parents[2] / "data" / pdf_name
    ingestor = create_ingestor(run_mode="batch").files([str(path)]).extract(image_format=image_format).embed()
    effective = ingestor._resolve_effective_extraction_inputs()
    graph = build_graph(
        extraction_mode=effective.extraction_mode,
        extract_params=effective.extract_params,
        embed_params=ingestor._embed_params,
    )
    actor = extract.PDFExtractionCPUActor(**get_node_kwargs(graph, "PDFExtractionActor"))
    assert actor.extract_kwargs["extract_images"] is False
    pages = _split_pdf_to_single_page_bytes(path.read_bytes())
    result = actor(pd.DataFrame({"bytes": pages}))
    assert len(result) == len(pages)
    expected = []
    for payload in pages:
        document = pdfium.PdfDocument(payload)
        page = document[0]
        bitmap = page.render(scale=extract._compute_fit_to_model_scale(page, (1024, 1024), max_dpi=200))
        try:
            if use_cv2:
                assert bitmap.mode == "BGR"
            expected.append(_legacy_encode(bitmap, image_format))
        finally:
            bitmap.close()
            page.close()
            document.close()
    assert all(metadata["error"] is None for metadata in result["metadata"])
    assert [image["image_b64"] for image in result["page_image"]] == expected
    captured = []

    class Model:
        input_shape = (1024, 1024)

        def __call__(self, batch, shapes):
            captured.append(batch.clone())
            return [{} for _ in range(len(batch))]

        def postprocess(self, predictions):
            return ([torch.empty((0, 4)) for _ in predictions], [[] for _ in predictions], [[] for _ in predictions])

    detections = shared.detect_page_elements_v3(result, model=Model())
    assert all(payload["error"] is None for payload in detections["page_elements_v3"])
    assert len(captured) == 1
    expected_tensor = torch.stack([_legacy_model_input(payload) for payload in expected])
    assert captured[0].shape == expected_tensor.shape
    assert captured[0].dtype == expected_tensor.dtype
    assert captured[0].numpy().tobytes() == expected_tensor.numpy().tobytes()
