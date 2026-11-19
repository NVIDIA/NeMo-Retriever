# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Optional batched CUDA rasterization without replacing PDFium's Python APIs."""

from __future__ import annotations

import collections
import contextvars
import functools
import importlib
import inspect
from types import SimpleNamespace
from typing import Any


class PdfRendererError(RuntimeError):
    """An explicitly selected PDF renderer failed; CPU fallback is prohibited."""


_CURRENT_BATCH: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "cuda_pdf_batch", default=None
)


def render_bitmap(page: Any, *, scale: float, rotation: int = 0) -> Any:
    """Render through the active CUDA batch, or the unchanged default PDFium API."""
    batch = _CURRENT_BATCH.get()
    if batch is None:
        return (
            page.render(scale=scale, rotation=rotation)
            if rotation
            else page.render(scale=scale)
        )
    try:
        return batch.take(float(scale), rotation)
    except Exception as exc:
        batch.failure = exc
        raise PdfRendererError(
            "CUDA PDF rendering failed; CPU fallback is disabled"
        ) from exc


class _SceneBitmap:
    def __init__(self, scene):
        self.scene = scene

    def close(self):
        pass

    def to_numpy(self):
        raise PdfRendererError(
            "GPU-resident scenes cannot be materialized as host bitmaps"
        )


class _CudaBatch:
    def __init__(self, data: Any, config: dict[str, Any], module: Any) -> None:
        self.bitmaps: list[Any] = []
        self.cache: dict[tuple[bytes, float, int], Any] = collections.defaultdict(
            collections.deque
        )
        self.packet: bytes | None = None
        self.failure: Exception | None = None
        try:
            backend = importlib.import_module("cuda_pdf_renderer")
        except ImportError as exc:
            raise PdfRendererError(
                "CUDA PDF rendering requires the separately built cuda-pdf-renderer package; encoded cuda mode also requires its renderer service"
            ) from exc
        packets: list[bytes] = []
        scales: list[float] = []
        rotations: list[int] = []
        for value in data["bytes"].tolist():
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise PdfRendererError(
                    "CUDA PDF rendering requires single-page PDF byte packets"
                )
            packet = bytes(value)
            document = module.pdfium.PdfDocument(packet)
            try:
                if len(document) != 1:
                    raise PdfRendererError(
                        "CUDA PDF rendering requires single-page PDF byte packets"
                    )
                if config["extract_images"] and config["pdf_renderer"] != "cuda_rdt":
                    page = document.get_page(0)
                    try:
                        width, height = page.get_size()
                        rotation = page.get_rotation()
                    finally:
                        page.close()
                else:
                    width, height = document.get_page_size(0)
                    rotation = 0
            finally:
                document.close()
            dpi = config["dpi"]
            if config["render_mode"] == "fit_to_model":
                dimensions = SimpleNamespace(
                    get_width=lambda value=width: value,
                    get_height=lambda value=height: value,
                )
                scale = module._compute_fit_to_model_scale(
                    dimensions, module._MODEL_INPUT_SIZE, max_dpi=dpi
                )
            else:
                scale = max(float(dpi) / 72.0, 0.01)
            packets.append(packet)
            scales.append(scale)
            rotations.append(0)
            if config["extract_images"] and config["pdf_renderer"] != "cuda_rdt":
                packets.append(packet)
                scales.append(1.0)
                rotations.append(rotation)
        try:
            if config["pdf_renderer"] == "cuda_rdt":
                from cuda_pdf_renderer.device import SceneRecorder

                recorder = SceneRecorder()
                self.bitmaps = [
                    _SceneBitmap(recorder.record(packet, scale, rotation))
                    for packet, scale, rotation in zip(packets, scales, rotations)
                ]
            else:
                self.bitmaps = list(backend.render_batch(packets, scales, rotations))
            if len(self.bitmaps) != len(packets):
                raise PdfRendererError(
                    "CUDA PDF renderer returned an incomplete page batch"
                )
            for packet, scale, rotation, bitmap in zip(
                packets, scales, rotations, self.bitmaps
            ):
                self.cache[packet, scale, rotation].append(bitmap)
        except Exception:
            self.close()
            raise

    def take(self, scale: float, rotation: int) -> Any:
        values = self.cache.get((self.packet, scale, rotation))
        if not values:
            raise PdfRendererError(
                "CUDA PDF render request did not match its prepared batch"
            )
        return values.popleft()

    def close(self) -> None:
        for bitmap in self.bitmaps:
            bitmap.close()
        self.bitmaps.clear()


def select_packet(packet: bytes | bytearray | memoryview) -> None:
    """Associate subsequent render callbacks with the current input row."""
    batch = _CURRENT_BATCH.get()
    if batch is not None:
        batch.packet = bytes(packet)


def with_cuda_rendering(function: Any) -> Any:
    """Keep the default path unchanged and prepare CUDA pages once per NRL batch."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapped(data: Any, *args: Any, **kwargs: Any) -> Any:
        renderer = kwargs.get("pdf_renderer", "pdfium")
        if renderer == "pdfium":
            return function(data, *args, **kwargs)
        if renderer not in ("cuda", "cuda_rdt"):
            raise ValueError(f"Unsupported PDF renderer: {renderer!r}")
        bound = signature.bind(data, *args, **kwargs)
        bound.apply_defaults()
        config = bound.arguments
        method = config["kwargs"].get("method") or config["text_extraction_method"]
        wants_raster = bool(
            config["extract_images"]
            or config["extract_page_as_image"]
            or config["extract_tables"]
            or config["extract_charts"]
            or config["extract_infographics"]
            or (config["extract_text"] and method in ("pdfium_hybrid", "ocr"))
        )
        if not wants_raster or not len(data.index):
            return function(data, *args, **kwargs)
        # The external native ABI accepts at most 256 render requests. Image
        # extraction needs a second scale/rotation request for each source page.
        page_limit = 128 if config["extract_images"] else 256
        if len(data.index) > page_limit:
            import pandas as pd

            return pd.concat(
                [
                    wrapped(data.iloc[offset : offset + page_limit], *args, **kwargs)
                    for offset in range(0, len(data.index), page_limit)
                ],
                ignore_index=True,
            )
        if _CURRENT_BATCH.get() is not None:
            raise PdfRendererError("Nested CUDA PDF rendering batches are unsupported")
        import sys

        batch = None
        token = None
        try:
            batch = _CudaBatch(data, config, sys.modules[function.__module__])
            token = _CURRENT_BATCH.set(batch)
            result = function(data, *args, **kwargs)
            if batch.failure is not None:
                raise PdfRendererError(
                    "CUDA PDF image extraction failed"
                ) from batch.failure
            return result
        except PdfRendererError:
            raise
        except Exception as exc:
            raise PdfRendererError(
                "CUDA PDF rendering failed; CPU fallback is disabled"
            ) from exc
        finally:
            if token is not None:
                _CURRENT_BATCH.reset(token)
            if batch is not None:
                batch.close()

    return wrapped
