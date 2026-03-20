# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union

import base64
import io
import os
from pathlib import Path  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F
from nemo_retriever.utils.hf_cache import configure_global_hf_cache_base
from torch import amp
from torchvision.transforms.functional import convert_image_dtype
from ..model import BaseModel, RunMode

from PIL import Image

# Max images per GPU batch (matches typical nemotron-ocr inference limits).
_OCR_MAX_GPU_BATCH = 32


class NemotronOCRV1(BaseModel):
    """
    Nemotron OCR v1 model for optical character recognition.

    End-to-end OCR model that integrates:
    - Text detector for region localization
    - Text recognizer for transcription
    - Relational model for layout and reading order analysis
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        configure_global_hf_cache_base()
        from nemotron_ocr.inference.pipeline import NemotronOCR  # local-only import

        if model_dir:
            self._model = NemotronOCR(model_dir=model_dir)
        else:
            self._model = NemotronOCR()
        # NemotronOCR is a high-level pipeline (not an nn.Module). We can optionally
        # TensorRT-compile individual submodules (e.g. the detector backbone) but
        # must keep post-processing (NMS, box decoding, etc.) in eager PyTorch/C++.
        self._enable_trt = os.getenv("RETRIEVER_ENABLE_TORCH_TRT", "").strip().lower() in {"1", "true", "yes", "on"}
        if self._enable_trt and self._model is not None:
            self._maybe_compile_submodules()

    def _maybe_compile_submodules(self) -> None:
        """
        Best-effort TensorRT compilation of internal nn.Modules.
        Any failure falls back to eager PyTorch without breaking initialization.
        """
        try:
            import torch_tensorrt  # type: ignore
        except Exception:
            return

        # Detector is the safest candidate: input is a BCHW image tensor.
        if self._model is None:
            return

        detector = getattr(self._model, "detector", None)
        if not isinstance(detector, torch.nn.Module):
            return

        # NemotronOCR internally resizes/pads to 1024 and runs B=1 (see upstream FIXME);
        # keep the TRT input shape fixed to avoid accidental batching issues.
        try:
            trt_input = torch_tensorrt.Input((1, 3, 1024, 1024), dtype=torch.float16)
        except TypeError:
            # Older/newer API variants: fall back to named arg.
            trt_input = torch_tensorrt.Input(shape=(1, 3, 1024, 1024), dtype=torch.float16)

        # If any torchvision NMS makes it into a compiled graph elsewhere, forcing
        # that op to run in Torch avoids hard failures.
        compile_kwargs: Dict[str, Any] = {
            "inputs": [trt_input],
            "enabled_precisions": {torch.float16},
        }
        if hasattr(torch_tensorrt, "compile"):
            for k in ("torch_executed_ops", "torch_executed_modules"):
                if k == "torch_executed_ops":
                    compile_kwargs[k] = {"torchvision::nms"}
                elif k == "torch_executed_modules":
                    compile_kwargs[k] = set()
            try:
                self._model.detector = torch_tensorrt.compile(detector, **compile_kwargs)
            except Exception:
                # Leave detector as-is on any failure.
                return

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess the input tensor."""
        # no-op for now
        return tensor

    @staticmethod
    def _tensor_to_png_b64(img: torch.Tensor) -> str:
        """
        Convert a CHW/BCHW tensor into a base64-encoded PNG.

        Accepts:
          - CHW (3,H,W) or (1,H,W)
        Returns:
          - base64 string (no data: prefix)
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(img)}")
        if img.ndim != 3:
            raise ValueError(f"Expected CHW tensor, got shape {tuple(img.shape)}")

        x = img.detach()
        if x.device.type != "cpu":
            x = x.cpu()

        # Convert to uint8 in [0,255]
        if x.dtype.is_floating_point:
            maxv = float(x.max().item()) if x.numel() else 1.0
            # Heuristic: treat [0,1] images as normalized.
            if maxv <= 1.5:
                x = x * 255.0
            x = x.clamp(0, 255).to(dtype=torch.uint8)
        else:
            x = x.clamp(0, 255).to(dtype=torch.uint8)

        c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])  # noqa: F841
        if c == 1:
            arr = x.squeeze(0).numpy()
            pil = Image.fromarray(arr, mode="L").convert("RGB")
        elif c == 3:
            arr = x.permute(1, 2, 0).contiguous().numpy()
            pil = Image.fromarray(arr, mode="RGB")
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {c}")

        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _torch_chw_to_float16_cpu(img: torch.Tensor) -> torch.Tensor:
        """RGB CHW tensor -> float16 CHW on CPU in [0, 1], as in ``NemotronOCR._load_image_to_tensor``."""
        if not isinstance(img, torch.Tensor) or img.ndim != 3:
            raise ValueError(f"Expected CHW torch.Tensor, got shape {getattr(img, 'shape', None)}")
        x = img.detach().cpu()
        if x.dtype.is_floating_point:
            maxv = float(x.max().item()) if x.numel() else 1.0
            if maxv <= 1.5:
                x = x * 255.0
            x = x.clamp(0, 255).to(dtype=torch.uint8)
        else:
            x = x.clamp(0, 255).to(dtype=torch.uint8)
        c = int(x.shape[0])
        if c == 1:
            x = x.repeat(3, 1, 1)
        elif c != 3:
            raise ValueError(f"Expected 1 or 3 channels, got {c}")
        return convert_image_dtype(x, dtype=torch.float16)

    @staticmethod
    def _numpy_hwc_to_chw_f16(image: np.ndarray) -> torch.Tensor:
        """HWC ndarray -> float16 CHW on CPU, matching ``NemotronOCR._load_image_to_tensor``."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[2] == 4:
            image = image[..., :3]
        img_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        return convert_image_dtype(img_tensor, dtype=torch.float16)

    def _batch_process_chw(
        self,
        images_chw: List[torch.Tensor],
        merge_level: str,
    ) -> List[List[Dict[str, Any]]]:
        """
        Run detector → NMS → recognizer → relational for a batch of RGB CHW tensors.

        Uses the same building blocks as ``nemotron_ocr.inference.pipeline.NemotronOCR._process_tensor``,
        with a shared square side ``M = max_i max(H_i, W_i)`` so all images stack for one detector call.
        Quad scaling uses ``M / INFER_LENGTH`` so coordinates match the resized ``M×M`` canvas.
        """
        from nemotron_ocr.inference.pre_processing import interpolate_and_pad, pad_to_square
        from nemotron_ocr.inference.pipeline import (
            DETECTOR_DOWNSAMPLE,
            INFER_LENGTH,
            MERGE_LEVELS,
            NMS_IOU_THRESHOLD,
            NMS_MAX_REGIONS,
            NMS_PROB_THRESHOLD,
            PAD_COLOR,
        )
        from nemotron_ocr.inference.post_processing.data.text_region import TextBlock
        from nemotron_ocr.inference.post_processing.research_ops import (
            parse_relational_results,
            reorder_boxes,
        )
        from nemotron_ocr_cpp import quad_non_maximal_suppression, region_counts_to_indices, rrect_to_quads

        mdl = self._model
        if mdl is None:
            raise RuntimeError("Local OCR model was not initialized.")

        bsz = len(images_chw)
        if bsz == 0:
            return []

        if merge_level not in MERGE_LEVELS:
            raise ValueError(f"Invalid merge level: {merge_level}. Must be one of {MERGE_LEVELS}.")

        original_shapes: List[Tuple[int, int]] = []
        for t in images_chw:
            _, h, w = t.shape
            original_shapes.append((int(h), int(w)))

        m_side = max(max(h, w) for h, w in original_shapes)

        square_rows: List[torch.Tensor] = []
        for t in images_chw:
            sq = pad_to_square(t, m_side, how="bottom_right").unsqueeze(0)
            square_rows.append(sq)
        batch_square = torch.cat(square_rows, dim=0)

        pad_color = PAD_COLOR.to(device=batch_square.device, dtype=batch_square.dtype)
        padded_image = interpolate_and_pad(batch_square, pad_color, INFER_LENGTH)

        with amp.autocast("cuda", enabled=True), torch.no_grad():
            det_conf, _, det_rboxes, det_feature_3 = mdl.detector(padded_image.cuda())

        with amp.autocast("cuda", enabled=True), torch.no_grad():
            e2e_det_conf = torch.sigmoid(det_conf)
            e2e_det_coords = rrect_to_quads(det_rboxes.float(), DETECTOR_DOWNSAMPLE)

            quads, confidence, region_counts = quad_non_maximal_suppression(
                e2e_det_coords,
                e2e_det_conf,
                prob_threshold=NMS_PROB_THRESHOLD,
                iou_threshold=NMS_IOU_THRESHOLD,
                kernel_height=2,
                kernel_width=3,
                max_regions=NMS_MAX_REGIONS,
                verbose=False,
            )[:3]

        region_counts = region_counts.reshape(-1).to(dtype=torch.int64)
        predictions_per_image: List[List[Dict[str, Any]]] = [[] for _ in range(bsz)]

        if quads.shape[0] == 0 or int(region_counts.sum().item()) == 0:
            return predictions_per_image

        rec_rectified_quads = mdl.recognizer_quad_rectifier(
            quads.detach(), padded_image.shape[2], padded_image.shape[3]
        )
        rel_rectified_quads = mdl.relational_quad_rectifier(
            quads.cuda().detach(), padded_image.shape[2], padded_image.shape[3]
        )

        input_indices = region_counts_to_indices(region_counts, quads.shape[0])

        rec_rectified_quads = mdl.grid_sampler(det_feature_3.float(), rec_rectified_quads.float(), input_indices)
        rel_rectified_quads = mdl.grid_sampler(
            det_feature_3.float().cuda(),
            rel_rectified_quads,
            input_indices.cuda(),
        )

        with amp.autocast("cuda", enabled=True), torch.no_grad():
            rec_output, rec_features = mdl.recognizer(rec_rectified_quads.cuda())

        rel_output = mdl.relational(
            rel_rectified_quads.cuda(),
            quads.cuda(),
            region_counts.cpu(),
            rec_features.cuda(),
        )
        words, lines, line_var = (
            rel_output["words"],
            rel_output["lines"],
            rel_output["line_log_var_unc"],
        )

        with amp.autocast("cuda", enabled=True), torch.no_grad():
            words = [F.softmax(r, dim=1, dtype=torch.float32)[:, 1:] for r in words]

            output: Dict[str, Any] = {
                "sequences": F.softmax(rec_output, dim=2, dtype=torch.float32),
                "region_counts": region_counts,
                "quads": quads,
                "raw_detector_confidence": e2e_det_conf,
                "confidence": confidence,
                "relations": words,
                "line_relations": lines,
                "line_rel_var": line_var,
                "fg_colors": None,
                "fonts": None,
                "tt_log_var_uncertainty": None,
                "e2e_recog_features": rec_features,
            }

        quads_scaled = output["quads"]
        qscale = float(m_side) / float(INFER_LENGTH)
        lengths_tensor = torch.full(
            (quads_scaled.shape[0], 1, 1),
            qscale,
            dtype=torch.float32,
            device=quads_scaled.device,
        )
        quads_scaled = quads_scaled * lengths_tensor
        output["quads"] = quads_scaled

        rec_batch = mdl.recog_encoder.convert_targets_to_labels(output, image_size=None, is_gt=False)
        relation_batch = mdl.relation_encoder.convert_targets_to_labels(output, image_size=None, is_gt=False)

        for example, rel_example in zip(rec_batch, relation_batch):
            example.relation_graph = rel_example.relation_graph
            example.prune_invalid_relations()

        for example in rec_batch:
            if example.relation_graph is None:
                continue
            for paragraph in example.relation_graph:
                block: List[Any] = []
                for line in paragraph:
                    for relational_idx in line:
                        block.append(example[relational_idx])
                if block:
                    example.blocks.append(TextBlock(block))

        for example in rec_batch:
            for text_region in example:
                text_region.region = text_region.region.vertices

        for ex_idx, example in enumerate(rec_batch):
            boxes, texts, scores = parse_relational_results(example, level=merge_level)
            boxes, texts, scores = reorder_boxes(boxes, texts, scores, mode="top_left", dbscan_eps=10)

            orig_h, orig_w = original_shapes[ex_idx]

            if len(boxes) == 0:
                boxes = ["nan"]
                texts = ["nan"]
                scores = ["nan"]
            else:
                boxes_array = np.array(boxes).reshape(-1, 4, 2)
                boxes_array[:, :, 0] = boxes_array[:, :, 0] / orig_w
                boxes_array[:, :, 1] = boxes_array[:, :, 1] / orig_h
                boxes = boxes_array.astype(np.float16).tolist()

            for box, text, conf in zip(boxes, texts, scores):
                if box == "nan":
                    break
                predictions_per_image[ex_idx].append(
                    {
                        "text": text,
                        "confidence": conf,
                        "left": min(p[0] for p in box),
                        "upper": max(p[1] for p in box),
                        "right": max(p[0] for p in box),
                        "lower": min(p[1] for p in box),
                    }
                )

        return predictions_per_image

    def _invoke_sequential(
        self,
        inputs: List[Union[torch.Tensor, np.ndarray]],
        merge_level: str,
        *,
        as_numpy: bool,
    ) -> List[Any]:
        """One pipeline call per item (used when TensorRT compilation fixes detector batch to 1)."""
        out: List[Any] = []
        for item in inputs:
            if as_numpy:
                out.append(self._model(item, merge_level=merge_level))  # type: ignore[arg-type]
            else:
                b64 = self._tensor_to_png_b64(item)  # type: ignore[arg-type]
                out.append(self._model(b64.encode("utf-8"), merge_level=merge_level))
        return out

    @staticmethod
    def _extract_text(obj: Any) -> str:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            for k in ("text", "output_text", "generated_text", "ocr_text"):
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            # Some APIs return nested structures; best-effort flatten common shapes.
            if "words" in obj and isinstance(obj["words"], list):
                parts: List[str] = []
                for w in obj["words"]:
                    if isinstance(w, dict) and isinstance(w.get("text"), str):
                        parts.append(w["text"])
                if parts:
                    return " ".join(parts).strip()
        return str(obj).strip()

    def invoke(
        self,
        input_data: Union[torch.Tensor, str, bytes, np.ndarray, io.BytesIO, List[np.ndarray]],
        merge_level: str = "paragraph",
    ) -> Any:
        """
        Invoke OCR locally.

        Supports:
          - file path (str) **only if it exists**
          - base64 (str/bytes) (str is treated as base64 unless it is an existing file path)
          - NumPy array (HWC)
          - list of NumPy arrays (HWC): batched GPU inference up to ``_OCR_MAX_GPU_BATCH`` per forward
          - io.BytesIO
          - torch.Tensor CHW: single image
          - torch.Tensor BCHW: batched inference; returns ``list[list[dict]]`` (one inner list per image)
        """
        if self._model is None:
            raise RuntimeError("Local OCR model was not initialized.")

        # Batched RGB crops (as produced by page-element OCR in ``ocr.py``).
        if isinstance(input_data, list):
            if not input_data:
                return []
            if not all(isinstance(x, np.ndarray) for x in input_data):
                raise TypeError(
                    "Batched invoke expects each list element to be a numpy.ndarray (HWC RGB). "
                    f"Got types: {[type(x).__name__ for x in input_data[:8]]}" + (" ..." if len(input_data) > 8 else "")
                )
            arrays: List[np.ndarray] = input_data  # type: ignore[assignment]
            if self._enable_trt:
                return self._invoke_sequential(arrays, merge_level, as_numpy=True)
            merged: List[List[Dict[str, Any]]] = []
            for start in range(0, len(arrays), _OCR_MAX_GPU_BATCH):
                chunk = arrays[start : start + _OCR_MAX_GPU_BATCH]
                chw = [self._numpy_hwc_to_chw_f16(a) for a in chunk]
                merged.extend(self._batch_process_chw(chw, merge_level))
            return merged

        if isinstance(input_data, torch.Tensor):
            if input_data.ndim == 4:
                n = int(input_data.shape[0])
                if self._enable_trt:
                    return self._invoke_sequential(
                        [input_data[i] for i in range(n)],
                        merge_level,
                        as_numpy=False,
                    )
                merged_t: List[List[Dict[str, Any]]] = []
                for start in range(0, n, _OCR_MAX_GPU_BATCH):
                    sl = input_data[start : start + _OCR_MAX_GPU_BATCH]
                    chw = [self._torch_chw_to_float16_cpu(sl[i]) for i in range(int(sl.shape[0]))]
                    merged_t.extend(self._batch_process_chw(chw, merge_level))
                return merged_t
            if input_data.ndim == 3:
                if self._enable_trt:
                    b64 = self._tensor_to_png_b64(input_data)
                    return self._model(b64.encode("utf-8"), merge_level=merge_level)
                single = self._batch_process_chw([self._torch_chw_to_float16_cpu(input_data)], merge_level)
                return single[0]
            raise ValueError(f"Unsupported torch tensor shape for OCR: {tuple(input_data.shape)}")

        # Disambiguate str: existing file path vs base64 string.
        if isinstance(input_data, str):
            # Treat as base64 string (nemotron_ocr expects bytes for base64).
            return self._model(input_data.encode("utf-8"), merge_level=merge_level)

        # bytes / ndarray / BytesIO are supported directly by nemotron_ocr.
        return self._model(input_data, merge_level=merge_level)

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return "Nemotron OCR v1"

    @property
    def model_type(self) -> str:
        """Model category/type."""
        return "ocr"

    @property
    def model_runmode(self) -> RunMode:
        """Execution mode: local, NIM, or build-endpoint."""
        return "local"

    @property
    def input(self) -> Any:
        """
        Input schema for the model.

        Returns:
            dict: Schema describing RGB image input with variable dimensions
        """
        return {
            "type": "image",
            "format": "RGB",
            "supported_formats": ["PNG", "JPEG"],
            "data_types": ["float32", "uint8"],
            "dimensions": "variable (H x W)",
            "batch_support": True,
            "value_range": {"float32": "[0, 1]", "uint8": "[0, 255] (auto-converted)"},
            "aggregation_levels": ["word", "sentence", "paragraph"],
            "description": "Document or scene image in RGB format with automatic multi-scale resizing",
        }

    @property
    def output(self) -> Any:
        """
        Output schema for the model.

        Returns:
            dict: Schema describing OCR output format
        """
        return {
            "type": "ocr_results",
            "format": "structured",
            "structure": {
                "boxes": "List[List[List[float]]] - quadrilateral bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]",  # noqa: E501
                "texts": "List[str] - recognized text strings",
                "confidences": "List[float] - confidence scores per detection",
            },
            "properties": {
                "reading_order": True,
                "layout_analysis": True,
                "multi_line_support": True,
                "multi_block_support": True,
            },
            "description": "Structured OCR results with bounding boxes, recognized text, and confidence scores",
        }

    @property
    def input_batch_size(self) -> int:
        """Maximum or default input batch size."""
        return _OCR_MAX_GPU_BATCH
