# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-native mixed file extraction for Ray Data batches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from nemo_retriever.audio import ASRActor
from nemo_retriever.audio import MediaChunkActor
from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.html.ray_data import HtmlSplitActor
from nemo_retriever.image.ray_data import ImageLoadActor
from nemo_retriever.image.load import SUPPORTED_IMAGE_EXTENSIONS
from nemo_retriever.ocr.ocr import NemotronParseActor, OCRActor
from nemo_retriever.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.params import ASRParams
from nemo_retriever.params import AudioChunkParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import HtmlChunkParams
from nemo_retriever.params import PdfSplitParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.pdf.extract import PDFExtractionActor
from nemo_retriever.pdf.split import PDFSplitActor
from nemo_retriever.table.table_detection import TableStructureActor
from nemo_retriever.chart.chart_detection import GraphicElementsActor
from nemo_retriever.txt.ray_data import TxtSplitActor
from nemo_retriever.utils.convert.to_pdf import DocToPdfConversionActor


# Define file type mappings
PDF_EXTENSIONS = {".pdf", ".docx", ".pptx"}
TEXT_EXTENSIONS = {".txt"}
HTML_EXTENSIONS = {".html"}
AUDIO_EXTENSIONS = {".mp3", ".wav"}
IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS
VIDEO_EXTENSIONS = {".mp4"}


class MultiTypeExtractOperator(AbstractOperator, GPUOperator):
    """Extract mixed or single-type Ray batches without recursing into the ingestor API."""

    def __init__(
        self,
        extraction_mode: str = "auto",
        extract_params: ExtractParams | None = None,
        text_params: TextChunkParams | None = None,
        html_params: HtmlChunkParams | None = None,
        audio_chunk_params: AudioChunkParams | None = None,
        asr_params: ASRParams | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            extraction_mode=extraction_mode,
            extract_params=extract_params,
            text_params=text_params,
            html_params=html_params,
            audio_chunk_params=audio_chunk_params,
            asr_params=asr_params,
            **kwargs,
        )
        self.extraction_mode = extraction_mode
        self.extract_params = extract_params or ExtractParams()
        self.text_params = text_params or TextChunkParams()
        self.html_params = html_params or HtmlChunkParams()
        self.audio_chunk_params = audio_chunk_params or AudioChunkParams()
        self.asr_params = asr_params or ASRParams()

    def preprocess(self, data: Any, **kwargs: Any) -> pd.DataFrame | dict[str, list[str]]:
        if isinstance(data, pd.DataFrame):
            return data
        return self._group_file_inputs(data)

    def process(self, batch_df: Any, **kwargs: Any) -> pd.DataFrame | list[Any]:
        if isinstance(batch_df, dict):
            if not any(batch_df.values()):
                return []
            raise ValueError("MultiTypeExtractOperator.process expects a pandas DataFrame for non-empty grouped inputs")

        if not isinstance(batch_df, pd.DataFrame) or batch_df.empty:
            return pd.DataFrame()

        grouped = self._group_batches(batch_df)
        outputs: list[pd.DataFrame] = []

        if not grouped["pdf"].empty:
            outputs.append(self._run_pdf_pipeline(grouped["pdf"]))
        if not grouped["image"].empty:
            outputs.append(self._run_image_pipeline(grouped["image"]))
        if not grouped["text"].empty:
            outputs.append(TxtSplitActor(params=self.text_params).run(grouped["text"]))
        if not grouped["html"].empty:
            outputs.append(HtmlSplitActor(params=self.html_params).run(grouped["html"]))
        if not grouped["audio"].empty:
            audio_df = MediaChunkActor(params=self.audio_chunk_params).run(grouped["audio"])
            outputs.append(ASRActor(params=self.asr_params).run(audio_df))
        if not grouped["video"].empty:
            video_df = MediaChunkActor(params=self.audio_chunk_params).run(grouped["video"])
            outputs.append(ASRActor(params=self.asr_params).run(video_df))

        non_empty = [df for df in outputs if isinstance(df, pd.DataFrame) and not df.empty]
        if not non_empty:
            return pd.DataFrame()
        return pd.concat(non_empty, ignore_index=True, sort=False)

    def _group_batches(self, batch_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        grouped: dict[str, list[int]] = {"pdf": [], "image": [], "text": [], "html": [], "audio": [], "video": []}
        explicit_mode = self.extraction_mode

        for idx, row in batch_df.iterrows():
            path = str(row.get("path") or "")
            ext = Path(path).suffix.lower()
            target = explicit_mode if explicit_mode != "auto" else self._mode_for_extension(ext)
            if target in grouped:
                grouped[target].append(idx)

        return {
            key: batch_df.loc[indexes].reset_index(drop=True) if indexes else pd.DataFrame(columns=batch_df.columns)
            for key, indexes in grouped.items()
        }

    def _mode_for_extension(self, ext: str) -> str:
        if ext in PDF_EXTENSIONS:
            return "pdf"
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in TEXT_EXTENSIONS:
            return "text"
        if ext in HTML_EXTENSIONS:
            return "html"
        if ext in AUDIO_EXTENSIONS:
            return "audio"
        if ext in VIDEO_EXTENSIONS:
            return "video"
        return ""

    def _group_file_inputs(self, data: Any) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {"pdf": [], "image": [], "text": [], "html": [], "audio": [], "video": []}

        if isinstance(data, (str, os.PathLike)):
            path = Path(data)
            if path.is_dir():
                files = [str(candidate) for candidate in path.rglob("*") if candidate.is_file()]
            else:
                files = [str(path)]
        elif isinstance(data, (list, tuple)):
            files = [str(item) for item in data]
        else:
            raise ValueError("MultiTypeExtractOperator expects a pandas DataFrame Ray batch")

        explicit_mode = self.extraction_mode
        for path in files:
            ext = Path(path).suffix.lower()
            target = explicit_mode if explicit_mode != "auto" else self._mode_for_extension(ext)
            if target in grouped:
                grouped[target].append(path)

        return grouped

    def _run_pdf_pipeline(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        extract_params = self.extract_params
        split_actor = PDFSplitActor(
            split_params=PdfSplitParams(
                start_page=extract_params.start_page,
                end_page=extract_params.end_page,
            )
        )
        batch_df = DocToPdfConversionActor().run(batch_df)
        batch_df = split_actor.run(batch_df)

        tuning = getattr(extract_params, "batch_tuning", None)
        parse_mode = extract_params.method == "nemotron_parse" or (
            tuning is not None
            and all(
                getattr(tuning, name, None)
                for name in ("nemotron_parse_workers", "gpu_nemotron_parse", "nemotron_parse_batch_size")
            )
        )

        if parse_mode:
            parse_kwargs: dict[str, Any] = {
                "extract_text": extract_params.extract_text,
                "extract_tables": extract_params.extract_tables,
                "extract_charts": extract_params.extract_charts,
                "extract_infographics": extract_params.extract_infographics,
            }
            if extract_params.api_key:
                parse_kwargs["api_key"] = extract_params.api_key
            return NemotronParseActor(**parse_kwargs).run(batch_df)

        extract_kwargs: dict[str, Any] = {
            "method": extract_params.method,
            "dpi": int(extract_params.dpi),
            "extract_text": extract_params.extract_text,
            "extract_tables": extract_params.extract_tables,
            "extract_charts": extract_params.extract_charts,
            "extract_infographics": extract_params.extract_infographics,
            "extract_page_as_image": extract_params.extract_page_as_image,
            "api_key": extract_params.api_key,
        }
        batch_df = PDFExtractionActor(**extract_kwargs).run(batch_df)
        return self._run_detection_pipeline(batch_df)

    def _run_image_pipeline(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        batch_df = ImageLoadActor().run(batch_df)
        return self._run_detection_pipeline(batch_df)

    def _run_detection_pipeline(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        extract_params = self.extract_params
        tuning = getattr(extract_params, "batch_tuning", None)

        detect_kwargs: dict[str, Any] = {}
        if extract_params.page_elements_invoke_url:
            detect_kwargs["invoke_url"] = extract_params.page_elements_invoke_url
        if extract_params.api_key:
            detect_kwargs["api_key"] = extract_params.api_key
        inference_batch_size = getattr(extract_params, "inference_batch_size", None) or getattr(
            tuning, "page_elements_batch_size", None
        )
        if inference_batch_size:
            detect_kwargs["inference_batch_size"] = int(inference_batch_size)
        batch_df = PageElementDetectionActor(**detect_kwargs).run(batch_df)

        if extract_params.use_table_structure and extract_params.extract_tables:
            table_kwargs: dict[str, Any] = {}
            if extract_params.table_structure_invoke_url:
                table_kwargs["table_structure_invoke_url"] = extract_params.table_structure_invoke_url
            if extract_params.ocr_invoke_url:
                table_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
            if extract_params.api_key:
                table_kwargs["api_key"] = extract_params.api_key
            if extract_params.table_output_format:
                table_kwargs["table_output_format"] = extract_params.table_output_format
            batch_df = TableStructureActor(**table_kwargs).run(batch_df)

        if extract_params.use_graphic_elements and extract_params.extract_charts:
            graphic_kwargs: dict[str, Any] = {}
            if extract_params.graphic_elements_invoke_url:
                graphic_kwargs["graphic_elements_invoke_url"] = extract_params.graphic_elements_invoke_url
            if extract_params.ocr_invoke_url:
                graphic_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
            if extract_params.api_key:
                graphic_kwargs["api_key"] = extract_params.api_key
            batch_df = GraphicElementsActor(**graphic_kwargs).run(batch_df)

        ocr_kwargs: dict[str, Any] = {"use_graphic_elements": extract_params.use_graphic_elements}
        if extract_params.method in ("pdfium_hybrid", "ocr") and extract_params.extract_text:
            ocr_kwargs["extract_text"] = True
        if extract_params.extract_tables and not extract_params.use_table_structure:
            ocr_kwargs["extract_tables"] = True
        if extract_params.extract_charts and not extract_params.use_graphic_elements:
            ocr_kwargs["extract_charts"] = True
        if extract_params.extract_infographics:
            ocr_kwargs["extract_infographics"] = True
        if extract_params.ocr_invoke_url:
            ocr_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
        if extract_params.api_key:
            ocr_kwargs["api_key"] = extract_params.api_key
        if inference_batch_size:
            ocr_kwargs["inference_batch_size"] = int(inference_batch_size)

        if any(
            ocr_kwargs.get(key) for key in ("extract_text", "extract_tables", "extract_charts", "extract_infographics")
        ):
            batch_df = OCRActor(**ocr_kwargs).run(batch_df)

        return batch_df

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
