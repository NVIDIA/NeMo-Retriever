# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MultiTypeExtractOperator: Handles mixed file types in a folder.

This operator takes a folder path, groups files by type, applies the appropriate
extract method for each type, and collates the results into a single output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever import create_ingestor
from nemo_retriever.image.load import SUPPORTED_IMAGE_EXTENSIONS


# Define file type mappings
PDF_EXTENSIONS = {".pdf", ".docx", ".pptx"}
TEXT_EXTENSIONS = {".txt"}
HTML_EXTENSIONS = {".html"}
AUDIO_EXTENSIONS = {".mp3", ".wav"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}  # Assuming based on README mention
IMAGE_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS


class MultiTypeExtractOperator(AbstractOperator):
    """Operator to extract content from mixed file types in a folder."""

    def __init__(self, extract_params: Dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.extract_params = extract_params or {}

        # Separate params by type
        self.pdf_params = {
            k: v
            for k, v in self.extract_params.items()
            if k
            in [
                "extract_text",
                "extract_tables",
                "extract_charts",
                "extract_infographics",
                "invoke_url",
                "page_elements_invoke_url",
                "ocr_invoke_url",
                "api_key",
            ]
        }
        self.image_params = self.pdf_params.copy()  # Same as PDF for now
        self.text_params = {k: v for k, v in self.extract_params.items() if k in ["max_tokens", "overlap_tokens"]}
        self.html_params = self.text_params.copy()
        self.audio_params = {
            k: v for k, v in self.extract_params.items() if k in ["audio_chunk_duration", "asr_params"]
        }
        self.video_params = self.audio_params.copy()  # Placeholder

    def preprocess(self, data: Any, **kwargs: Any) -> Dict[str, List[str]]:
        """Group files by type. Input: folder path (str) or list of paths."""
        if isinstance(data, str):
            folder_path = Path(data)
            if folder_path.is_dir():
                files = [str(p) for p in folder_path.rglob("*") if p.is_file()]
            else:
                files = [data]
        elif isinstance(data, list):
            files = data
        else:
            raise ValueError("Input must be folder path (str) or list of file paths")

        grouped: Dict[str, List[str]] = {
            "pdf": [],
            "image": [],
            "text": [],
            "html": [],
            "audio": [],
            "video": [],
        }

        for file in files:
            ext = Path(file).suffix.lower()
            if ext in PDF_EXTENSIONS:
                grouped["pdf"].append(file)
            elif ext in IMAGE_EXTENSIONS:
                grouped["image"].append(file)
            elif ext in TEXT_EXTENSIONS:
                grouped["text"].append(file)
            elif ext in HTML_EXTENSIONS:
                grouped["html"].append(file)
            elif ext in AUDIO_EXTENSIONS:
                grouped["audio"].append(file)
            elif ext in VIDEO_EXTENSIONS:
                grouped["video"].append(file)
            # Ignore unknown types

        return grouped

    def process(self, grouped_files: Dict[str, List[str]], **kwargs: Any) -> Any:
        """Process each file type group and collate results."""
        all_datasets = []

        # PDFs
        if grouped_files["pdf"]:
            ingestor = create_ingestor(run_mode="batch")
            ingestor = ingestor.files(grouped_files["pdf"]).extract(**self.pdf_params)
            ingestor.ingest()  # Execute the pipeline
            ds = ingestor.get_dataset()
            if ds:
                all_datasets.append(ds)

        # Images
        if grouped_files["image"]:
            ingestor = create_ingestor(run_mode="batch")
            ingestor = ingestor.files(grouped_files["image"]).extract_image_files(**self.image_params)
            ingestor.ingest()
            ds = ingestor.get_dataset()
            if ds:
                all_datasets.append(ds)

        # Text
        if grouped_files["text"]:
            ingestor = create_ingestor(run_mode="batch")
            ingestor = ingestor.files(grouped_files["text"]).extract_txt(**self.text_params)
            ingestor.ingest()
            ds = ingestor.get_dataset()
            if ds:
                all_datasets.append(ds)

        # HTML
        if grouped_files["html"]:
            ingestor = create_ingestor(run_mode="batch")
            ingestor = ingestor.files(grouped_files["html"]).extract_html(**self.html_params)
            ingestor.ingest()
            ds = ingestor.get_dataset()
            if ds:
                all_datasets.append(ds)

        # Audio
        if grouped_files["audio"]:
            ingestor = create_ingestor(run_mode="batch")
            ingestor = ingestor.files(grouped_files["audio"]).extract_audio(**self.audio_params)
            ingestor.ingest()
            ds = ingestor.get_dataset()
            if ds:
                all_datasets.append(ds)

        # Video (assuming similar to audio for now)
        if grouped_files["video"]:
            ingestor = create_ingestor(run_mode="batch")
            ingestor = ingestor.files(grouped_files["video"]).extract_audio(**self.video_params)  # Placeholder
            ingestor.ingest()
            ds = ingestor.get_dataset()
            if ds:
                all_datasets.append(ds)

        # Union all datasets
        if all_datasets:
            combined_rows = []
            for ds in all_datasets:
                combined_rows.extend(ds.take_all())
            return combined_rows
        else:
            return []

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        """Optional postprocessing."""
        return data
