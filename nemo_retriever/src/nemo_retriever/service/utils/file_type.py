# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify uploaded files by filename suffix and reject unsupported types."""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

from fastapi import HTTPException, UploadFile

from nemo_retriever.service.models.base import RichModel


class FileCategory(str, Enum):
    """High-level media category for an ingested file."""

    DOCUMENT = "document"
    TEXT = "text"
    HTML = "html"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class FileClassification(RichModel):
    """Result of classifying an uploaded file."""

    filename: str
    suffix: str
    category: FileCategory
    content_type: str


class FileClassifier:
    """Identify uploaded file types from filename suffix.

    Supported extensions are derived from the pipeline's extraction operators:
      - multi_type_extract_operator.py  (PDF, text, html, image, audio, video)
      - utils/input_files.py            (CLI input type patterns)
      - audio/stage.py                  (ffmpeg-capable containers)
      - utils/convert/to_pdf.py         (office → PDF conversion)
    """

    SUFFIX_MAP: ClassVar[dict[str, tuple[FileCategory, str]]] = {
        # Document / office
        ".pdf": (FileCategory.DOCUMENT, "application/pdf"),
        ".docx": (FileCategory.DOCUMENT, "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ".pptx": (FileCategory.DOCUMENT, "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        # Plain text
        ".txt": (FileCategory.TEXT, "text/plain"),
        # Web / markup
        ".html": (FileCategory.HTML, "text/html"),
        # Image
        ".png": (FileCategory.IMAGE, "image/png"),
        ".jpg": (FileCategory.IMAGE, "image/jpeg"),
        ".jpeg": (FileCategory.IMAGE, "image/jpeg"),
        ".bmp": (FileCategory.IMAGE, "image/bmp"),
        ".tiff": (FileCategory.IMAGE, "image/tiff"),
        ".tif": (FileCategory.IMAGE, "image/tiff"),
        ".svg": (FileCategory.IMAGE, "image/svg+xml"),
        # Audio
        ".mp3": (FileCategory.AUDIO, "audio/mpeg"),
        ".wav": (FileCategory.AUDIO, "audio/wav"),
        ".m4a": (FileCategory.AUDIO, "audio/mp4"),
        # Video
        ".mp4": (FileCategory.VIDEO, "video/mp4"),
        ".mov": (FileCategory.VIDEO, "video/quicktime"),
        ".mkv": (FileCategory.VIDEO, "video/x-matroska"),
        ".avi": (FileCategory.VIDEO, "video/x-msvideo"),
    }

    @classmethod
    def supported_extensions(cls) -> frozenset[str]:
        return frozenset(cls.SUFFIX_MAP)

    @classmethod
    def classify(cls, upload: UploadFile, *, filename_override: str = "") -> FileClassification:
        """Determine file category from suffix, raising 415 for unsupported types.

        Parameters
        ----------
        upload:
            The FastAPI ``UploadFile`` from the request.
        filename_override:
            If provided, use this instead of ``upload.filename`` (useful for
            the ``/ingest/page`` endpoint where the original document filename
            is submitted separately).
        """
        name = filename_override or upload.filename or "unknown"
        dot = name.rfind(".")
        suffix = name[dot:].lower() if dot != -1 else ""

        entry = cls.SUFFIX_MAP.get(suffix)
        if entry is None:
            supported = ", ".join(sorted(cls.SUFFIX_MAP))
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported file type '{suffix or '(none)'}' for file '{name}'. "
                    f"Supported extensions: {supported}"
                ),
            )

        category, content_type = entry
        return FileClassification(
            filename=name,
            suffix=suffix,
            category=category,
            content_type=content_type,
        )
