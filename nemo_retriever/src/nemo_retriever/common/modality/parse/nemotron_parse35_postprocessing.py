# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode Parse 3.5 output with HF helpers and convert Table blocks to the selected format."""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import logging
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType
from typing import Any

from bs4 import BeautifulSoup

from nemo_retriever.common.params.utils import NEMOTRON_PARSE35_MODEL
from nemo_retriever.models.hf_cache import resolve_hf_cache_dir
from nemo_retriever.models.hf_model_registry import get_hf_revision

logger = logging.getLogger(__name__)
_IMPORT_LOCK = Lock()


@lru_cache(maxsize=1)
def _load_parse35_helpers() -> ModuleType:
    """Import HF's ``nemotron_parse`` postprocessing package at the pinned revision.

    Fetch only the helper package, without model weights. Offline runs use the
    existing HF cache. Import it under a revision-specific module name to avoid
    collisions with other installed packages, then reuse it within this process.
    """
    from huggingface_hub import constants, snapshot_download

    revision = get_hf_revision(NEMOTRON_PARSE35_MODEL)
    snapshot = Path(
        snapshot_download(
            NEMOTRON_PARSE35_MODEL,
            revision=revision,
            cache_dir=resolve_hf_cache_dir(),
            allow_patterns=["nemotron_parse/**"],
            local_files_only=constants.HF_HUB_OFFLINE,
        )
    )
    package_path = snapshot / "nemotron_parse" / "__init__.py"
    package_name = f"_nrl_parse35_postprocessing_{revision}"
    with _IMPORT_LOCK:
        existing = sys.modules.get(package_name)
        if existing is not None:
            if Path(existing.__file__).resolve() != package_path.resolve():
                raise RuntimeError("Parse 3.5 postprocessing helpers changed location; use a fresh worker process.")
            return existing
        spec = importlib.util.spec_from_file_location(
            package_name, package_path, submodule_search_locations=[str(package_path.parent)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Parse 3.5 postprocessing helpers from {package_path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for name in list(sys.modules):
                if name == package_name or name.startswith(package_name + "."):
                    sys.modules.pop(name, None)
            raise
        return module


def _postprocess_parse35_blocks(raw_text: str, *, table_format: str = "HTML") -> list[dict[str, Any]]:
    """Decode tagged model output and set the rendering of Table blocks.

    HF's ``postprocess_document`` extracts blocks with their classes, boxes,
    raw text, and rendered content. Keep the returned block order and all
    non-Table content, including Chart rendering.

    For Table blocks, ``HTML`` keeps the HF rendering. ``simple_html`` removes
    only cell spans equal to one, preserving merged cells. ``markdown`` and
    ``latex`` call HF's text formatter on the raw Table text. If Markdown
    conversion fails, retain the HTML content and log a warning.

    Return the blocks with only their Table ``content`` fields reformatted;
    the raw ``text``, class labels, and bounding boxes remain available.
    """
    if table_format not in ("HTML", "simple_html", "markdown", "latex"):
        raise ValueError("table_format must be HTML, simple_html, markdown, or latex.")
    helpers = _load_parse35_helpers()
    document = helpers.postprocess_document(raw_text)
    for warning in document.get("warnings", []):
        logger.warning("Parse 3.5 postprocessing: %s", warning["message"])
    blocks = document["blocks"]
    for block in blocks:
        if block["class"] != "Table":
            continue
        if table_format in ("markdown", "latex"):
            try:
                block["content"] = helpers.postprocessing.postprocess_text(
                    block["text"], "Table", table_format=table_format
                )
            except ValueError:
                logger.warning("Keeping HTML for a Parse 3.5 table that could not be converted to Markdown.")
        elif table_format == "simple_html" and "<table" in block["content"].lower():
            soup = BeautifulSoup(block["content"], "html.parser")
            for cell in soup.find_all(["th", "td"]):
                for attr in ("rowspan", "colspan"):
                    if cell.get(attr) == "1":
                        del cell[attr]
            block["content"] = str(soup)
    return blocks
