# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

NEMO_RETRIEVER_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = NEMO_RETRIEVER_ROOT.parent

VALID_BEIR_LOADERS = {"bo10k_csv", "bo767_csv", "earnings_csv", "financebench_json", "jp20_csv", "vidore_hf"}
VALID_BEIR_DOC_ID_FIELDS = {"pdf_basename", "pdf_page", "pdf_page_modality", "source_id", "path"}


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object at top-level: {path}")
    return data
