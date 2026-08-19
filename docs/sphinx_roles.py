# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Sphinx roles in Google-style docstrings for the MkDocs API page.

mkdocstrings-python does not resolve ``:class:`...``` roles when
``docstring_style: google``. This Griffe extension rewrites those roles
before rendering so the published API reference shows readable names and
valid 26.08 import paths instead of raw reStructuredText.
"""

from __future__ import annotations

import re
from typing import Any

import griffe

_ROLE_RE = re.compile(r":(?P<role>class|meth|func|mod|attr|exc|data|const|obj):`(?P<tilde>~)?(?P<path>[^`]+)`")

# Longest prefixes first.
_PATH_PREFIX_ALIASES: tuple[tuple[str, str], ...] = (
    ("nemo_retriever.llm.clients.judge.", "nemo_retriever.models.llm."),
    ("nemo_retriever.llm.clients.", "nemo_retriever.models.llm."),
    ("nemo_retriever.params.", "nemo_retriever.common.params."),
    ("nemo_retriever.vdb.operators.", "nemo_retriever.operators.vdb."),
    ("nemo_retriever.rerank.rerank.", "nemo_retriever.operators.rerank."),
    ("nemo_retriever.evaluation.eval_operator.", "nemo_retriever.operators.graph_ops.eval_operator."),
    ("nemo_retriever.video.", "nemo_retriever.operators.extract.video."),
    ("nemo_retriever.graph.pipeline_graph.", "nemo_retriever.graph."),
)

_PARAM_CLASS_NAMES: tuple[str, ...] = (
    "ASRParams",
    "AudioChunkParams",
    "AudioVisualFuseParams",
    "BatchTuningParams",
    "CaptionParams",
    "ChartParams",
    "DedupParams",
    "EmbedParams",
    "ExtractParams",
    "GpuAllocationParams",
    "HtmlChunkParams",
    "IngestExecuteParams",
    "IngestorCreateParams",
    "IngestorRunMode",
    "LanceDbParams",
    "LLMInferenceParams",
    "LLMRemoteClientParams",
    "LLMSamplingOverrides",
    "ModelRuntimeParams",
    "OcrParams",
    "PageElementsParams",
    "PdfSplitParams",
    "RemoteInvokeParams",
    "RemoteRetryParams",
    "StoreParams",
    "TabularExtractParams",
    "TableParams",
    "TextChunkParams",
    "TextGenerationParams",
    "MetaJoinKey",
    "VdbUploadParams",
    "VideoFrameParams",
    "VideoFrameTextDedupParams",
    "WebhookParams",
)

# Must match the ::: identifiers on nemo-retriever-api-reference.md.
_LINKABLE_PATHS: dict[str, str] = {
    **{name: f"nemo_retriever.common.params.{name}" for name in _PARAM_CLASS_NAMES},
    "create_ingestor": "nemo_retriever.ingestor.core.create_ingestor",
    "GraphIngestor": "nemo_retriever.ingestor.graph_ingestor.GraphIngestor",
    "GraphIngestionError": "nemo_retriever.ingestor.graph_ingestor.GraphIngestionError",
    "Retriever": "nemo_retriever.graph.retriever.Retriever",
    "Retriever.pipeline": "nemo_retriever.graph.retriever.Retriever.pipeline",
    "RetrieverPipelineBuilder": "nemo_retriever.graph.retriever.RetrieverPipelineBuilder",
    "LiteLLMClient": "nemo_retriever.models.llm.clients.litellm.LiteLLMClient",
    "LLMJudge": "nemo_retriever.models.llm.clients.judge.LLMJudge",
}


def _apply_path_aliases(path: str) -> str:
    for old, new in _PATH_PREFIX_ALIASES:
        if path.startswith(old):
            return new + path[len(old) :]
    return path


def _display_name(path: str) -> str:
    parts = path.split(".")
    if len(parts) == 2 and parts[0][:1].isupper():
        return path
    return parts[-1]


def _inventory_id(path: str) -> str | None:
    if path in _LINKABLE_PATHS.values():
        return path
    return _LINKABLE_PATHS.get(path) or _LINKABLE_PATHS.get(_display_name(path))


def convert_sphinx_roles(text: str) -> str:
    """Replace Sphinx roles with Markdown cross-refs or readable names."""

    def _replace(match: re.Match[str]) -> str:
        canonical = _apply_path_aliases(match.group("path").strip())
        name = _display_name(canonical)
        target = _inventory_id(canonical)
        if target:
            return f"[{name}][{target}]"
        return f"`{name}`"

    return _ROLE_RE.sub(_replace, text)


class SphinxRoleConverter(griffe.Extension):
    """Rewrite Sphinx roles in collected docstrings before Google parsing."""

    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        self._convert(obj)

    def on_object(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        self._convert(obj)

    def _convert(self, obj: griffe.Object) -> None:
        docstring = obj.docstring
        if docstring is None or not docstring.value:
            return
        converted = convert_sphinx_roles(docstring.value)
        if converted == docstring.value:
            return
        obj.docstring = griffe.Docstring(
            converted,
            parent=obj,
            parser=getattr(docstring, "parser", None),
            parser_options=getattr(docstring, "parser_options", None) or {},
        )
