# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable text-generation task strategies."""

from nemo_retriever.models.llm.tasks.base import TextGenerationTask, GenerationTaskError
from nemo_retriever.models.llm.tasks.generic import GenericPromptTask
from nemo_retriever.models.llm.tasks.rag_answer import RagAnswerTask
from nemo_retriever.models.llm.tasks.summarize import SummarizeTask

__all__ = [
    "TextGenerationTask",
    "GenerationTaskError",
    "GenericPromptTask",
    "RagAnswerTask",
    "SummarizeTask",
]
