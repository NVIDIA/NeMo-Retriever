# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Annotated, Literal


# ==================== TYPE ALIASES ====================

NonEmptyStr = Annotated[str, Field(min_length=1, description="Non-empty string")]

NonEmptyStrList = Annotated[list[str], Field(min_length=1, description="Non-empty list of strings")]


class StrictModel(BaseModel):
    """Base model with strict validation settings."""

    model_config = ConfigDict(
        extra="forbid",  # forbid extra fields
        validate_assignment=True,  # re-check on assignment
        str_min_length=1,  # all strings must be non-empty by default
    )


# ==================== SCORE MODELS ====================


class ItemScore(BaseModel):
    """Represents a custom analysis item with classification."""

    id: NonEmptyStr
    label: Literal["custom_analysis", "column", "query", "table"] = Field(
        ...,
        description="The label of the custom analysis item",
    )
    classification: bool = Field(
        ...,
        description=(
            "True/False usage classification (True if the custom analysis was used in constructing "
            "the answer - either in SQL code or in deriving the answer from file contents/graph information)"
        ),
    )


NonEmptyItemScoreList = Annotated[
    List[ItemScore],
    Field(min_length=1, description="Non-empty list of custom analysis classifications"),
]


class TableRelevanceModel(BaseModel):
    """LLM output for table relevance filtering."""

    reasoning: str = Field(
        default="",
        description="Brief reasoning (1-2 sentences max) on which tables are relevant.",
    )
    tables_to_remove: list[str] = Field(
        default_factory=list,
        description="Names of tables that can be safely removed. Leave empty if unsure.",
    )


class CustomAnalysisRelevanceModel(BaseModel):
    """LLM output for custom analysis relevance filtering."""

    reasoning: str = Field(
        default="",
        description="Brief reasoning (1-2 sentences max) on which custom analyses are relevant.",
    )
    analyses_to_remove: list[str] = Field(
        default_factory=list,
        description="Names of custom analyses that can be safely removed. Leave empty if unsure.",
    )


class SQLGenerationModel(StrictModel):
    """Model for SQL generation without formatting requirements.

    This model is used by SQL generation agents to return structured SQL data.
    Formatting is handled separately by SQLResponseFormattingAgent.

    Field order matters: the LLM fills fields sequentially, so ``thought``
    comes first to drain reasoning before it writes the clean output fields.
    """

    thought: str = Field(
        ...,
        description=(
            "Internal reasoning (1-2 sentences): briefly explain your approach "
            "and key decisions. This is NOT shown to the user."
        ),
    )
    sql_code: NonEmptyStr = Field(
        ...,
        description=("The complete, executable SQL query. No comments, no delimiters, no explanation."),
    )
    response: NonEmptyStr = Field(
        ...,
        description=(
            "User-facing summary (1-2 sentences): what the query calculates and "
            "which tables it uses. Do NOT include reasoning, self-corrections, "
            "formatting notes, or internal thoughts."
        ),
    )

    @field_validator("sql_code", "response")
    @classmethod
    def reject_placeholder_strings(cls, v: str, info) -> str:
        """Block LLM stubs like literal '...' that satisfy min length but are not valid output."""
        t = (v or "").strip()
        if t in ("...", "…", "..", ".") or (len(t) <= 3 and not t.isalnum() and set(t) <= {".", "…", " "}):
            raise ValueError(
                f"{info.field_name!r} must be real content, not an ellipsis placeholder. "
                "sql_code must be the full executable statement; response must be a real explanation."
            )
        return v
