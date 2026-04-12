from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Annotated, Literal


# ==================== BASE MODEL ====================

# ==================== TYPE ALIASES ====================

NonEmptyStr = Annotated[str, Field(min_length=1, description="Non-empty string")]

NonEmptyStrList = Annotated[
    list[str], Field(min_length=1, description="Non-empty list of strings")
]



class StrictModel(BaseModel):
    """Base model with strict validation settings."""

    model_config = ConfigDict(
        extra="forbid",  # forbid extra fields
        validate_assignment=True,  # re-check on assignment
        str_min_length=1,  # all strings must be non-empty by default
    )



# ==================== SCORE MODELS ====================


class ItemScore(BaseModel):
    """Represents a semantic entity (attribute, metric, or analysis) with classification."""

    id: NonEmptyStr
    label: Literal["custom_analysis", "column", "query", "table"] = Field(
        ...,
        description="The label of the semantic entity (attribute | metric | analysis)",
    )
    classification: bool = Field(
        ...,
        description="True/False usage classification (True if the semantic entity was used in constructing the answer - either in SQL code or in deriving the answer from file contents/graph information)",
    )
    # score: confloat(ge=0.0, le=1.0) = Field(..., description="0..1 influence weight")
    # reason: NonEmptyStr


NonEmptyItemScoreList = Annotated[
    List[ItemScore],
    Field(min_length=1, description="Non-empty list of semantic item classifications"),
]



class SQLGenerationModel(StrictModel):
    """Model for SQL generation without formatting requirements.

    This model is used by SQL generation agents to return structured SQL data.
    Formatting is handled separately by SQLResponseFormattingAgent.
    """

    sql_code: NonEmptyStr = Field(
        ...,
        description="The SQL code that answers the user's question based on chosen snippet/s and appropriate joins. This field is REQUIRED and must not be empty. Always construct SQL even if file contents are present (use file contents as constants/filters within the SQL).",
    )
    tables_ids: list[str] = Field(
        default_factory=list,
        description="A valid python list with ids of all tables selected in the SQL query.",
    )

    response: NonEmptyStr = Field(
        ...,
        description="A short explanation of the answer and SQL parts: what the query does, which tables/columns are used, and how the SQL components work together to answer the question.",
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
