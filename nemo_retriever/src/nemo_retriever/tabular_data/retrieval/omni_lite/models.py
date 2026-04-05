from pydantic import BaseModel, ConfigDict, Field
from typing import List, Annotated


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
    label: Literal["attribute", "metric", "analysis"] = Field(
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
    semantic_elements: List[ItemScore] = Field(
        default_factory=list,
        description=(
            "Semantic elements is a list of metrics, analyses and attributes that were in the candidates list.\n"
            "A list of dictionaries with classification of the usage in the final SQL of the given snippets.\n"
            "Only included if snippets of semantic entities were provided.\n"
            "Each dictionary must have:\n"
            "  - 'id': the unique semantic entity ID from the provided list.\n"
            "  - 'label': the label of the semantic entity (attribute|metric|analysis).\n"
            "  - 'classification': True if at least one table or column taken from that snippet appears in your final sql_code.\n"
        ),
    )
    connection: NonEmptyStr = Field(
        ...,
        description="The SQL query was constructed using the connection type—e.g., snowflake from the predefined list of permitted dialects. This field is REQUIRED.",
    )
    response: NonEmptyStr = Field(
        ...,
        description="A short explanation of the answer and SQL parts: what the query does, which tables/columns are used, and how the SQL components work together to answer the question.",
    )
