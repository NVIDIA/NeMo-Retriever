# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM client construction and structured-output invocation wrappers."""

import logging
import os
from typing import Type, TypeVar

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


RETRY_MAX_ATTEMPTS = 3
T = TypeVar("T", bound=BaseModel)


def get_llm_client() -> ChatNVIDIA:
    return ChatNVIDIA(
        base_url=os.environ.get("BASE_URL"),
        api_key=os.environ.get("NVIDIA_API_KEY"),
        model=os.environ.get("MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b"),
    )


def safe_invoke_with_structured_output(
    llm: ChatNVIDIA,
    messages: list[BaseMessage],
    schema: Type[T],
    method: str = "function_calling",
) -> T:
    """LLM structured call with retry"""
    current_messages = messages.copy()
    schema_name = getattr(schema, "__name__", str(schema))

    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            model_llm = llm.with_structured_output(schema, method=method)
            result = model_llm.invoke(current_messages)
            return result
        except ValidationError as e:
            if attempt < RETRY_MAX_ATTEMPTS:
                # Explain what's missing/invalid and ask the model to fix it
                current_messages.append(
                    SystemMessage(
                        content=(
                            "Your previous output did not validate. "
                            f"Validation errors:\n{str(e)}\n"
                            "Please return a **fully valid** object that satisfies the schema. "
                            "Do not omit required fields. Do not include extra keys."
                        )
                    )
                )
            else:
                logger.error(f"Validation failed after {RETRY_MAX_ATTEMPTS} attempts for {schema_name}")
                raise  # If still failing after max tries, raise

        except Exception as e:
            logger.error(
                f"Unexpected error on attempt {attempt + 1}/{RETRY_MAX_ATTEMPTS} for {schema_name}: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise


def invoke_with_structured_output(
    llm: ChatNVIDIA,
    messages: list[BaseMessage],
    schema: Type[T],
    method: str = "function_calling",
) -> T | None:
    """Safe wrapper for invoke_with_structured_output that returns None on failure"""
    try:
        schema_name = getattr(schema, "__name__", str(schema))
        return safe_invoke_with_structured_output(llm, messages, schema, method)
    except Exception as e:
        logger.error(
            f"invoke_with_structured_output failed for {schema_name} after {RETRY_MAX_ATTEMPTS} attempts: "
            f"{type(e).__name__}: {e}",
            exc_info=True,
        )
        return None
