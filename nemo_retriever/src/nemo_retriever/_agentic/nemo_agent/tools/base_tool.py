# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base tool abstraction shared by every tool the agent can call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolError(Exception):
    """Deliberate, LLM-recoverable tool failure.

    Raise from a tool implementation when the *call* was invalid (bad argument
    values, unusable input). The public wrappers turn it into an
    ``"Error calling ..."`` string returned as the tool result, so the LLM can
    see what it did wrong and retry on the next turn.
    """


class ToolContractError(Exception):
    """A tool implementation violated a base-class contract.

    Deliberately NOT caught by the ``call``/``acall`` wrappers: a contract
    violation is an integration bug, not something the LLM can correct. It
    propagates to the agent's error policy so it surfaces as "your tool
    returned X" instead of a ``KeyError`` deep inside the agent loop.
    """


def tool_error_text(tool_name: str, exc: BaseException) -> str:
    """The LLM-visible error string for a recoverable tool failure."""
    return f"Error calling '{tool_name}' tool. {type(exc).__name__}: {str(exc)}"


class BaseTool(ABC):
    """Define a tool to be passed to the LLM.

    Subclasses implement :meth:`_spec` (OpenAI function-tool spec dict) plus
    ``_call`` and/or ``_acall``. The public ``call``/``acall`` wrappers convert
    ``TypeError`` (signature mismatch from LLM-supplied kwargs) and
    :class:`ToolError` into LLM-visible error strings; every other exception
    propagates untranslated.
    """

    @abstractmethod
    def _spec(self) -> dict:
        raise NotImplementedError

    def _call(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def _acall(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def call(self, **kwargs: Any) -> Any:
        try:
            output = self._call(**kwargs)
        except (TypeError, ToolError) as e:
            output = tool_error_text(self.name, e)
        return output

    async def acall(self, **kwargs: Any) -> Any:
        try:
            output = await self._acall(**kwargs)
        except (TypeError, ToolError) as e:
            output = tool_error_text(self.name, e)
        return output

    def __call__(self, **kwargs: Any) -> Any:
        return self.call(**kwargs)

    @property
    def spec(self) -> dict:
        return self._spec()

    @property
    def name(self) -> str:
        return str(self.spec["function"]["name"])
