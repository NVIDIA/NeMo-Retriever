# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt rendering for the agent's system prompts.

The agent renders its system prompt through :func:`render_system_prompt` and
never touches a template engine. Resolution order: an existing filesystem path
wins (so a caller can point at any file); otherwise the name must be a
packaged template under ``templates/``.

jinja2 is imported lazily, only when a prompt is actually rendered. The render
*variables* and the templates' conditional logic are the contract — if jinja2
ever becomes unacceptable as a dependency, this module's internals swap
template files for plain Python functions and nothing outside it changes.

Render variables used by the packaged system prompts: ``with_init_docs``,
``enforce_top_k``, ``top_k``, ``extended_relevance``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def available_prompts() -> List[str]:
    """Sorted relative names of the packaged prompt templates.

    Templates in subdirectories keep their subpath, e.g. the selection-agent
    set is listed (and rendered) as ``"selection/01_v0.j2"``.
    """
    if not _TEMPLATES_DIR.is_dir():
        return []
    return sorted(p.relative_to(_TEMPLATES_DIR).as_posix() for p in _TEMPLATES_DIR.rglob("*.j2"))


def render_system_prompt(name_or_path: str, **variables: Any) -> str:
    """Render a system prompt by packaged template name or filesystem path.

    An existing file path is rendered directly; otherwise ``name_or_path``
    must be a packaged template filename. The rendered prompt is stripped of
    surrounding whitespace.
    """
    path = Path(name_or_path)
    if not path.is_file():
        path = _TEMPLATES_DIR / name_or_path
        if not path.is_file():
            raise ValueError(
                f"Unknown prompt {name_or_path!r}: not an existing file path and not one of "
                f"the packaged templates: {available_prompts()}."
            )
    return _render_template_text(path.read_text(), variables)


def _render_template_text(text: str, variables: Dict[str, Any]) -> str:
    try:
        import jinja2
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise RuntimeError("Rendering prompts requires the 'jinja2' package.") from e
    return jinja2.Template(text.strip()).render(**variables).strip()
