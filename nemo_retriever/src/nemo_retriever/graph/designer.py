# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decorator-driven registration of Designer components.

Usage::

    from typing import Annotated
    from nemo_retriever.graph.designer import designer_component, Param

    @designer_component(
        name="PDF Splitter",
        category="Document Processing",
        compute="cpu",
        description="Splits multi-page PDFs into individual pages",
    )
    class PDFSplitActor(AbstractOperator):
        def __init__(
            self,
            mode: Annotated[str, Param(label="Split Mode", choices=["page", "chapter"])] = "page",
        ):
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Param:
    """Parameter metadata for Designer UI, used with ``typing.Annotated``."""

    label: str | None = None
    description: str | None = None
    choices: list[Any] | None = None
    min_val: float | None = None
    max_val: float | None = None
    hidden: bool = False
    placeholder: str | None = None


_REGISTRY: dict[str, dict[str, Any]] = {}


def designer_component(
    *,
    name: str,
    category: str = "General",
    compute: str = "undefined",
    description: str = "",
    category_color: str | None = None,
    component_type: str | None = None,
):
    """Register a class or function as a Designer-visible component.

    Parameters
    ----------
    name : str
        Friendly display name shown in the palette and on the canvas.
    category : str
        Section heading in the component palette (e.g. "Document Processing").
    compute : str
        Resource hint: ``"gpu"``, ``"cpu"``, or ``"undefined"``.
    description : str
        Short description shown as a tooltip.
    category_color : str | None
        Optional hex colour for the category section (e.g. ``"#64b4ff"``).
    component_type : str | None
        Special type tag for code-generation routing (e.g. ``"pipeline_sink"``,
        ``"pipeline_evaluator"``).  Leave *None* for standard ``map_batches``
        operators.
    """

    def decorator(cls_or_fn):
        key = f"{cls_or_fn.__module__}.{cls_or_fn.__qualname__}"
        meta = {
            "target": cls_or_fn,
            "name": name,
            "category": category,
            "compute": compute,
            "description": description,
            "category_color": category_color,
            "component_type": component_type,
        }
        _REGISTRY[key] = meta
        cls_or_fn._designer_meta = {
            "name": name,
            "category": category,
            "compute": compute,
            "description": description,
            "category_color": category_color,
            "component_type": component_type,
        }
        return cls_or_fn

    return decorator


def get_registry() -> dict[str, dict[str, Any]]:
    """Return a snapshot of all registered designer components."""
    return dict(_REGISTRY)
