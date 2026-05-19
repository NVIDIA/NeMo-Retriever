# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final-response shaping for the chat UI.

* :func:`_prepare_link` builds the deep-link target for each entity label.
* :func:`_extract_entities_with_id_name_label` walks the candidate tree and
  flattens it into ``{id: (name, label, parent_id)}``.
* :func:`_highlight_entity` rewrites ``[[[name/id]]]`` markers in the LLM's
  response into rich links.
* :func:`format_response` is the public entry point.
"""

from __future__ import annotations

import logging
import re

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.retrieval.data_access.graph_schemas import get_item_by_id

logger = logging.getLogger(__name__)


def _prepare_link(name: str, id: str, label: Labels, parent_id: str = None) -> str:
    match label:
        case label if label in [Labels.CUSTOM_ANALYSIS]:
            return f"{label}/{id}|{name}"
        case Labels.COLUMN:
            return f"data/{parent_id}?searchId={id}|{name}"
        case _:
            return f"data/{id}|{name}"


def _extract_entities_with_id_name_label(data):
    result = {}

    def recurse(obj):
        if isinstance(obj, dict):
            # Main entity case: id + name + (type or label)
            if "id" in obj and "name" in obj and ("type" in obj or "label" in obj):

                final_label = obj["label"]

                result[obj["id"]] = (
                    obj["name"],
                    final_label,
                    obj.get("parent_id"),
                )

            # explicitly capture table inside column
            if "table" in obj and isinstance(obj["table"], dict):
                table = obj["table"]
                if "id" in table and "name" in table:
                    result[table["id"]] = (table["name"], "table", None)

            for value in obj.values():
                recurse(value)

        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(data)
    return result


def _highlight_entity(items_present: dict, text: str) -> str:
    """Processes [[[entity]]] patterns in the text.

    Supported formats:
    - [[[name/id]]]
    - [[[label/id|display_name]]]

    Replaces valid ones with hyperlinks using `_prepare_link`.
    Falls back to bolding just the entity name if invalid or not found.
    """

    def replace_entity(match):
        raw = match.group(1)
        cleaned = re.sub(r"\s*/\s*", "/", raw.strip())

        if "|" in cleaned:
            entity_part, display_name = cleaned.split("|", 1)
            display_name = display_name.strip()
        else:
            entity_part, display_name = cleaned, None

        if "/" in entity_part:
            name_or_label, eid = entity_part.split("/", 1)
            name_or_label = name_or_label.strip()
            eid = eid.strip()

            entity = items_present.get(eid)
            if entity and (
                entity[0].lower() == name_or_label.lower()
                or (display_name and entity[0].lower() == display_name.lower())
            ):
                shown_name = display_name or name_or_label
                return f"<{_prepare_link(shown_name, eid, entity[1], entity[2])}>"
            elif entity:
                logger.warning(
                    "ASSUMPTION: the name in link is not found correctly by llm, take it by id from candidates"
                )
                return f"<{_prepare_link(entity[0], eid, entity[1], entity[2])}>"

            try:
                item = get_item_by_id(eid, name_or_label)
            except Exception:
                logger.error("Something not ok with id, error raised")
                return f"*{display_name or name_or_label}*"

            if item:
                return f"<{_prepare_link(item['name'], eid, name_or_label)}>"
            else:
                logger.warning(f"Entity ID mismatch or not found: {name_or_label}/{eid}")
                return f"*{display_name or name_or_label}*"
        else:
            logger.warning(f"No ID found in entity: {cleaned}")
            return f"*{cleaned}*"

    return re.sub(r"\[\[\[(.*?)\]\]\]", replace_entity, text)


def format_response(candidates, response):
    final_response_formatted = response.replace("%%%", "```").replace("**", "*")
    final_response_formatted = re.sub(r"(\\+n|\n)", "\n ", final_response_formatted)
    all_entities_present = _extract_entities_with_id_name_label(candidates)

    try:
        final_response_highlighted = _highlight_entity(all_entities_present, final_response_formatted)
    except Exception:
        return final_response_formatted
    return final_response_highlighted
