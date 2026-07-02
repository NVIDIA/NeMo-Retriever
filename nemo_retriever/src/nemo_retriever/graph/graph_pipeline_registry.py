# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph Pipeline Registry — manage, inspect, compare, and serialize golden pipeline graphs.

Provides a central :class:`GraphPipelineRegistry` that stores named graph
*blueprints* (factory functions + metadata).  Graphs built from the registry
can be inspected, diffed against each other, serialized to / loaded from JSON,
and configured with kwarg overrides — all without touching the code that
originally defined them.

A module-level :data:`default_registry` is provided for convenience so that
graph definitions scattered across the codebase can all register to a single
shared instance.

Quick-start::

    from nemo_retriever.graph.graph_pipeline_registry import default_registry

    @default_registry.register("my-pipeline", description="Demo pipeline")
    def _build():
        from nemo_retriever.graph import Graph
        return Graph() >> SomeOperator() >> AnotherOperator()

    graph = default_registry.build("my-pipeline")
    default_registry.print_graph("my-pipeline")
"""

from __future__ import annotations

import importlib
import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydantic import BaseModel

from nemo_retriever.graph.pipeline_graph import Graph, Node
from nemo_retriever.operators.abstract_operator import AbstractOperator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qualified_name(cls: type) -> str:
    """Return the fully qualified ``module.ClassName`` string for *cls*."""
    module = cls.__module__ or "__main__"
    return f"{module}.{cls.__qualname__}"


def _import_class(qualified: str) -> type:
    """Import and return a class from its fully qualified dotted path."""
    module_path, _, class_name = qualified.rpartition(".")
    if not module_path:
        raise ImportError(f"Cannot import class from unqualified name: {qualified!r}")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Module {module_path!r} has no attribute {class_name!r}")
    return cls


_GRAPH_FORMAT_VERSION = 2
_PYDANTIC_MODEL_MARKER = "__pydantic_model__"
_PYDANTIC_FIELDS = "fields"
_PYDANTIC_FIELDS_SET = "fields_set"
_SECRET_NO_AUTH_MARKER = "__secret_no_auth__"
_TUPLE_MARKER = "__tuple__"
_FROZENSET_MARKER = "__frozenset__"
_MAPPING_MARKER = "__mapping__"
_OMIT_FIELD = object()
_ENV_REFERENCE_PREFIX = "os.environ/"
_ENV_VARIABLE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class GraphSerializationError(ValueError):
    """Raised when graph state cannot be serialized safely and losslessly."""


def _normalize_field_name(field_name: str) -> str:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field_name)
    return snake_case.replace("-", "_").lower()


def _is_api_key_field(field_name: Optional[str]) -> bool:
    if not field_name:
        return False
    normalized = _normalize_field_name(field_name)
    compact = normalized.replace("_", "")
    return normalized == "api_key" or normalized.endswith("_api_key") or compact.endswith("apikey")


def _is_obvious_secret_field(field_name: Optional[str]) -> bool:
    if not field_name:
        return False
    normalized = _normalize_field_name(field_name)
    compact = normalized.replace("_", "")
    if _is_api_key_field(normalized):
        return True
    if normalized in {
        "access_key",
        "account_key",
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "password",
        "passwd",
        "private_key",
        "secret",
        "secret_key",
    }:
        return True
    parts = set(normalized.split("_"))
    if parts & {
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "password",
        "passwd",
        "secret",
        "token",
    }:
        return True
    if {"access", "key"} <= parts or {"account", "key"} <= parts:
        return True
    if normalized == "token" or normalized.endswith("_token"):
        return True
    if compact in {
        "accesskey",
        "accountkey",
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "credential",
        "credentials",
    }:
        return True
    if "accesskey" in compact or "accountkey" in compact:
        return True
    return compact.endswith(
        (
            "apikey",
            "password",
            "passwd",
            "secret",
            "secretkey",
            "privatekey",
            "token",
            "cookie",
            "bearer",
            "credential",
        )
    )


def _is_storage_option_secret_field(field_name: Optional[str]) -> bool:
    if not field_name:
        return False
    normalized = _normalize_field_name(field_name)
    return normalized == "key" or _is_obvious_secret_field(field_name)


def _is_empty_secret(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return len(value) == 0
    return False


def _import_qualified_object(qualified: str) -> Any:
    """Import a qualified module attribute, including nested attributes."""
    parts = qualified.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        try:
            obj: Any = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        try:
            for part in parts[index:]:
                obj = getattr(obj, part)
        except AttributeError as exc:
            raise ImportError(f"Cannot import qualified object {qualified!r}") from exc
        return obj
    raise ImportError(f"Cannot import qualified object {qualified!r}")


def _model_uses_no_api_key(model: BaseModel, field_name: str, *, path: str) -> bool:
    checker = getattr(model, "_uses_no_api_key", None)
    if callable(checker):
        provenance_error: Optional[GraphSerializationError] = None
        try:
            uses_no_auth = bool(checker(field_name))
        except Exception as exc:
            exc_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
            provenance_error = GraphSerializationError(f"{path}: no-auth credential provenance failed ({exc_type})")
        if provenance_error is not None:
            raise provenance_error
        return uses_no_auth
    return field_name in getattr(model, "_no_api_key_fields", set())


def _normalize_env_reference(value: str, *, path: str) -> str:
    """Return a canonical os.environ/NAME reference or raise safely."""
    candidate = value.strip()
    if not candidate.startswith(_ENV_REFERENCE_PREFIX):
        raise GraphSerializationError(
            f"{path}: API keys cannot be persisted literally; use " f"{_ENV_REFERENCE_PREFIX}VARIABLE_NAME"
        )
    variable = candidate.removeprefix(_ENV_REFERENCE_PREFIX)
    if not _ENV_VARIABLE_RE.fullmatch(variable):
        raise GraphSerializationError(
            f"{path}: invalid environment reference; expected " f"{_ENV_REFERENCE_PREFIX}VARIABLE_NAME"
        )
    return f"{_ENV_REFERENCE_PREFIX}{variable}"


def _model_api_key_env_reference(model: BaseModel, field_name: str, *, path: str) -> Optional[str]:
    """Read optional credential provenance without depending on a model base."""
    getter = getattr(model, "_api_key_env_reference", None)
    if not callable(getter):
        return None
    provenance_error: Optional[GraphSerializationError] = None
    try:
        reference = getter(field_name)
    except Exception as exc:
        exc_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
        provenance_error = GraphSerializationError(f"{path}: API-key environment provenance failed ({exc_type})")
        reference = None
    if provenance_error is not None:
        raise provenance_error
    if reference is None:
        return None
    if not isinstance(reference, str):
        raise GraphSerializationError(f"{path}: API-key environment provenance must be a string or null")
    return _normalize_env_reference(reference, path=path)


def _contains_pydantic_model(value: Any, seen: Optional[Set[int]] = None) -> bool:
    if isinstance(value, BaseModel):
        return True
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return False
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return False
    seen.add(value_id)
    if isinstance(value, dict):
        return any(_contains_pydantic_model(item, seen) for pair in value.items() for item in pair)
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_pydantic_model(item, seen) for item in value)
    return False


def _encode_secret(
    value: Any,
    *,
    field_name: str,
    path: str,
    owner: Optional[BaseModel],
    allow_api_key_env: bool,
) -> Any:
    if _is_api_key_field(field_name):
        if (isinstance(value, str) and value == "") or (
            value is None and owner is not None and _model_uses_no_api_key(owner, field_name, path=path)
        ):
            return {_SECRET_NO_AUTH_MARKER: ""}
        if owner is not None:
            reference = _model_api_key_env_reference(owner, field_name, path=path)
            if reference is not None:
                return reference
        if isinstance(value, str) and value.strip().startswith(_ENV_REFERENCE_PREFIX):
            if owner is not None or allow_api_key_env:
                return _normalize_env_reference(value, path=path)
        if owner is None and not allow_api_key_env:
            if value is None:
                return None
            raise GraphSerializationError(
                f"{path}: refusing to serialize an API key inside an opaque mapping; "
                "move it to a typed params field or top-level operator kwarg"
            )
        if value is not None and not isinstance(value, str):
            raise GraphSerializationError(f"{path}: API-key fields must be strings, null, or the no-auth marker")
        if value is None:
            return None
        raise GraphSerializationError(
            f"{path}: API keys cannot be persisted literally; use " f"{_ENV_REFERENCE_PREFIX}VARIABLE_NAME"
        )
    if value is None or (isinstance(value, str) and value == ""):
        return value
    raise GraphSerializationError(f"{path}: refusing to serialize non-rehydratable secret field {field_name!r}")


def _encode_value(
    value: Any,
    *,
    path: str,
    field_name: Optional[str] = None,
    owner: Optional[BaseModel] = None,
    allow_api_key_env: bool = False,
    inside_storage_options: bool = False,
) -> Any:
    """Recursively encode ``value`` into lossless, JSON-native graph state."""
    normalized_field = _normalize_field_name(field_name) if field_name else None
    inside_storage_options = inside_storage_options or normalized_field == "storage_options"
    if _is_obvious_secret_field(field_name) or (inside_storage_options and _is_storage_option_secret_field(field_name)):
        return _encode_secret(
            value,
            field_name=field_name or "",
            path=path,
            owner=owner,
            allow_api_key_env=allow_api_key_env,
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        model_type = type(value)
        qualified = _qualified_name(model_type)
        try:
            restored_type = _import_qualified_object(qualified)
        except ImportError as exc:
            raise GraphSerializationError(f"{path}: Pydantic model {qualified!r} is not rehydratable") from exc
        if restored_type is not model_type:
            raise GraphSerializationError(f"{path}: Pydantic model {qualified!r} does not round-trip by identity")
        serialization_error: Optional[GraphSerializationError] = None
        try:
            dumped = value.model_dump(mode="python")
        except Exception as exc:
            exc_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
            serialization_error = GraphSerializationError(f"{path}: Pydantic model serialization failed ({exc_type})")
            dumped = None
        if serialization_error is not None:
            raise serialization_error
        if not isinstance(dumped, dict):
            raise GraphSerializationError(
                f"{path}: Pydantic model serializer must return a mapping, got {type(dumped).__name__}"
            )
        fields: Dict[str, Any] = {}
        for name, dumped_item in dumped.items():
            actual = getattr(value, name, dumped_item)
            item = actual if _contains_pydantic_model(actual) else dumped_item
            fields[name] = _encode_value(
                item,
                path=f"{path}.{name}",
                field_name=name,
                owner=value,
                inside_storage_options=inside_storage_options,
            )
        return {
            _PYDANTIC_MODEL_MARKER: qualified,
            _PYDANTIC_FIELDS: fields,
            _PYDANTIC_FIELDS_SET: sorted(value.model_fields_set),
        }
    if isinstance(value, type):
        qualified = _qualified_name(value)
        try:
            restored = _import_qualified_object(qualified)
        except ImportError as exc:
            raise GraphSerializationError(f"{path}: type {qualified!r} is not rehydratable") from exc
        if restored is not value:
            raise GraphSerializationError(f"{path}: type {qualified!r} does not round-trip by identity")
        return {"__type_ref__": qualified}
    if callable(value) and hasattr(value, "__qualname__"):
        module = getattr(value, "__module__", None) or ""
        qualified = f"{module}.{value.__qualname__}"
        try:
            restored = _import_qualified_object(qualified)
        except ImportError as exc:
            raise GraphSerializationError(f"{path}: callable {qualified!r} is not rehydratable") from exc
        if restored is not value:
            raise GraphSerializationError(f"{path}: callable {qualified!r} does not round-trip by identity")
        return {"__callable_ref__": qualified}
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, tuple):
        return {
            _TUPLE_MARKER: [
                _encode_value(
                    item,
                    path=f"{path}[{index}]",
                    inside_storage_options=inside_storage_options,
                )
                for index, item in enumerate(value)
            ]
        }
    if isinstance(value, list):
        return [
            _encode_value(
                item,
                path=f"{path}[{index}]",
                inside_storage_options=inside_storage_options,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        encoded = [
            _encode_value(
                item,
                path=f"{path}[{index}]",
                inside_storage_options=inside_storage_options,
            )
            for index, item in enumerate(value)
        ]
        encoded.sort(key=lambda item: json.dumps(item, sort_keys=True))
        marker = _FROZENSET_MARKER if isinstance(value, frozenset) else "__set__"
        return {marker: encoded}
    if isinstance(value, dict):
        encoded_dict: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                key_type = f"{type(key).__module__}.{type(key).__qualname__}"
                raise GraphSerializationError(
                    f"{path}: mapping key of type {key_type} is not a string and cannot round-trip through JSON"
                )
            encoded_dict[key] = _encode_value(
                item,
                path=f"{path}.{key}",
                field_name=key,
                inside_storage_options=inside_storage_options,
            )
        return {_MAPPING_MARKER: encoded_dict}
    raise GraphSerializationError(
        f"{path}: unsupported value of type {type(value).__module__}.{type(value).__qualname__}"
    )


def _validate_value_envelope(
    value: dict,
    *,
    required: Set[str],
    optional: Set[str] = frozenset(),
    path: str,
    label: str,
) -> None:
    keys = set(value)
    if not required <= keys or not keys <= required | optional:
        raise GraphSerializationError(f"{path}: malformed {label} envelope")


def _decode_value(
    value: Any,
    *,
    path: str,
    format_version: int,
    field_name: Optional[str] = None,
    inside_storage_options: bool = False,
) -> Any:
    """Recursively restore graph state encoded by v2 or accepted v1 markers."""
    normalized_field = _normalize_field_name(field_name) if field_name else None
    inside_storage_options = inside_storage_options or normalized_field == "storage_options"
    is_api_key = _is_api_key_field(field_name)
    is_secret = _is_obvious_secret_field(field_name) or (
        inside_storage_options and _is_storage_option_secret_field(field_name)
    )
    if format_version >= 2 and is_secret and not is_api_key:
        if value is None or (isinstance(value, str) and value == ""):
            return value
        raise GraphSerializationError(f"{path}: serialized secret fields must be empty")
    if format_version >= 2 and is_api_key:
        if value is None:
            return None
        if isinstance(value, str):
            return _normalize_env_reference(value, path=path)
        if not (isinstance(value, dict) and _SECRET_NO_AUTH_MARKER in value):
            raise GraphSerializationError(f"{path}: invalid encoded API-key value")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [
            _decode_value(
                item,
                path=f"{path}[{index}]",
                format_version=format_version,
                inside_storage_options=inside_storage_options,
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        if format_version == 1:
            return value
        raise GraphSerializationError(f"{path}: expected JSON-native graph state")
    if format_version >= 2 and _SECRET_NO_AUTH_MARKER in value:
        _validate_value_envelope(
            value,
            required={_SECRET_NO_AUTH_MARKER},
            path=path,
            label="no-auth API-key",
        )
        if not _is_api_key_field(field_name) or value[_SECRET_NO_AUTH_MARKER] != "":
            raise GraphSerializationError(f"{path}: invalid no-auth API-key marker")
        return ""
    if format_version >= 2 and _MAPPING_MARKER in value:
        _validate_value_envelope(
            value,
            required={_MAPPING_MARKER},
            path=path,
            label="mapping",
        )
        items = value[_MAPPING_MARKER]
        if not isinstance(items, dict) or not all(isinstance(key, str) for key in items):
            raise GraphSerializationError(f"{path}: malformed mapping envelope")
        decoded_mapping: Dict[str, Any] = {}
        for key, item in items.items():
            decoded = _decode_value(
                item,
                path=f"{path}.{key}",
                format_version=format_version,
                field_name=key,
                inside_storage_options=inside_storage_options,
            )
            if decoded is not _OMIT_FIELD:
                decoded_mapping[key] = decoded
        return decoded_mapping
    if format_version >= 2 and _PYDANTIC_MODEL_MARKER in value:
        _validate_value_envelope(
            value,
            required={_PYDANTIC_MODEL_MARKER, _PYDANTIC_FIELDS},
            optional={_PYDANTIC_FIELDS_SET},
            path=path,
            label="Pydantic model",
        )
        qualified = value[_PYDANTIC_MODEL_MARKER]
        fields = value[_PYDANTIC_FIELDS]
        fields_set = value.get(_PYDANTIC_FIELDS_SET, [])
        if (
            not isinstance(qualified, str)
            or not isinstance(fields, dict)
            or not all(isinstance(name, str) for name in fields)
        ):
            raise GraphSerializationError(f"{path}: malformed Pydantic model envelope")
        try:
            model_cls = _import_qualified_object(qualified)
        except ImportError as exc:
            raise GraphSerializationError(f"{path}: Pydantic model {qualified!r} is not importable") from exc
        if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
            raise GraphSerializationError(f"{path}: {qualified!r} is not a Pydantic model type")
        decoded_fields: Dict[str, Any] = {}
        for name, item in fields.items():
            decoded = _decode_value(
                item,
                path=f"{path}.{name}",
                format_version=format_version,
                field_name=name,
                inside_storage_options=inside_storage_options,
            )
            if decoded is not _OMIT_FIELD:
                decoded_fields[name] = decoded
        validation_error: Optional[GraphSerializationError] = None
        try:
            model = model_cls.model_validate(decoded_fields)
        except Exception as exc:
            exc_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
            validation_error = GraphSerializationError(
                f"{path}: failed to validate restored Pydantic model {qualified!r} ({exc_type})"
            )
            model = None
        if validation_error is not None:
            raise validation_error
        if not isinstance(fields_set, list) or not all(isinstance(name, str) for name in fields_set):
            raise GraphSerializationError(f"{path}: malformed Pydantic fields_set")
        unknown = set(fields_set) - set(type(model).model_fields)
        if unknown:
            raise GraphSerializationError(f"{path}: Pydantic fields_set contains unknown fields: {sorted(unknown)}")
        model.__pydantic_fields_set__ = set(fields_set)
        return model
    if "__type_ref__" in value:
        if format_version >= 2:
            _validate_value_envelope(
                value,
                required={"__type_ref__"},
                path=path,
                label="type reference",
            )
        qualified = value["__type_ref__"]
        if format_version >= 2 and not isinstance(qualified, str):
            raise GraphSerializationError(f"{path}: malformed type-reference envelope")
        try:
            restored = _import_qualified_object(qualified)
        except (ImportError, TypeError) as exc:
            if format_version == 1:
                return value
            raise GraphSerializationError(f"{path}: type reference {qualified!r} is not importable") from exc
        if not isinstance(restored, type):
            if format_version == 1:
                return value
            raise GraphSerializationError(f"{path}: type reference {qualified!r} is not a type")
        return restored
    if "__callable_ref__" in value:
        if format_version >= 2:
            _validate_value_envelope(
                value,
                required={"__callable_ref__"},
                path=path,
                label="callable reference",
            )
        qualified = value["__callable_ref__"]
        if format_version >= 2 and not isinstance(qualified, str):
            raise GraphSerializationError(f"{path}: malformed callable-reference envelope")
        try:
            restored = _import_qualified_object(qualified)
        except (ImportError, TypeError) as exc:
            if format_version == 1:
                return value
            raise GraphSerializationError(f"{path}: callable reference {qualified!r} is not importable") from exc
        if not callable(restored):
            if format_version == 1:
                return value
            raise GraphSerializationError(f"{path}: callable reference {qualified!r} is not callable")
        return restored
    if "__path__" in value:
        if format_version >= 2:
            _validate_value_envelope(
                value,
                required={"__path__"},
                path=path,
                label="path",
            )
            if not isinstance(value["__path__"], str):
                raise GraphSerializationError(f"{path}: malformed path envelope")
        return Path(value["__path__"])
    if "__set__" in value:
        if format_version >= 2:
            _validate_value_envelope(
                value,
                required={"__set__"},
                path=path,
                label="set",
            )
            if not isinstance(value["__set__"], list):
                raise GraphSerializationError(f"{path}: malformed set envelope")
        return {
            _decode_value(
                item,
                path=f"{path}[{index}]",
                format_version=format_version,
                inside_storage_options=inside_storage_options,
            )
            for index, item in enumerate(value["__set__"])
        }
    if format_version >= 2 and _FROZENSET_MARKER in value:
        _validate_value_envelope(
            value,
            required={_FROZENSET_MARKER},
            path=path,
            label="frozenset",
        )
        if not isinstance(value[_FROZENSET_MARKER], list):
            raise GraphSerializationError(f"{path}: malformed frozenset envelope")
        return frozenset(
            _decode_value(
                item,
                path=f"{path}[{index}]",
                format_version=format_version,
                inside_storage_options=inside_storage_options,
            )
            for index, item in enumerate(value[_FROZENSET_MARKER])
        )
    if format_version >= 2 and _TUPLE_MARKER in value:
        _validate_value_envelope(
            value,
            required={_TUPLE_MARKER},
            path=path,
            label="tuple",
        )
        if not isinstance(value[_TUPLE_MARKER], list):
            raise GraphSerializationError(f"{path}: malformed tuple envelope")
        return tuple(
            _decode_value(
                item,
                path=f"{path}[{index}]",
                format_version=format_version,
                inside_storage_options=inside_storage_options,
            )
            for index, item in enumerate(value[_TUPLE_MARKER])
        )
    if format_version >= 2:
        raise GraphSerializationError(f"{path}: unrecognized encoded mapping")
    return {
        key: _decode_value(
            item,
            path=f"{path}.{key}",
            format_version=format_version,
            field_name=key,
            inside_storage_options=inside_storage_options,
        )
        for key, item in value.items()
    }


class _RegistryJSONEncoder(json.JSONEncoder):
    """Compatibility encoder delegating non-native values to the v2 codec."""

    def default(self, obj: Any) -> Any:
        return _encode_value(obj, path="$json")


def _safe_serialize_value(value: Any) -> Any:
    """Encode a value without lossy repr fallback (compatibility helper)."""
    return _encode_value(value, path="$value")


# ---------------------------------------------------------------------------
# Graph walking / introspection utilities
# ---------------------------------------------------------------------------


def walk_nodes(graph: Graph) -> Iterator[Tuple[Node, int]]:
    """Yield ``(node, depth)`` for every unique node via depth-first traversal."""
    visited: Set[int] = set()

    def _dfs(node: Node, depth: int) -> Iterator[Tuple[Node, int]]:
        nid = id(node)
        if nid in visited:
            return
        visited.add(nid)
        yield node, depth
        for child in node.children:
            yield from _dfs(child, depth + 1)

    for root in graph.roots:
        yield from _dfs(root, 0)


def collect_nodes(graph: Graph) -> List[Node]:
    """Return an ordered list of all unique nodes in the graph."""
    return [node for node, _ in walk_nodes(graph)]


def node_count(graph: Graph) -> int:
    """Return the total number of unique nodes in the graph."""
    return len(collect_nodes(graph))


def max_depth(graph: Graph) -> int:
    """Return the maximum depth (longest root-to-leaf path) of the graph."""
    return max((d for _, d in walk_nodes(graph)), default=0)


def find_node(graph: Graph, name: str) -> Optional[Node]:
    """Return the first node whose ``name`` matches *name*, or ``None``."""
    for node, _ in walk_nodes(graph):
        if node.name == name:
            return node
    return None


def find_nodes(graph: Graph, name: str) -> List[Node]:
    """Return every node whose ``name`` matches *name*."""
    return [node for node, _ in walk_nodes(graph) if node.name == name]


def leaf_nodes(graph: Graph) -> List[Node]:
    """Return all leaf nodes (nodes with no children)."""
    return [node for node in collect_nodes(graph) if not node.children]


def get_node_kwargs(graph: Graph, name: str) -> Dict[str, Any]:
    """Return the ``operator_kwargs`` for the first node named *name*.

    Raises ``KeyError`` if no node matches.
    """
    node = find_node(graph, name)
    if node is None:
        raise KeyError(f"No node named {name!r} in graph")
    return dict(node.operator_kwargs)


def list_all_kwargs(graph: Graph) -> Dict[str, Dict[str, Any]]:
    """Return ``{node_name: operator_kwargs}`` for every node in the graph."""
    return {node.name: dict(node.operator_kwargs) for node in collect_nodes(graph)}


# ---------------------------------------------------------------------------
# Pretty-print / inspection
# ---------------------------------------------------------------------------


def _redact_display_value(
    value: Any,
    *,
    field_name: Optional[str] = None,
    seen: Optional[Set[int]] = None,
    inside_storage_options: bool = False,
) -> Any:
    """Return a recursively redacted copy suitable only for diagnostics."""
    normalized_field = _normalize_field_name(field_name) if field_name else None
    inside_storage_options = inside_storage_options or normalized_field == "storage_options"
    is_secret = _is_obvious_secret_field(field_name) or (
        inside_storage_options and _is_storage_option_secret_field(field_name)
    )
    if is_secret and not _is_empty_secret(value):
        return "***"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return "<recursive>"
    seen.add(value_id)

    if isinstance(value, BaseModel):
        return {
            name: _redact_display_value(
                getattr(value, name),
                field_name=name,
                seen=seen,
                inside_storage_options=inside_storage_options,
            )
            for name in type(value).model_fields
        }
    if isinstance(value, dict):
        redacted_mapping: Dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, (str, int, float, bool)) or key is None:
                display_key = key
                nested_field = key if isinstance(key, str) else None
            else:
                key_type = type(key)
                display_key = f"<{key_type.__module__}.{key_type.__qualname__}>"
                nested_field = None
            redacted_mapping[display_key] = _redact_display_value(
                item,
                field_name=nested_field,
                seen=seen,
                inside_storage_options=inside_storage_options,
            )
        return redacted_mapping
    if isinstance(value, (list, tuple)):
        redacted = [
            _redact_display_value(
                item,
                seen=seen,
                inside_storage_options=inside_storage_options,
            )
            for item in value
        ]
        return tuple(redacted) if isinstance(value, tuple) else redacted
    if isinstance(value, (set, frozenset)):
        return [
            _redact_display_value(
                item,
                seen=seen,
                inside_storage_options=inside_storage_options,
            )
            for item in value
        ]
    value_type = type(value)
    return f"<{value_type.__module__}.{value_type.__qualname__}>"


def _display_repr(field_name: str, value: Any) -> str:
    return repr(_redact_display_value(value, field_name=field_name))


def format_graph_tree(
    graph: Graph,
    *,
    show_kwargs: bool = False,
    show_class: bool = True,
    max_value_width: int = 120,
) -> str:
    """Return a human-readable tree representation of the graph.

    Parameters
    ----------
    graph
        The graph to format.
    show_kwargs
        Display each node's ``operator_kwargs`` beneath it.
    show_class
        Show the fully qualified operator class next to the node name.
    max_value_width
        Truncate kwarg value reprs longer than this.
    """
    lines: List[str] = []
    visited: Set[int] = set()

    def _resource_marker(node: Node) -> str:
        try:
            from nemo_retriever.operators.cpu_operator import CPUOperator
            from nemo_retriever.operators.gpu_operator import GPUOperator

            if isinstance(node.operator, GPUOperator):
                return " [GPU]"
            if isinstance(node.operator, CPUOperator):
                return " [CPU]"
        except ImportError:
            pass
        return ""

    def _render(node: Node, prefix: str, is_last: bool, is_root: bool) -> None:
        nid = id(node)
        if nid in visited:
            connector = "" if is_root else ("└── " if is_last else "├── ")
            lines.append(f"{prefix}{connector}↻ {node.name} (back-ref)")
            return
        visited.add(nid)

        connector = "" if is_root else ("└── " if is_last else "├── ")
        marker = _resource_marker(node)
        class_info = f"  ({_qualified_name(node.operator_class)})" if show_class else ""
        lines.append(f"{prefix}{connector}{node.name}{marker}{class_info}")

        if show_kwargs and node.operator_kwargs:
            kw_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
            for key, val in sorted(node.operator_kwargs.items()):
                val_repr = _display_repr(key, val)
                if len(val_repr) > max_value_width:
                    val_repr = val_repr[: max_value_width - 3] + "..."
                lines.append(f"{kw_prefix}  ╰ {key} = {val_repr}")

        child_prefix = prefix + ("" if is_root else ("    " if is_last else "│   "))
        for i, child in enumerate(node.children):
            _render(
                child,
                child_prefix,
                is_last=(i == len(node.children) - 1),
                is_root=False,
            )

    for i, root in enumerate(graph.roots):
        if i > 0:
            lines.append("")
        _render(root, "", is_last=(i == len(graph.roots) - 1), is_root=True)

    return "\n".join(lines)


def format_node_details(node: Node) -> str:
    """Return a detailed multi-line description of a single node."""
    lines = [
        f"Node: {node.name}",
        f"  Operator class : {_qualified_name(node.operator_class)}",
        f"  Children       : {[c.name for c in node.children]}",
        f"  Kwargs ({len(node.operator_kwargs)}):",
    ]
    for key, val in sorted(node.operator_kwargs.items()):
        val_repr = _display_repr(key, val)
        if len(val_repr) > 200:
            val_repr = val_repr[:197] + "..."
        lines.append(f"    {key:30s} = {val_repr}")
    return "\n".join(lines)


def format_graph_summary(graph: Graph) -> str:
    """Return a concise summary: node count, depth, root/leaf names."""
    nodes = collect_nodes(graph)
    leaves = [n for n in nodes if not n.children]
    root_names = [r.name for r in graph.roots]
    leaf_names = [n.name for n in leaves]
    return (
        f"Graph Summary\n"
        f"  Roots ({len(root_names)}) : {root_names}\n"
        f"  Leaves ({len(leaf_names)}): {leaf_names}\n"
        f"  Total nodes    : {len(nodes)}\n"
        f"  Max depth      : {max_depth(graph)}"
    )


def format_full_report(graph: Graph, *, show_kwargs: bool = True) -> str:
    """Return a complete inspection report: summary + tree + per-node details."""
    sections: List[str] = [
        format_graph_summary(graph),
        "",
        format_graph_tree(graph, show_kwargs=show_kwargs),
        "",
    ]
    for node in collect_nodes(graph):
        sections.append(format_node_details(node))
        sections.append("")
    return "\n".join(sections)


def print_graph(graph: Graph, *, show_kwargs: bool = True) -> None:
    """Print a full graph inspection to stdout."""
    print(format_full_report(graph, show_kwargs=show_kwargs))


# ---------------------------------------------------------------------------
# Configuration update
# ---------------------------------------------------------------------------


def update_node_kwargs(
    graph: Graph,
    node_name: str,
    updates: Dict[str, Any],
    *,
    all_matches: bool = False,
) -> int:
    """Update ``operator_kwargs`` for node(s) matching *node_name* in-place.

    Parameters
    ----------
    graph
        The graph to modify.
    node_name
        Name of the target node(s).
    updates
        ``{kwarg_key: new_value}`` pairs to merge in.
    all_matches
        If ``True``, update every matching node.  Otherwise update only the
        first match and raise ``KeyError`` if none is found.

    Returns
    -------
    int
        Number of nodes updated.
    """
    if all_matches:
        targets = find_nodes(graph, node_name)
    else:
        target = find_node(graph, node_name)
        if target is None:
            raise KeyError(f"No node named {node_name!r} found in graph")
        targets = [target]

    for node in targets:
        node.operator_kwargs.update(updates)
    return len(targets)


def remove_node_kwargs(
    graph: Graph,
    node_name: str,
    keys: Sequence[str],
    *,
    all_matches: bool = False,
) -> int:
    """Remove specific kwarg keys from node(s) matching *node_name*.

    Returns the number of nodes modified.  Missing keys are silently ignored.
    """
    if all_matches:
        targets = find_nodes(graph, node_name)
    else:
        target = find_node(graph, node_name)
        if target is None:
            raise KeyError(f"No node named {node_name!r} found in graph")
        targets = [target]

    for node in targets:
        for key in keys:
            node.operator_kwargs.pop(key, None)
    return len(targets)


def replace_node_kwargs(
    graph: Graph,
    node_name: str,
    new_kwargs: Dict[str, Any],
    *,
    all_matches: bool = False,
) -> int:
    """Replace the entire ``operator_kwargs`` dict for matching node(s).

    Returns the number of nodes modified.
    """
    if all_matches:
        targets = find_nodes(graph, node_name)
    else:
        target = find_node(graph, node_name)
        if target is None:
            raise KeyError(f"No node named {node_name!r} found in graph")
        targets = [target]

    for node in targets:
        node.operator_kwargs.clear()
        node.operator_kwargs.update(new_kwargs)
    return len(targets)


# ---------------------------------------------------------------------------
# Graph comparison / diff
# ---------------------------------------------------------------------------


@dataclass
class NodeDiff:
    """Differences between two nodes at corresponding positions."""

    position: str
    node_a_name: str
    node_b_name: str
    name_changed: bool = False
    class_changed: bool = False
    class_a: str = ""
    class_b: str = ""
    kwargs_added: Dict[str, Any] = field(default_factory=dict)
    kwargs_removed: Dict[str, Any] = field(default_factory=dict)
    kwargs_changed: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    children_a_only: List[str] = field(default_factory=list)
    children_b_only: List[str] = field(default_factory=list)


@dataclass
class GraphDiff:
    """Full diff result between two graphs."""

    identical: bool
    structural_match: bool
    node_count_a: int
    node_count_b: int
    roots_a: List[str]
    roots_b: List[str]
    node_diffs: List[NodeDiff] = field(default_factory=list)
    nodes_only_in_a: List[str] = field(default_factory=list)
    nodes_only_in_b: List[str] = field(default_factory=list)

    def format(self) -> str:
        """Return a human-readable diff report."""
        lines: List[str] = []
        sep = "=" * 72
        lines.append(sep)
        lines.append("GRAPH COMPARISON REPORT")
        lines.append(sep)
        lines.append(f"  Identical        : {self.identical}")
        lines.append(f"  Structural match : {self.structural_match}")
        lines.append(f"  Nodes (A / B)    : {self.node_count_a} / {self.node_count_b}")
        lines.append(f"  Roots (A)        : {self.roots_a}")
        lines.append(f"  Roots (B)        : {self.roots_b}")

        if self.nodes_only_in_a:
            lines.append(f"\n  Nodes only in A: {self.nodes_only_in_a}")
        if self.nodes_only_in_b:
            lines.append(f"  Nodes only in B: {self.nodes_only_in_b}")

        if self.node_diffs:
            lines.append("")
            lines.append("-" * 72)
            lines.append("NODE DIFFS")
            lines.append("-" * 72)
            for nd in self.node_diffs:
                lines.append(f"\n  Position: {nd.position}")
                if nd.name_changed:
                    lines.append(f"    Name     : {nd.node_a_name!r} -> {nd.node_b_name!r}")
                else:
                    lines.append(f"    Node     : {nd.node_a_name!r}")
                if nd.class_changed:
                    lines.append(f"    Class    : {nd.class_a} -> {nd.class_b}")
                if nd.kwargs_added:
                    lines.append("    + Added kwargs:")
                    for k, v in sorted(nd.kwargs_added.items()):
                        lines.append(f"        {k} = {_display_repr(k, v)}")
                if nd.kwargs_removed:
                    lines.append("    - Removed kwargs:")
                    for k, v in sorted(nd.kwargs_removed.items()):
                        lines.append(f"        {k} = {_display_repr(k, v)}")
                if nd.kwargs_changed:
                    lines.append("    ~ Changed kwargs:")
                    for k, (old, new) in sorted(nd.kwargs_changed.items()):
                        lines.append(f"        {k}: {_display_repr(k, old)} -> {_display_repr(k, new)}")
                if nd.children_a_only:
                    lines.append(f"    Children only in A: {nd.children_a_only}")
                if nd.children_b_only:
                    lines.append(f"    Children only in B: {nd.children_b_only}")

        if self.identical:
            lines.append("\nGraphs are identical.")
        lines.append(sep)
        return "\n".join(lines)


def _diff_kwargs(kwargs_a: dict, kwargs_b: dict) -> Tuple[dict, dict, dict]:
    """Return ``(added, removed, changed)`` between two kwarg dicts."""
    all_keys = set(kwargs_a) | set(kwargs_b)
    added: dict = {}
    removed: dict = {}
    changed: dict = {}
    for key in sorted(all_keys):
        in_a = key in kwargs_a
        in_b = key in kwargs_b
        if in_a and not in_b:
            removed[key] = kwargs_a[key]
        elif in_b and not in_a:
            added[key] = kwargs_b[key]
        else:
            if kwargs_a[key] is kwargs_b[key]:
                equal = True
            else:
                try:
                    comparison = kwargs_a[key] == kwargs_b[key]
                    equal = bool(comparison)
                except Exception:
                    equal = False
            if not equal:
                changed[key] = (kwargs_a[key], kwargs_b[key])
    return added, removed, changed


def diff_graphs(graph_a: Graph, graph_b: Graph) -> GraphDiff:
    """Compute a structural + configuration diff between two graphs.

    Performs a parallel DFS walk and compares node names, operator classes,
    operator kwargs, and child topology at each corresponding position.
    """
    nodes_a = collect_nodes(graph_a)
    nodes_b = collect_nodes(graph_b)
    names_a = {n.name for n in nodes_a}
    names_b = {n.name for n in nodes_b}

    result = GraphDiff(
        identical=True,
        structural_match=True,
        node_count_a=len(nodes_a),
        node_count_b=len(nodes_b),
        roots_a=[r.name for r in graph_a.roots],
        roots_b=[r.name for r in graph_b.roots],
        nodes_only_in_a=sorted(names_a - names_b),
        nodes_only_in_b=sorted(names_b - names_a),
    )

    if result.nodes_only_in_a or result.nodes_only_in_b:
        result.identical = False
    if len(graph_a.roots) != len(graph_b.roots):
        result.structural_match = False
        result.identical = False

    visited_pairs: Set[Tuple[int, int]] = set()

    def _compare(node_a: Node, node_b: Node, path: str) -> None:
        pair = (id(node_a), id(node_b))
        if pair in visited_pairs:
            return
        visited_pairs.add(pair)

        nd = NodeDiff(position=path, node_a_name=node_a.name, node_b_name=node_b.name)
        has_diff = False

        if node_a.name != node_b.name:
            nd.name_changed = True
            has_diff = True

        cls_a = _qualified_name(node_a.operator_class)
        cls_b = _qualified_name(node_b.operator_class)
        if cls_a != cls_b:
            nd.class_changed = True
            nd.class_a = cls_a
            nd.class_b = cls_b
            has_diff = True

        added, removed, changed = _diff_kwargs(node_a.operator_kwargs, node_b.operator_kwargs)
        if added or removed or changed:
            nd.kwargs_added = added
            nd.kwargs_removed = removed
            nd.kwargs_changed = changed
            has_diff = True

        children_a_names = [c.name for c in node_a.children]
        children_b_names = [c.name for c in node_b.children]
        if children_a_names != children_b_names:
            nd.children_a_only = [n for n in children_a_names if n not in children_b_names]
            nd.children_b_only = [n for n in children_b_names if n not in children_a_names]
            has_diff = True
            result.structural_match = False

        if has_diff:
            result.identical = False
            result.node_diffs.append(nd)

        children_b_map = {c.name: c for c in node_b.children}
        for child_a in node_a.children:
            child_b = children_b_map.get(child_a.name)
            if child_b is not None:
                _compare(child_a, child_b, f"{path} -> {child_a.name}")

    for i, (ra, rb) in enumerate(zip(graph_a.roots, graph_b.roots)):
        _compare(ra, rb, f"root[{i}]/{ra.name}")

    return result


def print_diff(graph_a: Graph, graph_b: Graph) -> None:
    """Print a human-readable diff between two graphs to stdout."""
    print(diff_graphs(graph_a, graph_b).format())


# ---------------------------------------------------------------------------
# Serialization / deserialization
# ---------------------------------------------------------------------------


def _serialize_operator_state(node: Node, *, path: str) -> dict:
    """Serialize one node's operator state without traversing its edges."""
    if not isinstance(node.name, str):
        raise GraphSerializationError(f"{path}.name: expected a string")
    if not isinstance(node.operator_class, type) or not issubclass(node.operator_class, AbstractOperator):
        raise GraphSerializationError(f"{path}.operator_class: expected an AbstractOperator class")
    qualified = _qualified_name(node.operator_class)
    try:
        restored_class = _import_class(qualified)
    except (ImportError, TypeError) as exc:
        raise GraphSerializationError(f"{path}.operator_class: {qualified!r} is not importable") from exc
    if restored_class is not node.operator_class:
        raise GraphSerializationError(f"{path}.operator_class: {qualified!r} does not round-trip by identity")
    if not isinstance(node.operator_kwargs, dict):
        raise GraphSerializationError(f"{path}.operator_kwargs: expected a mapping")

    safe_kwargs: Dict[str, Any] = {}
    for key, value in node.operator_kwargs.items():
        if not isinstance(key, str):
            raise GraphSerializationError(f"{path}.operator_kwargs: kwarg names must be strings")
        safe_kwargs[key] = _encode_value(
            value,
            path=f"{path}.operator_kwargs.{key}",
            field_name=key,
            allow_api_key_env=True,
        )
    return {
        "name": node.name,
        "operator_class": qualified,
        "operator_kwargs": safe_kwargs,
        "children": [],
    }


def _normalized_max_depth(root_ids: Sequence[str], nodes: Dict[str, dict]) -> int:
    if not root_ids:
        return 0
    memo: Dict[str, int] = {}

    def _depth(node_id: str) -> int:
        if node_id not in memo:
            children = nodes[node_id]["children"]
            memo[node_id] = 0 if not children else 1 + max(_depth(child_id) for child_id in children)
        return memo[node_id]

    return max(_depth(root_id) for root_id in root_ids)


def serialize_graph(graph: Graph) -> dict:
    """Serialize a DAG to deterministic, normalized version-2 graph state."""
    if not isinstance(graph, Graph):
        raise GraphSerializationError(f"graph must be a Graph, got {type(graph).__name__}")

    object_ids: Dict[int, str] = {}
    nodes: Dict[str, dict] = {}
    active: List[int] = []
    complete: Set[int] = set()

    def _visit(node: Node, *, path: str) -> str:
        if not isinstance(node, Node):
            raise GraphSerializationError(f"{path}: expected a Node, got {type(node).__name__}")
        object_id = id(node)
        node_id = object_ids.get(object_id)
        if node_id is None:
            node_id = f"node_{len(object_ids)}"
            object_ids[object_id] = node_id
            nodes[node_id] = {}

        if object_id in active:
            cycle_start = active.index(object_id)
            cycle = [object_ids[item] for item in active[cycle_start:]] + [node_id]
            raise GraphSerializationError(f"{path}: graph cycle detected: {' -> '.join(cycle)}")
        if object_id in complete:
            return node_id

        active.append(object_id)
        state = _serialize_operator_state(node, path=f"nodes.{node_id}")
        child_ids = [
            _visit(child, path=f"nodes.{node_id}.children[{index}]") for index, child in enumerate(node.children)
        ]
        if len(set(child_ids)) != len(child_ids):
            raise GraphSerializationError(f"nodes.{node_id}.children: duplicate child node IDs are not allowed")
        state["children"] = child_ids
        nodes[node_id] = state
        active.pop()
        complete.add(object_id)
        return node_id

    root_ids = [_visit(root, path=f"roots[{index}]") for index, root in enumerate(graph.roots)]
    if len(set(root_ids)) != len(root_ids):
        raise GraphSerializationError("roots: duplicate root node IDs are not allowed")
    return {
        "format_version": _GRAPH_FORMAT_VERSION,
        "roots": root_ids,
        "nodes": nodes,
        "metadata": {
            "node_count": len(nodes),
            "max_depth": _normalized_max_depth(root_ids, nodes),
        },
    }


class _PlaceholderOperator(AbstractOperator):
    """Stand-in used when a legacy operator cannot be instantiated."""

    def __init__(
        self,
        original_class: str = "",
        original_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._original_class = original_class
        self._original_kwargs = original_kwargs or {}

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"PlaceholderOperator for {self._original_class!r} cannot process data. "
            f"The original operator class could not be instantiated."
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def _restore_special_values(
    kwargs: dict,
    *,
    format_version: int = 1,
    path: str = "operator_kwargs",
) -> dict:
    """Recursively restore encoded operator kwargs, including v1 markers."""
    cleaned: Dict[str, Any] = {}
    for key, value in kwargs.items():
        decoded = _decode_value(
            value,
            path=f"{path}.{key}",
            format_version=format_version,
            field_name=key,
        )
        if decoded is not _OMIT_FIELD:
            cleaned[key] = decoded
    return cleaned


def _deserialize_v1_node(data: Any, *, path: str) -> Node:
    """Reconstruct one nested node from the versionless legacy format."""
    if not isinstance(data, dict):
        raise GraphSerializationError(f"{path}: expected a node mapping")
    qualified = data.get("operator_class")
    if not isinstance(qualified, str):
        raise GraphSerializationError(f"{path}.operator_class: expected a string")
    try:
        cls = _import_class(qualified)
    except (ImportError, TypeError) as exc:
        raise GraphSerializationError(f"{path}.operator_class: {qualified!r} is not importable") from exc
    raw_kwargs = data.get("operator_kwargs", {})
    if not isinstance(raw_kwargs, dict):
        raise GraphSerializationError(f"{path}.operator_kwargs: expected a mapping")
    cleaned = _restore_special_values(raw_kwargs, format_version=1, path=f"{path}.operator_kwargs")

    try:
        op = cls(**cleaned)
    except Exception:
        op = _PlaceholderOperator(original_class=qualified, original_kwargs=cleaned)

    node = Node(op, name=data.get("name"), operator_class=cls, operator_kwargs=cleaned)
    children = data.get("children", [])
    if not isinstance(children, list):
        raise GraphSerializationError(f"{path}.children: expected a list")
    for index, child_data in enumerate(children):
        node.children.append(_deserialize_v1_node(child_data, path=f"{path}.children[{index}]"))
    return node


def _validate_v2_topology(data: dict) -> Tuple[List[str], Dict[str, dict]]:
    required_top_level = {"format_version", "roots", "nodes"}
    allowed_top_level = required_top_level | {"metadata"}
    top_level_keys = set(data)
    if not required_top_level <= top_level_keys:
        raise GraphSerializationError("version 2 graph is missing required top-level fields")
    if not top_level_keys <= allowed_top_level:
        raise GraphSerializationError("version 2 graph contains unknown top-level fields")

    metadata = data.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict) or set(metadata) != {"node_count", "max_depth"}:
            raise GraphSerializationError("metadata: expected node_count and max_depth")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in metadata.values()):
            raise GraphSerializationError("metadata: node_count and max_depth must be non-negative integers")

    roots = data.get("roots")
    raw_nodes = data.get("nodes")
    if not isinstance(roots, list):
        raise GraphSerializationError("roots: version 2 requires a list of node IDs")
    if not all(isinstance(node_id, str) and node_id for node_id in roots):
        raise GraphSerializationError("roots: every node ID must be a non-empty string")
    if len(set(roots)) != len(roots):
        raise GraphSerializationError("roots: duplicate root node IDs are not allowed")
    if not isinstance(raw_nodes, dict):
        raise GraphSerializationError("nodes: version 2 requires a node-ID mapping")

    required_fields = {"name", "operator_class", "operator_kwargs", "children"}
    nodes: Dict[str, dict] = {}
    for node_id, record in raw_nodes.items():
        path = f"nodes.{node_id}"
        if not isinstance(node_id, str) or not node_id:
            raise GraphSerializationError("nodes: every node ID must be a non-empty string")
        if not isinstance(record, dict):
            raise GraphSerializationError(f"{path}: expected a node mapping")
        missing = required_fields - set(record)
        unknown = set(record) - required_fields
        if missing:
            raise GraphSerializationError(f"{path}: missing fields: {sorted(missing)}")
        if unknown:
            raise GraphSerializationError(f"{path}: unknown fields: {sorted(unknown)}")
        if not isinstance(record["name"], str):
            raise GraphSerializationError(f"{path}.name: expected a string")
        if not isinstance(record["operator_class"], str):
            raise GraphSerializationError(f"{path}.operator_class: expected a string")
        if not isinstance(record["operator_kwargs"], dict):
            raise GraphSerializationError(f"{path}.operator_kwargs: expected a mapping")
        children = record["children"]
        if not isinstance(children, list) or not all(isinstance(child_id, str) and child_id for child_id in children):
            raise GraphSerializationError(f"{path}.children: expected a list of non-empty node IDs")
        if len(set(children)) != len(children):
            raise GraphSerializationError(f"{path}.children: duplicate child node IDs are not allowed")
        nodes[node_id] = record

    for index, root_id in enumerate(roots):
        if root_id not in nodes:
            raise GraphSerializationError(f"roots[{index}]: unknown node ID {root_id!r}")
    for node_id, record in nodes.items():
        for index, child_id in enumerate(record["children"]):
            if child_id not in nodes:
                raise GraphSerializationError(f"nodes.{node_id}.children[{index}]: unknown node ID {child_id!r}")

    state: Dict[str, int] = {}
    reachable: Set[str] = set()
    stack: List[str] = []

    def _visit(node_id: str) -> None:
        marker = state.get(node_id, 0)
        if marker == 1:
            cycle_start = stack.index(node_id)
            cycle = stack[cycle_start:] + [node_id]
            raise GraphSerializationError(f"nodes: graph cycle detected: {' -> '.join(cycle)}")
        if marker == 2:
            return
        state[node_id] = 1
        reachable.add(node_id)
        stack.append(node_id)
        for child_id in nodes[node_id]["children"]:
            _visit(child_id)
        stack.pop()
        state[node_id] = 2

    for root_id in roots:
        _visit(root_id)
    unreachable = sorted(set(nodes) - reachable)
    if unreachable:
        raise GraphSerializationError(f"nodes: unreachable node IDs: {unreachable}")
    if metadata is not None:
        if metadata["node_count"] != len(nodes):
            raise GraphSerializationError("metadata.node_count does not match the normalized graph")
        if metadata["max_depth"] != _normalized_max_depth(roots, nodes):
            raise GraphSerializationError("metadata.max_depth does not match the normalized graph")
    return roots, nodes


def _instantiate_v2_node(node_id: str, record: dict) -> Node:
    path = f"nodes.{node_id}"
    qualified = record["operator_class"]
    try:
        cls = _import_class(qualified)
    except (ImportError, TypeError) as exc:
        raise GraphSerializationError(f"{path}.operator_class: {qualified!r} is not importable") from exc
    if not isinstance(cls, type) or not issubclass(cls, AbstractOperator):
        raise GraphSerializationError(f"{path}.operator_class: {qualified!r} is not an AbstractOperator class")

    cleaned = _restore_special_values(
        record["operator_kwargs"],
        format_version=_GRAPH_FORMAT_VERSION,
        path=f"{path}.operator_kwargs",
    )
    construction_error: Optional[GraphSerializationError] = None
    try:
        op = cls(**cleaned)
    except Exception as exc:
        exc_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
        construction_error = GraphSerializationError(f"{path}: failed to construct operator {qualified!r} ({exc_type})")
        op = None
    if construction_error is not None:
        raise construction_error
    return Node(op, name=record["name"], operator_class=cls, operator_kwargs=cleaned)


def _read_format_version(data: dict) -> int:
    version = data.get("format_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise GraphSerializationError("format_version must be an integer")
    if version not in (1, _GRAPH_FORMAT_VERSION):
        raise GraphSerializationError(f"unsupported graph format_version: {version}")
    return version


def deserialize_graph(data: dict) -> Graph:
    """Load trusted v2 graph state, or a nested versionless-v1 payload.

    Loading imports the recorded operator classes and runs their constructors.
    Only deserialize graph artifacts from a trusted source.
    """
    if not isinstance(data, dict):
        raise GraphSerializationError("serialized graph must be a mapping")
    format_version = _read_format_version(data)
    graph = Graph()
    if format_version == 1:
        roots = data.get("roots", [])
        if not isinstance(roots, list):
            raise GraphSerializationError("roots: expected a list")
        graph.roots = [_deserialize_v1_node(root_data, path=f"roots[{index}]") for index, root_data in enumerate(roots)]
        return graph

    root_ids, records = _validate_v2_topology(data)
    nodes = {node_id: _instantiate_v2_node(node_id, record) for node_id, record in records.items()}
    for node_id, record in records.items():
        nodes[node_id].children = [nodes[child_id] for child_id in record["children"]]
    graph.roots = [nodes[root_id] for root_id in root_ids]
    return graph


def save_graph(graph: Graph, path: Union[str, Path], *, indent: int = 2) -> Path:
    """Serialize *graph* and write it to a JSON file at *path*.

    Returns the resolved :class:`Path` that was written.
    """
    path = Path(path)
    payload = serialize_graph(graph)
    path.write_text(json.dumps(payload, indent=indent))
    return path


def _reject_duplicate_json_keys(pairs: List[Tuple[str, Any]]) -> dict:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GraphSerializationError("serialized graph contains a duplicate JSON object key")
        result[key] = value
    return result


def _load_json_payload(path: Path) -> Any:
    try:
        return json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as exc:
        raise GraphSerializationError(f"invalid graph JSON at line {exc.lineno}, column {exc.colno}") from exc


def load_graph(path: Union[str, Path]) -> Graph:
    """Load trusted graph state from a module or registry graph JSON file."""
    payload = _load_json_payload(Path(path))
    if not isinstance(payload, dict):
        raise GraphSerializationError("serialized graph must be a mapping")
    if "blueprint" in payload:
        if not isinstance(payload["blueprint"], dict):
            raise GraphSerializationError("blueprint metadata must be a mapping")
        payload = {key: value for key, value in payload.items() if key != "blueprint"}
    return deserialize_graph(payload)


def clone_graph(graph: Graph) -> Graph:
    """Create a structural deep-copy of *graph* by round-tripping through serialization.

    This produces new ``Node`` / operator instances so modifications to the
    clone do not affect the original.
    """
    return deserialize_graph(serialize_graph(graph))


# ---------------------------------------------------------------------------
# Blueprint — metadata wrapper for a registered graph
# ---------------------------------------------------------------------------


@dataclass
class GraphBlueprint:
    """A named, versioned graph definition held in the registry."""

    name: str
    graph_factory: Callable[[], Graph]
    description: str = ""
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def build(self) -> Graph:
        """Construct a fresh :class:`Graph` from the stored factory."""
        return self.graph_factory()

    def info(self) -> str:
        """Return a concise multi-line info string (builds the graph once to inspect it)."""
        graph = self.build()
        tag_str = ", ".join(self.tags) if self.tags else "(none)"
        return (
            f"Blueprint: {self.name}\n"
            f"  Version     : {self.version}\n"
            f"  Tags        : {tag_str}\n"
            f"  Description : {self.description}\n"
            f"  Created at  : {self.created_at}\n"
            f"  Updated at  : {self.updated_at}\n"
            f"  {format_graph_summary(graph)}"
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class GraphPipelineRegistry:
    """Central registry for golden pipeline graph definitions.

    Stores :class:`GraphBlueprint` objects keyed by name.  Supports
    decorator and imperative registration, building fresh graph instances,
    inspection / pretty-printing, diffing between graphs, kwarg overrides,
    and JSON serialization / deserialization of the entire registry.

    Usage::

        registry = GraphPipelineRegistry()

        @registry.register("my-pipeline", description="Demo", version="1.0")
        def _build():
            return Graph() >> SomeOperator() >> AnotherOperator()

        graph = registry.build("my-pipeline")
        registry.print_graph("my-pipeline")
    """

    def __init__(self) -> None:
        self._blueprints: OrderedDict[str, GraphBlueprint] = OrderedDict()

    # -- registration -------------------------------------------------------

    def register(
        self,
        name: str,
        *,
        description: str = "",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Callable[[Callable[[], Graph]], Callable[[], Graph]]:
        """Decorator that registers a graph factory function.

        Example::

            @registry.register("pdf-extract", description="PDF extraction pipeline")
            def _build():
                return Graph() >> PDFSplitActor() >> PDFExtractionActor()
        """

        def decorator(factory: Callable[[], Graph]) -> Callable[[], Graph]:
            if name in self._blueprints and not overwrite:
                raise ValueError(f"Graph {name!r} is already registered. Pass overwrite=True to replace it.")
            self._blueprints[name] = GraphBlueprint(
                name=name,
                graph_factory=factory,
                description=description,
                version=version,
                tags=tags or [],
            )
            return factory

        return decorator

    def register_graph(
        self,
        name: str,
        factory: Callable[[], Graph],
        *,
        description: str = "",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """Programmatically register a graph factory (non-decorator form)."""
        if name in self._blueprints and not overwrite:
            raise ValueError(f"Graph {name!r} is already registered. Pass overwrite=True to replace it.")
        self._blueprints[name] = GraphBlueprint(
            name=name,
            graph_factory=factory,
            description=description,
            version=version,
            tags=tags or [],
        )

    def unregister(self, name: str) -> GraphBlueprint:
        """Remove and return the blueprint for *name*.

        Raises ``KeyError`` if *name* is not registered.
        """
        if name not in self._blueprints:
            raise KeyError(f"No graph registered under {name!r}")
        return self._blueprints.pop(name)

    # -- retrieval ----------------------------------------------------------

    def get_blueprint(self, name: str) -> GraphBlueprint:
        """Return the :class:`GraphBlueprint` for *name*.

        Raises ``KeyError`` if not found.
        """
        if name not in self._blueprints:
            raise KeyError(f"No graph registered under {name!r}")
        return self._blueprints[name]

    def build(self, name: str) -> Graph:
        """Build and return a fresh :class:`Graph` from the named blueprint."""
        return self.get_blueprint(name).build()

    def list_names(self) -> List[str]:
        """Return all registered graph names in insertion order."""
        return list(self._blueprints.keys())

    def list_blueprints(self, *, tag: Optional[str] = None) -> List[GraphBlueprint]:
        """Return all blueprints, optionally filtered by *tag*."""
        bps = list(self._blueprints.values())
        if tag is not None:
            bps = [bp for bp in bps if tag in bp.tags]
        return bps

    def __contains__(self, name: str) -> bool:
        return name in self._blueprints

    def __len__(self) -> int:
        return len(self._blueprints)

    def __iter__(self) -> Iterator[str]:
        return iter(self._blueprints)

    def __repr__(self) -> str:
        names = self.list_names()
        return f"GraphPipelineRegistry(graphs={names})"

    # -- inspection ---------------------------------------------------------

    def print_graph(self, name: str, *, show_kwargs: bool = True) -> None:
        """Build and pretty-print the named graph with full details."""
        bp = self.get_blueprint(name)
        print(bp.info())
        print()
        graph = bp.build()
        print(format_graph_tree(graph, show_kwargs=show_kwargs))
        print()
        for node in collect_nodes(graph):
            print(format_node_details(node))
            print()

    def print_summary(self) -> None:
        """Print a compact table of every registered graph."""
        if not self._blueprints:
            print("(registry is empty)")
            return
        header = f"{'Name':35s} {'Version':10s} {'Nodes':>6s} {'Depth':>6s}  {'Tags'}"
        print(header)
        print("-" * len(header))
        for bp in self._blueprints.values():
            graph = bp.build()
            nc = node_count(graph)
            d = max_depth(graph)
            tag_str = ", ".join(bp.tags) if bp.tags else ""
            print(f"{bp.name:35s} {bp.version:10s} {nc:>6d} {d:>6d}  {tag_str}")

    def get_graph_info(self, name: str) -> str:
        """Return the full inspection report for a named graph as a string."""
        graph = self.build(name)
        bp = self.get_blueprint(name)
        return bp.info() + "\n\n" + format_full_report(graph)

    # -- comparison ---------------------------------------------------------

    def diff(self, name_a: str, name_b: str) -> GraphDiff:
        """Build both named graphs and return a :class:`GraphDiff`."""
        return diff_graphs(self.build(name_a), self.build(name_b))

    def print_diff(self, name_a: str, name_b: str) -> None:
        """Print a human-readable diff between two registered graphs."""
        print(self.diff(name_a, name_b).format())

    # -- configuration overrides --------------------------------------------

    def build_with_overrides(self, name: str, overrides: Dict[str, Dict[str, Any]]) -> Graph:
        """Build a graph and apply kwarg overrides to named nodes.

        Parameters
        ----------
        name
            Registered graph name.
        overrides
            ``{node_name: {kwarg_key: new_value, ...}}`` — each matching
            node's ``operator_kwargs`` are updated with the given values.
        """
        graph = self.build(name)
        for node_name, updates in overrides.items():
            update_node_kwargs(graph, node_name, updates, all_matches=True)
        return graph

    # -- serialization (registry-wide) --------------------------------------

    def save_all(self, path: Union[str, Path], *, indent: int = 2) -> Path:
        """Serialize every registered graph to a single JSON file.

        Version 2 stores graphs under a versioned ``graphs`` mapping. Returns
        the resolved path.
        """
        path = Path(path)
        graphs_payload: Dict[str, Any] = {}
        for name, bp in self._blueprints.items():
            graph = bp.build()
            entry = serialize_graph(graph)
            entry["blueprint"] = {
                "name": bp.name,
                "description": bp.description,
                "version": bp.version,
                "tags": bp.tags,
                "created_at": bp.created_at,
                "updated_at": bp.updated_at,
            }
            graphs_payload[name] = entry
        payload = {"format_version": _GRAPH_FORMAT_VERSION, "graphs": graphs_payload}
        path.write_text(json.dumps(payload, indent=indent))
        return path

    def load_all(self, path: Union[str, Path], *, overwrite: bool = False) -> List[str]:
        """Load graphs from a JSON file produced by :meth:`save_all`.

        Each loaded graph is registered as a factory that deserializes the
        stored structure.  Returns the list of graph names loaded.
        """
        path = Path(path)
        payload = _load_json_payload(path)
        if not isinstance(payload, dict):
            raise GraphSerializationError("serialized graph registry must be a mapping")
        version_marker = payload.get("format_version")
        if isinstance(version_marker, int) and not isinstance(version_marker, bool):
            _read_format_version(payload)
            if set(payload) != {"format_version", "graphs"}:
                raise GraphSerializationError("version 2 graph registry contains unknown top-level fields")
            entries = payload.get("graphs")
            if not isinstance(entries, dict):
                raise GraphSerializationError("version 2 graph registry requires a graphs mapping")
        else:
            # Versionless v1 registries were a direct name -> graph mapping,
            # and graph names were unrestricted (including "format_version").
            entries = payload
        prepared: List[Tuple[str, dict, dict]] = []
        for name, entry in entries.items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                raise GraphSerializationError("serialized registry entries must map graph names to graph mappings")
            bp_meta = entry.get("blueprint", {})
            if not isinstance(bp_meta, dict):
                raise GraphSerializationError("blueprint metadata must be a mapping")
            graph_data = {key: value for key, value in entry.items() if key != "blueprint"}
            deserialize_graph(graph_data)
            prepared.append((name, bp_meta, graph_data))

        loaded: List[str] = []
        for name, bp_meta, graph_data in prepared:

            def _factory(_gd: dict = graph_data) -> Graph:
                return deserialize_graph(_gd)

            self.register_graph(
                name,
                _factory,
                description=bp_meta.get("description", ""),
                version=bp_meta.get("version", "1.0.0"),
                tags=bp_meta.get("tags", []),
                overwrite=overwrite,
            )
            restored_bp = self.get_blueprint(name)
            if isinstance(bp_meta.get("created_at"), str):
                restored_bp.created_at = bp_meta["created_at"]
            if isinstance(bp_meta.get("updated_at"), str):
                restored_bp.updated_at = bp_meta["updated_at"]
            loaded.append(name)
        return loaded

    def save_graph(self, name: str, path: Union[str, Path], *, indent: int = 2) -> Path:
        """Serialize a single named graph to a JSON file."""
        graph = self.build(name)
        bp = self.get_blueprint(name)
        payload = serialize_graph(graph)
        payload["blueprint"] = {
            "name": bp.name,
            "description": bp.description,
            "version": bp.version,
            "tags": bp.tags,
            "created_at": bp.created_at,
            "updated_at": bp.updated_at,
        }
        path = Path(path)
        path.write_text(json.dumps(payload, indent=indent))
        return path

    def load_graph(
        self,
        path: Union[str, Path],
        *,
        name: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """Load a single graph from a JSON file and register it.

        If *name* is not provided, the blueprint name stored in the file is
        used (falls back to the file stem).  Returns the registered name.
        """
        path = Path(path)
        payload = _load_json_payload(path)
        if not isinstance(payload, dict):
            raise GraphSerializationError("serialized graph must be a mapping")
        _read_format_version(payload)
        bp_meta = payload.get("blueprint", {})
        if not isinstance(bp_meta, dict):
            raise GraphSerializationError("blueprint metadata must be a mapping")
        graph_data = {k: v for k, v in payload.items() if k != "blueprint"}
        deserialize_graph(graph_data)
        resolved_name = name or bp_meta.get("name") or path.stem

        def _factory(_gd: dict = graph_data) -> Graph:
            return deserialize_graph(_gd)

        self.register_graph(
            resolved_name,
            _factory,
            description=bp_meta.get("description", ""),
            version=bp_meta.get("version", "1.0.0"),
            tags=bp_meta.get("tags", []),
            overwrite=overwrite,
        )
        restored_bp = self.get_blueprint(resolved_name)
        if isinstance(bp_meta.get("created_at"), str):
            restored_bp.created_at = bp_meta["created_at"]
        if isinstance(bp_meta.get("updated_at"), str):
            restored_bp.updated_at = bp_meta["updated_at"]
        return resolved_name


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

default_registry = GraphPipelineRegistry()
