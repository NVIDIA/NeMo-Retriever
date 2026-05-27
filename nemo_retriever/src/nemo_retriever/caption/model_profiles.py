# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from nemo_retriever.utils.hf_model_registry import get_hf_revision

CaptionTarget = Literal["local", "remote"]

NANO_BF16_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
NANO_FP8_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8"
NANO_NVFP4_QAD_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"
NANO_REMOTE_MODEL_ID = "nvidia/nemotron-nano-12b-v2-vl"

OMNI_BF16_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
OMNI_FP8_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8"
OMNI_NVFP4_MODEL_ID = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4"
OMNI_REMOTE_MODEL_ID = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

_NANO_BF16_REMOTE_MODEL_ID = NANO_REMOTE_MODEL_ID
_NANO_FP8_REMOTE_MODEL_ID = f"{NANO_REMOTE_MODEL_ID}-fp8"
_NANO_NVFP4_QAD_REMOTE_MODEL_ID = f"{NANO_REMOTE_MODEL_ID}-nvfp4-qad"
_OMNI_NO_THINK_EXTRAS = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class CaptionCapabilities:
    """Capabilities currently exposed by the caption pipeline for a model profile."""

    image_captioning: bool = True
    audio_input: bool = False
    video_input: bool = False
    document_intelligence: bool = False
    reasoning_control: bool = False


@dataclass(frozen=True)
class CaptionModelProfile:
    """Immutable caption model metadata for local and remote model resolution."""

    family: str
    variant: str
    local_model_id: str
    remote_model_id: str
    revision: str | None
    capabilities: CaptionCapabilities = field(default_factory=CaptionCapabilities)
    local_request_extras: Mapping[str, Any] = field(default_factory=dict)
    remote_request_extras: Mapping[str, Any] = field(default_factory=dict)
    local_engine_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_request_extras", _freeze_metadata(self.local_request_extras))
        object.__setattr__(self, "remote_request_extras", _freeze_metadata(self.remote_request_extras))
        object.__setattr__(self, "local_engine_kwargs", _freeze_metadata(self.local_engine_kwargs))

    def request_extras_for(self, target: CaptionTarget | str) -> dict[str, Any]:
        """Return mutable default request extras for the local or remote caption target."""

        target = _validate_target(target)
        extras = self.local_request_extras if target == "local" else self.remote_request_extras
        return _mutable_metadata_copy(extras)

    def engine_kwargs_for_local(self) -> dict[str, Any]:
        """Return mutable vLLM engine defaults for local caption model loading."""

        return _mutable_metadata_copy(self.local_engine_kwargs)


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_metadata(item) for item in value)
    return deepcopy(value)


def _mutable_metadata_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_metadata_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_metadata_copy(item) for item in value]
    if isinstance(value, frozenset):
        return {_mutable_metadata_copy(item) for item in value}
    return deepcopy(value)


def merge_request_extras(
    defaults: Mapping[str, Any] | None,
    user: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Deep-merge profile default request extras with user request extras.

    Nested mappings are merged recursively. Default-only keys are preserved, while user values win
    when both inputs define the same key or nested leaf.
    """

    merged = _mutable_metadata_copy(defaults or {})
    for key, value in (user or {}).items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_request_extras(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def supported_caption_model_names(target: CaptionTarget | str = "local") -> tuple[str, ...]:
    """Return supported model IDs and aliases for the local or remote caption target."""

    target = _validate_target(target)
    return _SUPPORTED_NAMES_BY_TARGET[target]


def supported_caption_models_by_variant(target: CaptionTarget | str = "local") -> dict[str, str]:
    """Return canonical target model IDs keyed by model variant label."""

    target = _validate_target(target)
    return {variant: _target_model_id(profile, target) for variant, profile in _PROFILES_BY_VARIANT}


def caption_model_aliases(target: CaptionTarget | str = "local") -> dict[str, str]:
    """Return alias-to-canonical-model mappings for the local or remote caption target."""

    target = _validate_target(target)
    return {alias: _target_model_id(profile, target) for alias, profile in _PROFILE_LOOKUP_BY_TARGET[target].items()}


def caption_model_revisions() -> dict[str, str | None]:
    """Return pinned Hugging Face revisions keyed by local model ID."""

    return {profile.local_model_id: profile.revision for profile in _CAPTION_MODEL_PROFILES}


def get_caption_model_profile(
    name: str,
    *,
    target: CaptionTarget | str | None = None,
    strict: bool = True,
) -> CaptionModelProfile | None:
    """Look up a caption model profile by model ID or alias.

    ``target`` restricts lookup to ``"local"`` or ``"remote"`` model names; ``None`` searches both.
    When ``strict`` is True, unknown names raise ``ValueError``. When False, unknown names return None.
    """

    profile = _get_caption_model_profile(name, target=target)
    if profile is not None or not strict:
        return profile

    target_label = "any target" if target is None else f"target={target!r}"
    supported_names = _supported_names_for_error(target)
    raise ValueError(
        f"Unsupported caption model {name!r} for {target_label}.\n"
        "Supported caption models:\n" + "\n".join(f"  - {supported_name}" for supported_name in supported_names)
    )


def resolve_caption_model_name(name: str, *, target: CaptionTarget | str = "local") -> str:
    """Resolve a model ID or alias to the canonical model name for the target."""

    target = _validate_target(target)
    profile = get_caption_model_profile(name, target=target, strict=False)
    if profile is None:
        return name
    return _target_model_id(profile, target)


def _validate_target(target: CaptionTarget | str) -> CaptionTarget:
    if target not in ("local", "remote"):
        raise ValueError(f"Unsupported caption target {target!r}. Supported targets: 'local', 'remote'.")
    return target


def _normalize_name(name: str) -> str:
    return name.strip().casefold()


def _target_model_id(profile: CaptionModelProfile, target: CaptionTarget) -> str:
    return profile.local_model_id if target == "local" else profile.remote_model_id


def _get_caption_model_profile(
    name: str,
    *,
    target: CaptionTarget | str | None,
) -> CaptionModelProfile | None:
    normalized = _normalize_name(name)
    if target is None:
        return _LOCAL_PROFILE_LOOKUP.get(normalized) or _REMOTE_PROFILE_LOOKUP.get(normalized)

    target = _validate_target(target)
    return _PROFILE_LOOKUP_BY_TARGET[target].get(normalized)


def _supported_names_for_error(target: CaptionTarget | str | None) -> tuple[str, ...]:
    if target is None:
        return _unique(_LOCAL_SUPPORTED_NAMES + _REMOTE_SUPPORTED_NAMES)
    target = _validate_target(target)
    return _SUPPORTED_NAMES_BY_TARGET[target]


def _unique(names: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        if name not in seen:
            result.append(name)
            seen.add(name)
    return tuple(result)


_BF16_ENGINE_KWARGS = {"dtype": "bfloat16"}
# ModelOpt FP8 checkpoints provide the quantization config. These kwargs
# intentionally omit `quantization` and `hf_overrides`.
_MODEL_OPT_FP8_ENGINE_KWARGS = {"dtype": "auto"}
_NVFP4_ENGINE_KWARGS = {"dtype": "auto", "quantization": "modelopt"}

_OMNI_CAPABILITIES = CaptionCapabilities(
    # These describe capabilities exposed by the current caption pipeline,
    # not every modality the underlying Omni model may support.
    image_captioning=True,
    audio_input=False,
    video_input=False,
    document_intelligence=False,
    reasoning_control=True,
)

_NANO_BF16_PROFILE = CaptionModelProfile(
    family="nemotron-nano",
    variant="BF16",
    local_model_id=NANO_BF16_MODEL_ID,
    remote_model_id=_NANO_BF16_REMOTE_MODEL_ID,
    revision=get_hf_revision(NANO_BF16_MODEL_ID, strict=False),
    local_engine_kwargs=_BF16_ENGINE_KWARGS,
)
_NANO_FP8_PROFILE = CaptionModelProfile(
    family="nemotron-nano",
    variant="FP8",
    local_model_id=NANO_FP8_MODEL_ID,
    remote_model_id=_NANO_FP8_REMOTE_MODEL_ID,
    revision=get_hf_revision(NANO_FP8_MODEL_ID, strict=False),
    local_engine_kwargs=_MODEL_OPT_FP8_ENGINE_KWARGS,
)
_NANO_NVFP4_QAD_PROFILE = CaptionModelProfile(
    family="nemotron-nano",
    variant="NVFP4-QAD",
    local_model_id=NANO_NVFP4_QAD_MODEL_ID,
    remote_model_id=_NANO_NVFP4_QAD_REMOTE_MODEL_ID,
    revision=get_hf_revision(NANO_NVFP4_QAD_MODEL_ID, strict=False),
    local_engine_kwargs=_NVFP4_ENGINE_KWARGS,
)
_OMNI_BF16_PROFILE = CaptionModelProfile(
    family="nemotron-3-nano-omni",
    variant="BF16",
    local_model_id=OMNI_BF16_MODEL_ID,
    remote_model_id=OMNI_REMOTE_MODEL_ID,
    revision=get_hf_revision(OMNI_BF16_MODEL_ID, strict=False),
    capabilities=_OMNI_CAPABILITIES,
    local_request_extras=_OMNI_NO_THINK_EXTRAS,
    remote_request_extras=_OMNI_NO_THINK_EXTRAS,
    local_engine_kwargs=_BF16_ENGINE_KWARGS,
)
_OMNI_FP8_PROFILE = CaptionModelProfile(
    family="nemotron-3-nano-omni",
    variant="FP8",
    local_model_id=OMNI_FP8_MODEL_ID,
    remote_model_id=OMNI_REMOTE_MODEL_ID,
    revision=get_hf_revision(OMNI_FP8_MODEL_ID, strict=False),
    capabilities=_OMNI_CAPABILITIES,
    local_request_extras=_OMNI_NO_THINK_EXTRAS,
    remote_request_extras=_OMNI_NO_THINK_EXTRAS,
    local_engine_kwargs=_MODEL_OPT_FP8_ENGINE_KWARGS,
)
_OMNI_NVFP4_PROFILE = CaptionModelProfile(
    family="nemotron-3-nano-omni",
    variant="NVFP4",
    local_model_id=OMNI_NVFP4_MODEL_ID,
    remote_model_id=OMNI_REMOTE_MODEL_ID,
    revision=get_hf_revision(OMNI_NVFP4_MODEL_ID, strict=False),
    capabilities=_OMNI_CAPABILITIES,
    local_request_extras=_OMNI_NO_THINK_EXTRAS,
    remote_request_extras=_OMNI_NO_THINK_EXTRAS,
    local_engine_kwargs=_NVFP4_ENGINE_KWARGS,
)

_CAPTION_MODEL_PROFILES = (
    _NANO_BF16_PROFILE,
    _NANO_FP8_PROFILE,
    _NANO_NVFP4_QAD_PROFILE,
    _OMNI_BF16_PROFILE,
    _OMNI_FP8_PROFILE,
    _OMNI_NVFP4_PROFILE,
)

_PROFILES_BY_VARIANT = (
    ("BF16", _NANO_BF16_PROFILE),
    ("FP8", _NANO_FP8_PROFILE),
    ("NVFP4-QAD", _NANO_NVFP4_QAD_PROFILE),
    ("OMNI-BF16", _OMNI_BF16_PROFILE),
    ("OMNI-FP8", _OMNI_FP8_PROFILE),
    ("OMNI-NVFP4", _OMNI_NVFP4_PROFILE),
)

_NANO_BF16_ALIAS = f"{NANO_REMOTE_MODEL_ID}-bf16"
_OMNI_BF16_ALIAS = f"{OMNI_REMOTE_MODEL_ID}-bf16"
_OMNI_FP8_ALIAS = f"{OMNI_REMOTE_MODEL_ID}-fp8"
_OMNI_NVFP4_ALIAS = f"{OMNI_REMOTE_MODEL_ID}-nvfp4"

_LOCAL_SUPPORTED_NAMES = _unique(
    (
        NANO_BF16_MODEL_ID,
        NANO_FP8_MODEL_ID,
        NANO_NVFP4_QAD_MODEL_ID,
        NANO_REMOTE_MODEL_ID,
        _NANO_BF16_ALIAS,
        _NANO_FP8_REMOTE_MODEL_ID,
        _NANO_NVFP4_QAD_REMOTE_MODEL_ID,
        OMNI_BF16_MODEL_ID,
        OMNI_FP8_MODEL_ID,
        OMNI_NVFP4_MODEL_ID,
        OMNI_REMOTE_MODEL_ID,
        _OMNI_BF16_ALIAS,
        _OMNI_FP8_ALIAS,
        _OMNI_NVFP4_ALIAS,
    )
)
_REMOTE_SUPPORTED_NAMES = _LOCAL_SUPPORTED_NAMES

_LOCAL_PROFILE_LOOKUP = {
    _normalize_name(NANO_BF16_MODEL_ID): _NANO_BF16_PROFILE,
    _normalize_name(NANO_FP8_MODEL_ID): _NANO_FP8_PROFILE,
    _normalize_name(NANO_NVFP4_QAD_MODEL_ID): _NANO_NVFP4_QAD_PROFILE,
    _normalize_name(NANO_REMOTE_MODEL_ID): _NANO_BF16_PROFILE,
    _normalize_name(_NANO_BF16_ALIAS): _NANO_BF16_PROFILE,
    _normalize_name(_NANO_FP8_REMOTE_MODEL_ID): _NANO_FP8_PROFILE,
    _normalize_name(_NANO_NVFP4_QAD_REMOTE_MODEL_ID): _NANO_NVFP4_QAD_PROFILE,
    _normalize_name(OMNI_BF16_MODEL_ID): _OMNI_BF16_PROFILE,
    _normalize_name(OMNI_FP8_MODEL_ID): _OMNI_FP8_PROFILE,
    _normalize_name(OMNI_NVFP4_MODEL_ID): _OMNI_NVFP4_PROFILE,
    _normalize_name(OMNI_REMOTE_MODEL_ID): _OMNI_BF16_PROFILE,
    _normalize_name(_OMNI_BF16_ALIAS): _OMNI_BF16_PROFILE,
    _normalize_name(_OMNI_FP8_ALIAS): _OMNI_FP8_PROFILE,
    _normalize_name(_OMNI_NVFP4_ALIAS): _OMNI_NVFP4_PROFILE,
}
_REMOTE_PROFILE_LOOKUP = {
    _normalize_name(NANO_BF16_MODEL_ID): _NANO_BF16_PROFILE,
    _normalize_name(NANO_FP8_MODEL_ID): _NANO_FP8_PROFILE,
    _normalize_name(NANO_NVFP4_QAD_MODEL_ID): _NANO_NVFP4_QAD_PROFILE,
    _normalize_name(NANO_REMOTE_MODEL_ID): _NANO_BF16_PROFILE,
    _normalize_name(_NANO_BF16_ALIAS): _NANO_BF16_PROFILE,
    _normalize_name(_NANO_FP8_REMOTE_MODEL_ID): _NANO_FP8_PROFILE,
    _normalize_name(_NANO_NVFP4_QAD_REMOTE_MODEL_ID): _NANO_NVFP4_QAD_PROFILE,
    _normalize_name(OMNI_BF16_MODEL_ID): _OMNI_BF16_PROFILE,
    _normalize_name(OMNI_FP8_MODEL_ID): _OMNI_FP8_PROFILE,
    _normalize_name(OMNI_NVFP4_MODEL_ID): _OMNI_NVFP4_PROFILE,
    _normalize_name(OMNI_REMOTE_MODEL_ID): _OMNI_BF16_PROFILE,
    _normalize_name(_OMNI_BF16_ALIAS): _OMNI_BF16_PROFILE,
    _normalize_name(_OMNI_FP8_ALIAS): _OMNI_FP8_PROFILE,
    _normalize_name(_OMNI_NVFP4_ALIAS): _OMNI_NVFP4_PROFILE,
}
_PROFILE_LOOKUP_BY_TARGET = {
    "local": _LOCAL_PROFILE_LOOKUP,
    "remote": _REMOTE_PROFILE_LOOKUP,
}
_SUPPORTED_NAMES_BY_TARGET = {
    "local": _LOCAL_SUPPORTED_NAMES,
    "remote": _REMOTE_SUPPORTED_NAMES,
}

__all__ = [
    "CaptionCapabilities",
    "CaptionModelProfile",
    "CaptionTarget",
    "NANO_BF16_MODEL_ID",
    "NANO_FP8_MODEL_ID",
    "NANO_NVFP4_QAD_MODEL_ID",
    "NANO_REMOTE_MODEL_ID",
    "OMNI_BF16_MODEL_ID",
    "OMNI_FP8_MODEL_ID",
    "OMNI_NVFP4_MODEL_ID",
    "OMNI_REMOTE_MODEL_ID",
    "caption_model_aliases",
    "caption_model_revisions",
    "get_caption_model_profile",
    "merge_request_extras",
    "resolve_caption_model_name",
    "supported_caption_model_names",
    "supported_caption_models_by_variant",
]
