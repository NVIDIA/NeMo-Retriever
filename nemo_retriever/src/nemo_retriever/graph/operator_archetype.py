# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.utils.ray_resource_hueristics import ClusterResources, Resources, gather_local_resources


def _available_gpu_count(resources: ClusterResources | Resources) -> int:
    if isinstance(resources, ClusterResources):
        return int(resources.available_gpu_count())
    return int(resources.gpu_count)


class ArchetypeOperator(AbstractOperator):
    """Lightweight graph-facing operator that resolves to a hardware-specific variant."""

    _cpu_variant_class: type[AbstractOperator] | None = None
    _gpu_variant_class: type[AbstractOperator] | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._resolved_delegate: AbstractOperator | None = None
        self._resolved_delegate_key: tuple[int, int] | None = None

    @classmethod
    def prefers_cpu_variant(cls, operator_kwargs: dict[str, Any] | None = None) -> bool:
        return False

    @classmethod
    def resolve_operator_class(
        cls,
        resources: ClusterResources | Resources | None = None,
        operator_kwargs: dict[str, Any] | None = None,
    ) -> type[AbstractOperator]:
        detected = resources or gather_local_resources()
        if cls.prefers_cpu_variant(operator_kwargs or {}) and cls._cpu_variant_class is not None:
            return cls._cpu_variant_class
        available_gpus = _available_gpu_count(detected)
        if available_gpus > 0 and cls._gpu_variant_class is not None:
            return cls._gpu_variant_class
        if cls._cpu_variant_class is not None:
            return cls._cpu_variant_class
        if cls._gpu_variant_class is not None:
            return cls._gpu_variant_class
        return cls

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return self._resolve_delegate().preprocess(data, **kwargs)

    def process(self, data: Any, **kwargs: Any) -> Any:
        return self._resolve_delegate().process(data, **kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return self._resolve_delegate().postprocess(data, **kwargs)

    def run(self, data: Any, **kwargs: Any) -> Any:
        return self._resolve_delegate().run(data, **kwargs)

    def __call__(self, data: Any, **kwargs: Any) -> Any:
        return self._resolve_delegate()(data, **kwargs)

    def _resolve_delegate(self, resources: ClusterResources | Resources | None = None) -> AbstractOperator:
        if not hasattr(self, "_resolved_delegate"):
            self._resolved_delegate = None
            self._resolved_delegate_key = None
        detected = resources or gather_local_resources()
        cache_key = _delegate_cache_key(detected)
        if self._resolved_delegate is not None and self._resolved_delegate_key == cache_key:
            return self._resolved_delegate

        operator_kwargs = self.get_constructor_kwargs()
        operator_class = type(self).resolve_operator_class(detected, operator_kwargs=operator_kwargs)
        if operator_class is type(self):
            raise RuntimeError(f"{type(self).__name__} could not resolve a concrete hardware-specific operator.")

        delegate = operator_class(**operator_kwargs)
        self._resolved_delegate = delegate
        self._resolved_delegate_key = cache_key
        return delegate


def _delegate_cache_key(resources: ClusterResources | Resources) -> tuple[int, int]:
    if isinstance(resources, ClusterResources):
        return (
            int(resources.available_cpu_count()),
            int(resources.available_gpu_count()),
        )
    return (int(resources.cpu_count), int(resources.gpu_count))
