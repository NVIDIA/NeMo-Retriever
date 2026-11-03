# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU stages around local GPU page-element detection."""

from __future__ import annotations

from typing import Any

import nvtx

from nemo_retriever.common.modality.page_elements.shared import _error_payload
from nemo_retriever.common.modality.page_elements.split import (
    ERR_COLUMN,
    PRE_COLUMNS,
    _dump_err,
    out_columns,
    split_postprocess,
    split_preprocess,
)
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.cpu_operator import CPUOperator


class PageElementPreprocessCPU(AbstractOperator, CPUOperator):
    """Decode and resize/pad pages for the page-element model; no GPU."""

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @nvtx.annotate("nrl.batch::page_elements.pre_cpu", color="cyan")
    def process(self, data: Any, **kwargs: Any) -> Any:
        return split_preprocess(data)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, pages_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(pages_df, **override_kwargs)
        except Exception as exc:
            out = pages_df.copy()
            for col in PRE_COLUMNS:
                out[col] = [None] * len(out.index)
            out[ERR_COLUMN] = [_dump_err(_error_payload(stage="actor_call", exc=exc))] * len(out.index)
            return out


class PageElementPostprocessCPU(AbstractOperator, CPUOperator):
    """Turn raw page-element boxes into the ``page_elements_v3`` payload columns; no GPU."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._kwargs = dict(kwargs)

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    @nvtx.annotate("nrl.batch::page_elements.post_cpu", color="cyan")
    def process(self, data: Any, **kwargs: Any) -> Any:
        return split_postprocess(data, **self._kwargs, **kwargs)

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, pages_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(pages_df, **override_kwargs)
        except Exception as exc:
            out = pages_df.drop(columns=[c for c in out_columns() if c in pages_df.columns])
            payload = _error_payload(stage="actor_call", exc=exc)
            out["page_elements_v3"] = [payload for _ in range(len(out.index))]
            out["page_elements_v3_num_detections"] = [0 for _ in range(len(out.index))]
            out["page_elements_v3_counts_by_label"] = [{} for _ in range(len(out.index))]
            return out
