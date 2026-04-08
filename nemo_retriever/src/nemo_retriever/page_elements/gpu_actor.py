# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.gpu_operator import GPUOperator
from nemo_retriever.page_elements.shared import _error_payload, detect_page_elements_v3


class PageElementDetectionActor(AbstractOperator, GPUOperator):
    """
    Ray-friendly callable that initializes Nemotron Page Elements v3 once.

    Use with Ray Data:
      ds = ds.map_batches(PageElementDetectionActor, fn_constructor_kwargs={...}, batch_format="pandas")
    """

    def __init__(self, **detect_kwargs: Any) -> None:
        super().__init__(**detect_kwargs)
        self.detect_kwargs = dict(detect_kwargs)
        invoke_url = str(
            self.detect_kwargs.get("page_elements_invoke_url") or self.detect_kwargs.get("invoke_url") or ""
        ).strip()
        if invoke_url and "invoke_url" not in self.detect_kwargs:
            self.detect_kwargs["invoke_url"] = invoke_url
        if invoke_url:
            self._model = None
        else:
            from nemo_retriever.model.local import NemotronPageElementsV3

            self._model = NemotronPageElementsV3()

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        return detect_page_elements_v3(
            data,
            model=self._model,
            **self.detect_kwargs,
            **kwargs,
        )

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def __call__(self, pages_df: Any, **override_kwargs: Any) -> Any:
        try:
            return self.run(pages_df, **override_kwargs)
        except Exception as exc:
            if isinstance(pages_df, pd.DataFrame):
                out = pages_df.copy()
                payload = _error_payload(stage="actor_call", exc=exc)
                out["page_elements_v3"] = [payload for _ in range(len(out.index))]
                out["page_elements_v3_num_detections"] = [0 for _ in range(len(out.index))]
                out["page_elements_v3_counts_by_label"] = [{} for _ in range(len(out.index))]
                return out
            return [{"page_elements_v3": _error_payload(stage="actor_call", exc=exc)}]
