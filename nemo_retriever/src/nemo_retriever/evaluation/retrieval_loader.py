# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RetrievalLoaderOperator -- source operator for evaluation chains."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import pandas as pd

from nemo_retriever.evaluation.eval_operator import EvalOperator

logger = logging.getLogger(__name__)


class RetrievalLoaderOperator(EvalOperator):
    """Load retrieval JSON and ground truth CSV into a DataFrame.

    Acts as the first stage (source) in a ``>>`` evaluation chain.
    If *data* passed to ``process()`` is already a non-empty DataFrame,
    it is returned as-is (pass-through for pre-loaded data).

    Produces a DataFrame with columns: ``query``, ``reference_answer``,
    ``context``.  Each row is a ground-truth query joined with its
    pre-computed retrieval chunks.

    Constructor kwargs are all serializable strings, so
    ``get_constructor_kwargs()`` works for Ray reconstruction.
    """

    required_columns: ClassVar[tuple[str, ...]] = ()
    output_columns: ClassVar[tuple[str, ...]] = ("query", "reference_answer", "context")

    def __init__(
        self,
        retrieval_json: str,
        ground_truth_csv: str,
        *,
        data_dir: str | None = None,
        query_column: str = "query",
        answer_column: str = "answer",
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            retrieval_json=str(retrieval_json),
            ground_truth_csv=str(ground_truth_csv),
            data_dir=data_dir,
            query_column=query_column,
            answer_column=answer_column,
            top_k=top_k,
            **kwargs,
        )
        self._retrieval_json = str(retrieval_json)
        self._ground_truth_csv = str(ground_truth_csv)
        self._data_dir = data_dir
        self._query_column = query_column
        self._answer_column = answer_column
        self._top_k = top_k

    def process(self, data: Any, **kwargs: Any) -> Any:
        if isinstance(data, pd.DataFrame) and not data.empty:
            return data

        from nemo_retriever.evaluation.ground_truth import get_qa_dataset_loader, load_generic_csv
        from nemo_retriever.evaluation.retrievers import FileRetriever

        source = self._ground_truth_csv
        try:
            loader_fn = get_qa_dataset_loader(source)
        except ValueError:
            qa_pairs = load_generic_csv(source)
        else:
            qa_pairs = loader_fn(self._data_dir)
        retriever = FileRetriever(file_path=self._retrieval_json)

        coverage = retriever.check_coverage(qa_pairs)
        if coverage < 0.5:
            logger.warning(
                "Retrieval file covers only %.0f%% of queries -- results may be unreliable",
                coverage * 100,
            )

        rows: list[dict] = []
        for pair in qa_pairs:
            query = pair.get(self._query_column, "")
            reference = pair.get(self._answer_column, "")
            if not query:
                continue
            hit_df = retriever.retrieve(query, self._top_k)
            # FileRetriever.retrieve now returns a single-row DataFrame
            # with [query, chunks, metadata]; guarded empty-frame access
            # so missing queries (graceful-miss path) produce an empty
            # context list rather than an IndexError.
            chunks = hit_df.iloc[0]["chunks"] if len(hit_df) else []
            rows.append(
                {
                    "query": query,
                    "reference_answer": reference,
                    "context": chunks,
                }
            )

        if not rows:
            logger.warning("RetrievalLoaderOperator produced 0 rows")
            return pd.DataFrame(columns=["query", "reference_answer", "context"])

        return pd.DataFrame(rows)
