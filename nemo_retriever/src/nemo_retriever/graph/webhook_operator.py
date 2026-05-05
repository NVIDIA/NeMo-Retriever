# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph operator for posting processed results to a webhook endpoint."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import requests

import pandas as pd

from nemo_retriever.graph.abstract_operator import AbstractOperator
from nemo_retriever.graph.cpu_operator import CPUOperator
from nemo_retriever.graph.designer import designer_component

logger = logging.getLogger(__name__)


def _serialize_value(value: Any) -> Any:
    """Best-effort conversion of a cell value to a JSON-safe type."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    return str(value)


def _dataframe_to_records(df: pd.DataFrame, columns: list[str] | None) -> list[dict[str, Any]]:
    """Convert a DataFrame (or a column subset) to a list of JSON-safe dicts."""
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            logger.warning("WebhookNotifyOperator: requested columns missing from batch: %s", missing)
        available = [c for c in columns if c in df.columns]
        df = df[available] if available else df
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        records.append({k: _serialize_value(v) for k, v in row.items()})
    return records


@designer_component(
    name="Webhook Notify",
    category="I/O & Integration",
    compute="cpu",
    description="HTTP POST processed results to a configurable webhook endpoint",
    category_color="#f5a623",
)
class WebhookNotifyOperator(AbstractOperator, CPUOperator):
    """Post batch results to an external HTTP endpoint.

    This is a **side-effect-only** operator: it sends a JSON payload to a
    remote URL but passes the incoming data through unmodified.  If
    ``endpoint_url`` is ``None`` (the default) the operator is a no-op.

    Parameters
    ----------
    params
        A :class:`~nemo_retriever.params.WebhookParams` instance.  If
        ``None`` or ``params.endpoint_url`` is falsy the stage does nothing.
    """

    def __init__(self, *, params: Any = None) -> None:
        super().__init__()
        self._params = params
        self._session: "requests.Session | None" = None

    def _get_session(self) -> "requests.Session":
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            max_retries = getattr(self._params, "max_retries", 3)
            headers = dict(getattr(self._params, "headers", None) or {})
            headers.setdefault("Content-Type", "application/json")

            session = requests.Session()
            session.headers.update(headers)
            retry_strategy = Retry(
                total=max_retries,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"],
            )
            session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
            session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
            self._session = session

        return self._session

    @property
    def _endpoint_url(self) -> str | None:
        return getattr(self._params, "endpoint_url", None) if self._params else None

    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: Any, **kwargs: Any) -> Any:
        url = self._endpoint_url
        if not url:
            return data

        columns = getattr(self._params, "columns", None) or []
        timeout = getattr(self._params, "timeout_s", 30.0)

        records = _dataframe_to_records(data, columns or None)
        if not records:
            logger.debug("WebhookNotifyOperator: empty batch, skipping POST to %s", url)
            return data

        session = self._get_session()
        try:
            response = session.post(url, json=records, timeout=timeout)
            response.raise_for_status()
            logger.info(
                "WebhookNotifyOperator: POST %d records to %s — %s",
                len(records),
                url,
                response.status_code,
            )
        except Exception:
            logger.exception("WebhookNotifyOperator: failed to POST to %s", url)

        return data

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data
