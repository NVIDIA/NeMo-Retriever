# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write-only Elasticsearch backend: Ray workers send bulk requests, the driver swaps an alias once."""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

import requests

from nemo_retriever.common.vdb.adt_vdb import VDB, UnsupportedVDBOperation
from nemo_retriever.common.vdb.records import stored_row_fields


class Elasticsearch(VDB):
    """Store canonical records behind the ``index_name`` alias.

    Each write fills a new index that searches cannot see; the commit points the
    alias at it, replacing (``overwrite=True``) or joining the indices already
    behind it. Overwrite deletes a replaced index unless another alias still uses it. A failed write deletes its index.
    """

    supports_worker_write = True

    def __init__(
        self,
        url: str = "http://localhost:9200",
        index_name: str = "nemo_retriever",
        overwrite: bool = True,
        worker_write_rows: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.index_name = index_name
        self.overwrite = overwrite
        self.worker_write_rows = worker_write_rows
        self.timeout = timeout
        self._write_index = ""
        self._committed = False

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = requests.request(method, f"{self.url}/{path}", timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response.json()

    @contextmanager
    def worker_write(self) -> Iterator["Elasticsearch"]:
        """Yield a writer copy that fills a new index; leaving without a confirmed commit discards that index."""
        writer = copy.copy(self)
        writer._write_index = f"{self.index_name}-{uuid.uuid4().hex}"
        writer._committed = False
        self._request(
            "PUT",
            writer._write_index,
            json={
                "settings": {"refresh_interval": "-1"},
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "vector": {"type": "dense_vector"},
                        "metadata": {"type": "keyword", "index": False},
                        "source": {"type": "keyword", "index": False},
                    }
                },
            },
        )
        try:
            yield writer
        finally:
            if not writer._committed:
                writer._discard_uncommitted_index()

    def _discard_uncommitted_index(self) -> None:
        # A commit that raised may still have swapped the alias before the error.
        published = requests.get(f"{self.url}/{self._write_index}/_alias/{self.index_name}", timeout=self.timeout)
        if published.status_code == 404:
            deleted = requests.delete(f"{self.url}/{self._write_index}", timeout=self.timeout)
            if deleted.status_code != 404:
                deleted.raise_for_status()

    def write_partition_records(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Bulk-index canonical records into this write's index; return ``{"rows": n}``.

        Raises
        ------
        RuntimeError
            If Elasticsearch rejects any document in the bulk request.
        """
        lines = []
        for record in records:
            fields, _drop_reason = stored_row_fields(record)
            if fields is not None:
                lines += ['{"index":{}}', json.dumps({**fields, "vector": record["metadata"]["embedding"]})]
        if lines:
            result = self._request(
                "POST",
                f"{self._write_index}/_bulk",
                data="\n".join(lines) + "\n",
                headers={"Content-Type": "application/x-ndjson"},
            )
            if result["errors"]:
                failed = next(item["index"]["error"] for item in result["items"] if "error" in item["index"])
                raise RuntimeError(f"Elasticsearch bulk write failed: {failed}")
        return {"rows": len(lines) // 2}

    def commit_partitions(self, receipts: list[dict[str, Any]]) -> None:
        """Refresh this write's index, check its count, and point the alias at it in one step.

        Raises
        ------
        RuntimeError
            If the index holds a different number of documents than the receipts report.
        """
        rows = sum(receipt["rows"] for receipt in receipts)
        self._request("POST", f"{self._write_index}/_refresh")
        # A failed task may have written documents no receipt accounts for.
        stored = self._request("GET", f"{self._write_index}/_count")["count"]
        if stored != rows:
            raise RuntimeError(f"Elasticsearch index holds {stored} documents, receipts report {rows}")
        actions: list[dict[str, Any]] = [{"add": {"index": self._write_index, "alias": self.index_name}}]
        previous: list[str] = []
        if self.overwrite:
            current = requests.get(f"{self.url}/_alias/{self.index_name}", timeout=self.timeout)
            if current.status_code != 404:
                current.raise_for_status()
                previous = list(current.json())
                actions += [{"remove": {"index": index, "alias": self.index_name}} for index in previous]
        self._request("POST", "_aliases", json={"actions": actions})
        self._committed = True
        for index in previous:
            if not self._request("GET", f"{index}/_alias")[index]["aliases"]:
                self._request("DELETE", index)

    def run(self, records: Iterable[Iterable[dict[str, Any]]]) -> None:
        """Write batches of canonical records from the driver through the same index and alias swap."""
        with self.worker_write() as writer:
            receipt = writer.write_partition_records(record for batch in records for record in batch)
            writer.commit_partitions([receipt])

    def create_index(self, **kwargs: Any) -> None:
        """No-op: :meth:`worker_write` creates each write's index."""

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        """Same as :meth:`run`."""
        self.run(records)

    def retrieval(self, queries: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        """Not supported.

        Raises
        ------
        UnsupportedVDBOperation
            Always; use LanceDB to retrieve.
        """
        raise UnsupportedVDBOperation("Elasticsearch is write-only; use LanceDB to retrieve.")

    @staticmethod
    def _collections_not_supported(*args: Any, **kwargs: Any) -> Any:
        raise UnsupportedVDBOperation("Elasticsearch does not implement the collection API.")

    create_collection = get_collection = list_collections = update_collection = delete_collection = (
        _collections_not_supported
    )
    get_document = list_documents = delete_document = write_collection = retrieve_collection = (
        _collections_not_supported
    )
