"""Abstract Vector Database (VDB) operator API.

Defines the `VDB` abstract base class — the small interface that custom
vector-database operators implement to plug into NeMo Retriever.

The interface separates ingestion from retrieval so the same ABC works for
both halves of the pipeline:

- `create_index` / `write_to_index` / `run` — index lifecycle and bulk
  ingestion of Nemo Retriever Library (NRL) record batches.
- `retrieval` — nearest-neighbor search over **precomputed query vectors**.
  Query strings are embedded upstream (see `nemo_retriever.Retriever`);
  the VDB only sees vectors.

Methods accept `**kwargs` so backend-specific options (e.g. LanceDB's
`where` predicate for metadata filtering, refinement factors,
hybrid-search flags) flow through without changing the ABC.

See `nemo_retriever/vdb/README.md` for the concrete `LanceDB` backend and
the `IngestVdbOperator` / `RetrieveVdbOperator` wrappers, including the
metadata-filtering section and its reference notebook.
"""

from abc import ABC, abstractmethod


class VDB(ABC):
    """Abstract base class for vector-database operators.

    Subclasses implement the four abstract methods below. The interface is
    intentionally small; backend-specific options (connection URIs, index
    tuning, search filters) are passed via `**kwargs`.

    The reference implementation is `LanceDB` (see `lancedb.py`). For an
    overview of how `IngestVdbOperator` and `RetrieveVdbOperator` consume
    this interface, see the package README.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the operator.

        Implementations parse backend-specific connection and index
        parameters from `kwargs` and set up any client handles. Heavy
        operations (creating indexes, loading data) belong in
        `create_index`, not here, so the operator stays cheap to
        construct in tests.

        Common kwargs vary by backend. For LanceDB, for example:
        `uri`, `table_name`, `vector_dim`, `overwrite`, `index_type`,
        `metric`, `num_partitions`, `num_sub_vectors`, `hybrid`,
        `on_bad_vectors`.

        The base class stores all kwargs as attributes on the instance as
        a convenience; subclasses may rely on that or override.
        """
        self.__dict__.update(kwargs)

    @abstractmethod
    def create_index(self, **kwargs):
        """Create the index(es) needed for ingestion and retrieval.

        Implementations create the table / index with the appropriate
        vector schema (dimension, distance metric, ANN parameters) and any
        auxiliary indexes (e.g. an FTS index for hybrid search).

        Common kwargs:
        - recreate (bool): drop and recreate even if the index exists.

        Return value is backend-specific.
        """
        pass

    @abstractmethod
    def write_to_index(self, records: list, **kwargs):
        """Write a batch of NRL record batches to the index.

        `records` is a list of record batches — each batch is a list of
        record dicts as produced by the NRL pipeline. Implementations
        transform each record into the table's row format (typically
        columns `vector`, `text`, `metadata`, `source`) and use the
        backend's bulk-write API.

        Sidecar metadata (when supplied via `meta_dataframe` /
        `meta_source_field` / `meta_fields` at operator construction) is
        merged into each record's `content_metadata` upstream of this
        method — implementations only see the merged result.

        Records missing required fields (vector, text) should be skipped
        rather than raised, matching the reference `LanceDB` backend's
        `on_bad_vectors` behavior.

        Common kwargs:
        - batch_size (int): documents per bulk request.
        """
        pass

    @abstractmethod
    def retrieval(self, queries: list, **kwargs):
        """Run nearest-neighbor search for **precomputed query vectors**.

        Despite the parameter name `queries` (kept for backward
        compatibility), this method receives a list of embedding vectors,
        one per query — *not* raw text. Query text is embedded upstream,
        typically inside `nemo_retriever.Retriever`, before this method
        is called.

        Implementations search the index, apply any post-filtering, and
        return a list of hit lists aligned with the input (one inner list
        per input vector). Stored vector columns should be stripped from
        hits to keep payloads small.

        Common kwargs:
        - top_k (int): neighbors per query.
        - where / _filter (str): a SQL predicate evaluated against table
          columns. NRL stores `content_metadata` (including sidecar
          fields) as a **compact JSON string** in the `metadata` column,
          so JSON filters typically use `LIKE` against a substring of the
          serialized JSON, e.g.
          `metadata LIKE '%"meta_a":"alpha"%'`.
          The `_filter` alias is accepted in addition to `where`.
        - refine_factor / nprobes / search_kwargs: ANN tuning passed
          through to the backend.

        See `nemo_retriever/vdb/README.md` and
        `examples/nemo_retriever_retriever_query_metadata_filter.ipynb`
        for the full filter cookbook (sidecar merge, server-side vs
        client-side filtering, escaping).

        Hybrid search with precomputed vectors is not implemented by the
        reference `LanceDB` backend; passing `hybrid=True` raises
        `NotImplementedError` on that path.
        """
        pass

    @abstractmethod
    def run(self, records):
        """Pipeline entry point: ensure the index exists, then ingest.

        Minimal implementation::

            def run(self, records):
                self.create_index()
                self.write_to_index(records)

        Implementers may add metrics, retries, or commit hooks, but
        `run` should stay a thin orchestration layer so callers can
        reason about ingestion order.
        """
        pass

    def reindex(self, records: list, **kwargs):
        """Drop and rebuild the index, then re-ingest `records`.

        Optional hook for subclasses. Default implementation does nothing;
        a typical override is::

            def reindex(self, records, **kwargs):
                self.create_index(recreate=True)
                self.write_to_index(records)
        """
        pass
