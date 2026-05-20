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
from typing import Any


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

    def upsert(self, records: list, **kwargs: Any) -> dict[str, Any]:
        """Incrementally merge a batch of records into the target table/index.

        Note: this method is intentionally **not** decorated with
        :func:`abc.abstractmethod`. Marking it abstract would cause
        Python's ABC machinery to refuse instantiation of any concrete
        :class:`VDB` subclass that does not override ``upsert`` — which
        would in turn make the early-detection guard in
        :class:`~nemo_retriever.vdb.operators.UpsertVdbOperator` (which
        compares ``type(self._vdb).upsert is VDB.upsert``) permanently
        unreachable, since instantiation would already have failed.
        The default body below raises :class:`NotImplementedError` so
        backends that have not implemented stable-key merges fail fast
        and visibly at the first ``upsert`` call (and are caught by the
        operator-level guard at construction time).

        ``upsert`` exists as a separate entry point from
        :meth:`write_to_index` because it has fundamentally different
        semantics. Where ``write_to_index`` is an *append* (or full
        ingest) operation, ``upsert`` is a **stable-key merge**:

        * Rows whose key value already exists in the target table are
          **updated in place** — all stored columns (including the dense
          vector) are replaced with the values from ``records``.
        * Rows whose key value is absent are **inserted**.
        * Rows that already exist in the target but are *not* referenced
          by ``records`` are **left untouched**. ``upsert`` MUST NOT
          delete rows; tombstoning entities that have disappeared
          upstream is intentionally out of scope and belongs in a
          separate code path.

        This contract makes ``upsert`` suitable for incremental metadata
        patches and partial re-ingests where the caller knows the stable
        identity of the rows it is changing but does not want to rebuild
        the whole index.

        Implementations are expected to:

        * Validate / transform records the same way :meth:`write_to_index`
          does (e.g. enforce the embedding dimension, apply the
          ``on_bad_vectors`` policy), so that an upserted row is
          indistinguishable from one written via the full-ingest path.
        * Drop rows whose ``key`` value is empty or ``None`` — an empty
          merge key has no stable identity and would otherwise collapse
          unrelated rows together. Skipped rows should be logged.
        * Create the target table/index on the fly when it does not yet
          exist (e.g. a metadata patch lands before the first full
          ingest), if the backend supports it. Race-tolerance is
          recommended: if a parallel writer wins the create, fall back
          to opening the existing table and performing the merge.
        * Avoid building heavy secondary structures (e.g. IVF/HNSW
          vector indexes, FTS indexes) on the upsert path: incremental
          batches are typically too small to train such indexes
          meaningfully. Defer index builds to the next full
          :meth:`write_to_index` / :meth:`create_index` call.

        Parameters:
        - records (list): NV-Ingest-shaped batches (typically a list of
            lists of record dicts) to merge into the target. The shape
            mirrors what :meth:`write_to_index` accepts.
        - table_name (str, optional): override the operator's configured
            target table/index name for this call. When ``None``, the
            implementation should use its default target.
        - key (str, optional): name of the column used as the stable
            merge key. Defaults to ``"id"``. Rows missing this column
            (or with an empty value) should be skipped.

        Returns:
        - implementation-specific result describing what happened
            (typical fields include the number of rows merged, the
            number of rows skipped for missing keys, and whether the
            target table had to be created on the fly). Concrete
            implementations should document the exact return shape.

        Backends that genuinely cannot support stable-key merges should
        override this method and raise :class:`NotImplementedError`
        explicitly so that :class:`UpsertVdbOperator` (and any other
        caller) fails fast with a clear message instead of silently
        no-oping or duplicating rows.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement upsert(); "
            "incremental stable-key merges are not supported by this VDB backend."
        )

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
