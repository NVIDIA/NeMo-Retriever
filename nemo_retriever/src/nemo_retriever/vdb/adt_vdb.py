"""Abstract Vector Database (VDB) operator API.

Defines the `VDB` abstract base class — the small interface that custom
vector-database operators implement to plug into NeMo Retriever.

The interface separates ingestion from retrieval so the same ABC works for
both halves of the pipeline:

- `create_index` / `write_to_index` / `run` — index lifecycle and bulk
  ingestion of Nemo Retriever Library (NRL) record batches.
- `retrieval` — nearest-neighbor search over **precomputed query vectors**.
  Query strings are embedded upstream (see `nemo_retriever.Retriever`);
  the VDB search boundary receives vectors as the primary input. Backends
  that combine dense and lexical evidence may also receive aligned
  `query_texts` as execution-only retrieval context.

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
        one per query. Query text is embedded upstream, typically inside
        `nemo_retriever.Retriever`, before this method is called. Backends
        that need the raw query string for retrieval-time lexical matching
        may additionally consume `query_texts` from `kwargs`; those strings
        must be aligned one-to-one with the input vectors.

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
        - query_texts: raw query strings aligned with the input vectors.
          Dense retrieval backends should not require this. LanceDB hybrid
          search requires it because the backend combines the precomputed
          dense vector with a full-text query at search time.
        - refine_factor / nprobes / search_kwargs: ANN tuning passed
          through to the backend.

        See `nemo_retriever/vdb/README.md` and
        `examples/nemo_retriever_retriever_query_metadata_filter.ipynb`
        for the full filter cookbook (sidecar merge, server-side vs
        client-side filtering, escaping).

        `query_texts` is execution-only context. Operators should avoid
        persisting it in backend constructor kwargs and pass it only for
        retrieval calls whose effective mode needs raw text.
        """
        pass

    def put(self, records: list, **kwargs: Any) -> dict[str, Any]:
        """Replace a batch of existing rows in the target table/index.

        Note: this method is intentionally **not** decorated with
        :func:`abc.abstractmethod`. Marking it abstract would cause
        Python's ABC machinery to refuse instantiation of any concrete
        :class:`VDB` subclass that does not override ``put`` — which
        would in turn make the early-detection guard in
        :class:`~nemo_retriever.vdb.operators.PutVdbOperator` (which
        compares ``type(self._vdb).put is VDB.put``) permanently
        unreachable, since instantiation would already have failed.
        The default body below raises :class:`NotImplementedError` so
        backends that have not implemented stable-key puts fail fast
        and visibly at the first ``put`` call (and are caught by the
        operator-level guard at construction time).

        ``put`` exists as a separate entry point from
        :meth:`write_to_index` because it has fundamentally different
        semantics. Where ``write_to_index`` is an *append* (or full
        ingest) operation, ``put`` is a **strict in-place replace**:

        * Rows whose key value already exists in the target table are
          **updated in place** — all stored columns (including the dense
          vector) are replaced with the values from ``records``.
        * Rows whose key value does not exist in the target table
          MUST raise :class:`KeyError`. ``put`` MUST NOT insert new
          rows; ingestion of new rows belongs in
          :meth:`write_to_index` / :meth:`run`.
        * Rows whose key value is empty or ``None`` MUST raise
          :class:`KeyError` — a put has no stable identity to target
          without a key.
        * Rows that already exist in the target but are *not* referenced
          by ``records`` are **left untouched**. ``put`` MUST NOT
          delete rows.

        This contract makes ``put`` suitable for in-place metadata
        patches where the caller knows the exact set of existing rows
        it wants to change and would rather fail loudly than silently
        no-op or duplicate data.

        Implementations are expected to:

        * Validate / transform records the same way :meth:`write_to_index`
          does (e.g. enforce the embedding dimension, apply the
          ``on_bad_vectors`` policy), so that a put row is
          indistinguishable from one written via the full-ingest path.
        * Raise :class:`FileNotFoundError` (or an equivalent) when the
          target table does not yet exist. ``put`` MUST NOT create
          tables on the fly.
        * Avoid building heavy secondary structures (e.g. IVF/HNSW
          vector indexes, FTS indexes) on the put path: incremental
          batches are typically too small to train such indexes
          meaningfully. Defer index builds to the next full
          :meth:`write_to_index` / :meth:`create_index` call.

        Parameters:
        - records (list): NV-Ingest-shaped batches (typically a list of
            lists of record dicts) to put into the target. The shape
            mirrors what :meth:`write_to_index` accepts.
        - table_name (str, optional): override the operator's configured
            target table/index name for this call. When ``None``, the
            implementation should use its default target.
        - key (str, optional): name of the column used as the stable
            put key. Defaults to ``"id"``. Rows missing this column
            (or with an empty value) MUST raise :class:`KeyError`.

        Returns:
        - implementation-specific result describing what happened
            (typical fields include the number of rows put).
            Concrete implementations should document the exact return
            shape.

        Backends that genuinely cannot support stable-key puts should
        override this method and raise :class:`NotImplementedError`
        explicitly so that :class:`PutVdbOperator` (and any other
        caller) fails fast with a clear message instead of silently
        no-oping or duplicating rows.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement put(); "
            "in-place stable-key puts are not supported by this VDB backend."
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
