# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Sequence

from nemo_retriever.ingest.plan import ResolvedIngestPlan
from nemo_retriever.ingestor.manifest import format_branch_summary
from nemo_retriever.ingestor import Ingestor, create_ingestor
from nemo_retriever.common.vdb.records import to_sparse_client_vdb_records
from nemo_retriever.common.vdb.targets import VdbTarget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestExecutionResult:
    """Structured result from executing a resolved ingest plan.

    ``result`` is the raw value returned by ``Ingestor.ingest()`` and may be a
    large dataframe-like object. ``result_n_rows`` counts that current result
    when possible. ``initial_n_rows`` and ``n_rows`` are index row counts
    before and after execution when row verification is enabled.
    ``run_metadata`` holds execution-level details, not per-chunk metadata.
    """

    plan: ResolvedIngestPlan
    result: object
    n_rows: int | None
    result_n_rows: int | None
    initial_n_rows: int | None
    lancedb_uri: str
    table_name: str
    run_metadata: dict[str, Any]
    vdb_target: VdbTarget

    @property
    def documents(self) -> list[str]:
        return list(self.plan.documents)

    @property
    def lancedb_target(self) -> str:
        return f"{self.lancedb_uri}/{self.table_name}"

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "n_documents": len(self.plan.documents),
            "vdb_op": self.vdb_target.vdb_op,
            "vdb_target": self.vdb_target.describe(),
            "lancedb_uri": self.lancedb_uri,
            "table_name": self.table_name,
            "n_rows": self.n_rows,
            "result_n_rows": self.result_n_rows,
        }


def build_ingest_pipeline(plan: ResolvedIngestPlan) -> Ingestor:
    """Build the SDK ingest chain from a resolved plan without executing it.

    This is the shared implementation used by root ``retriever ingest`` and
    development callers that need the same manifest-routed extract/embed path.
    """

    extract_kwargs = plan.extract_call_kwargs()
    if plan.split_config is not None:
        extract_kwargs["split_config"] = plan.split_config

    ingestor = create_ingestor(**plan.create_kwargs).files(plan.documents)
    ingestor = ingestor.extract(plan.extract_params, **extract_kwargs)
    if plan.dedup_params is not None:
        ingestor = ingestor.dedup(plan.dedup_params)

    if plan.caption_params is not None:
        ingestor = ingestor.caption(plan.caption_params)

    if not plan.sparse:
        ingestor = ingestor.embed(plan.embed_params) if plan.embed_params is not None else ingestor.embed()
    if plan.store_params is not None:
        ingestor = ingestor.store(plan.store_params)

    if plan.vdb_params is not None and not plan.sparse:
        ingestor = ingestor.vdb_upload(plan.vdb_params)
    return ingestor


def execute_ingest_plan(
    plan: ResolvedIngestPlan,
    *,
    verify_rows: bool = True,
    raise_on_empty: bool = True,
) -> IngestExecutionResult:
    """Execute a resolved ingest plan and return structured execution data.

    Args:
        plan: Fully resolved ingest graph options.
        verify_rows: When true, use the VDB upload target as the success check.
            Verification counts rows returned by the current ingest result when
            possible, then counts rows in the target index after upload. Disable this
            for extraction/embed-only plans that intentionally omit VDB upload.
        raise_on_empty: When true, raise if verification proves the current run
            produced no rows or failed to add rows to an append target.
    """

    resolved_target = _resolve_vdb_target(plan)
    if verify_rows and resolved_target is None:
        raise ValueError(
            "Row verification checks the effective VDB upload target; "
            "pass verify_rows=False for extraction/embed-only plans."
        )
    target, overwrite = resolved_target or (plan.vdb_target, True)

    initial_n_rows = None
    if verify_rows and not overwrite:
        initial_n_rows = _count_index_rows(target)

    result = build_ingest_pipeline(plan).ingest()
    if plan.sparse:
        vdb_kwargs = plan.vdb_params.vdb_kwargs if plan.vdb_params else {}
        target.backend(**{**vdb_kwargs, "overwrite": overwrite, "sparse": True}).run(
            to_sparse_client_vdb_records(result)
        )

    result_n_rows = _count_result_rows(result) if verify_rows else None
    n_rows = _count_index_rows(target) if verify_rows else None
    if verify_rows and raise_on_empty:
        _raise_for_empty_ingest(
            documents=plan.documents,
            target=target,
            n_rows=n_rows,
            result_n_rows=result_n_rows,
            initial_n_rows=initial_n_rows,
        )

    return IngestExecutionResult(
        plan=plan,
        result=result,
        n_rows=n_rows,
        result_n_rows=result_n_rows,
        initial_n_rows=initial_n_rows,
        lancedb_uri=target.lancedb_uri,
        table_name=target.table_name,
        run_metadata={
            "lancedb_target": f"{target.lancedb_uri}/{target.table_name}",
            "vdb_target": target.describe(),
            "profile": plan.profile,
            "branch_summary": format_branch_summary(plan.branches),
        },
        vdb_target=target,
    )


def _resolve_vdb_target(plan: ResolvedIngestPlan) -> tuple[VdbTarget, bool] | None:
    if plan.vdb_params is None:
        return None
    vdb_kwargs = dict(plan.vdb_params.vdb_kwargs)
    target = replace(
        plan.vdb_target,
        vdb_op=plan.vdb_params.vdb_op,
        lancedb_uri=str(vdb_kwargs.get("uri") or vdb_kwargs.get("lancedb_uri") or plan.lancedb_uri),
        table_name=str(
            vdb_kwargs.get("table_name")
            or vdb_kwargs.get("lancedb_table")
            or vdb_kwargs.get("collection_name")
            or plan.table_name
        ),
    )
    return target, bool(vdb_kwargs.get("overwrite", True))


def _raise_for_empty_ingest(
    *,
    documents: Sequence[str],
    target: VdbTarget,
    n_rows: int | None,
    result_n_rows: int | None,
    initial_n_rows: int | None,
) -> None:
    backend = target.backend_name
    described = target.describe()
    if result_n_rows == 0:
        raise RuntimeError(
            f"retriever ingest produced 0 rows before {backend} write for {len(documents)} input file(s). "
            f"{described} may still contain rows from an earlier run; check the captured stage logs above, "
            "and verify NVIDIA_API_KEY/NGC_API_KEY or the configured local/remote endpoints."
        )
    if n_rows is None:
        raise RuntimeError(
            f"retriever ingest could not verify rows in {described} for {len(documents)} input file(s). "
            f"This usually means the {backend} {target.index_noun} was not created or could not be read after "
            "ingestion; check the captured stage logs above, and verify NVIDIA_API_KEY/NGC_API_KEY or the "
            "configured local/remote endpoints."
        )
    if n_rows > 0 and (initial_n_rows is None or n_rows > initial_n_rows):
        return

    if initial_n_rows is not None:
        if n_rows < initial_n_rows:
            row_count_detail = f"row count decreased from {initial_n_rows} to {n_rows}"
        else:
            row_count_detail = f"row count stayed at {n_rows}"
        raise RuntimeError(
            f"retriever ingest did not add rows to {described}; {row_count_detail} "
            f"for {len(documents)} input file(s). This usually means extraction or embedding failed before "
            "any rows were written; check the captured stage logs above, and verify NVIDIA_API_KEY/NGC_API_KEY "
            "or the configured local/remote endpoints."
        )

    raise RuntimeError(
        f"retriever ingest produced 0 rows in {described} for {len(documents)} input file(s). "
        "This usually means extraction or embedding failed before any rows were written; check the captured "
        "stage logs above, and verify NVIDIA_API_KEY/NGC_API_KEY or the configured local/remote endpoints."
    )


def _count_result_rows(result: object) -> int | None:
    try:
        return len(result)  # type: ignore[arg-type]
    except TypeError:
        pass

    count = getattr(result, "count", None)
    if callable(count):
        try:
            return int(count())
        except Exception as exc:  # noqa: BLE001 — diagnostic only
            logger.debug("could not count ingest result rows: %s", exc)
    return None


def _count_index_rows(target: VdbTarget) -> int | None:
    """Return the row count of the target index, or ``None`` when it cannot be read."""
    if target.vdb_op == "lancedb":
        return _count_lancedb_rows(target.lancedb_uri, target.table_name)
    try:
        health = target.backend().health()
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        logger.debug("could not count rows in %s: %s", target.describe(), exc)
        return None
    return int(health["total_rows"]) if health.get("table_exists") else None


def _count_lancedb_rows(lancedb_uri: str, table_name: str) -> int | None:
    """Return the actual row count in ``<lancedb_uri>/<table_name>`` or ``None``.

    The low-level reader is best-effort so callers can decide whether an
    unknown count is acceptable. Root ingest treats an unknown final count as a
    failure because agents need proof that rows landed.
    """
    try:
        import lancedb  # local import — keeps the CLI startup snappy

        return int(lancedb.connect(lancedb_uri).open_table(table_name).count_rows())
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        logger.debug("could not count rows in %s/%s: %s", lancedb_uri, table_name, exc)
        return None
