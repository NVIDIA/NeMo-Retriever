# SPDX-License-Identifier: Apache-2.0

DEFAULT_LANCEDB_TABLE_NAME = "nemo-retriever"
"""Canonical default LanceDB table name.

Single source of truth shared by the ingest, query, service, harness, and
tooling layers so that a default ingest and a default query always target the
same table. Historically these layers disagreed (``nv-ingest`` vs
``nemo-retriever`` vs ``nemo_retriever``); route new defaults through this
constant instead of hard-coding a string.
"""
