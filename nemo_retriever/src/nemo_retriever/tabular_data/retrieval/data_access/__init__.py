# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-access helpers used by the text-to-SQL and deep_agent pipelines.

Each submodule is focused on one source / shape of data:

* :mod:`semantic_search` — vector search primitives.
* :mod:`candidates` — vector hits + graph enrichment for the SQL agent.
* :mod:`relevant_tables` — table-dict shaping (used by the agent prompts).
* :mod:`foreign_keys` — FK / join relationships fetched from Neo4j.
* :mod:`graph_schemas` — schema / table / node lookups in Neo4j.
* :mod:`custom_analyses` — CustomAnalysis fetch + selection helpers.
* :mod:`response_formatting` — final-response entity highlighting and links.

Call sites import from the specific submodule (no re-exports here) so the
dependencies of every consumer remain explicit.
"""
