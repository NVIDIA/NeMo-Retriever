# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels


class Query:
    def __init__(
        self,
        schemas,
        id,
        sql_text,
        ltimestamp,
        count,
        dialect=None,
    ):
        self.id = id
        self.tables: list = []
        self.tables_ids: list[str] = []
        self.edges: list = []
        self.ast_node_count: int = 0

        month = ltimestamp.month
        year = ltimestamp.year
        props = {
            "name": f"query_{str(id)}",
            f"cnt_{month}_{year}": count,
            "total_counter": count,
            "sql_full_query": sql_text,
            "last_query_timestamp": ltimestamp,
        }
        self.sql_node = Neo4jNode(name="query_" + str(id), label=Labels.SQL, props=props, existing_id=id)

    def add_table_to_query(self, table_node, table_name: str):
        if not isinstance(table_node, Query):
            self.tables_ids.append(table_node.id)
        if (table_name, table_node) not in self.tables:
            self.tables.append((table_name, table_node))

    def get_tables_ids(self) -> list[str]:
        return list(set(self.tables_ids))

    def get_column_ids(self) -> list[str]:
        """Return deduplicated IDs of Column nodes linked by this query's edges.

        In the flat edge structure produced by ``parse_query_slim``, Column
        nodes are the leaf nodes — they have no further outgoing edges.
        This mirrors the old ``get_leafs_from_graph`` heuristic used for
        pre-filtering duplicate candidates.
        """
        return list({
            edge[1].id
            for edge in self.edges
            if edge[1].label == Labels.COLUMN
        })

    def get_nodes_counter(self) -> int:
        """Total number of nodes in the sqlglot AST.

        Set by ``parse_query_slim`` after ``extract_tables_and_columns``
        parses the SQL.  Acts as a cheap structural fingerprint for
        pre-filtering duplicate candidates — two queries with different
        AST sizes cannot be structurally equivalent.
        """
        return self.ast_node_count

    def get_edges(self) -> list:
        return self.edges
