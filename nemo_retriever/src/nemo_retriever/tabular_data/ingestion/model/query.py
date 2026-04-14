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
        self.nodes_counter: int = 0

        month = ltimestamp.month
        year = ltimestamp.year
        props = {
            "name": f"query_{str(id)}",
            f"cnt_{month}_{year}": count,
            "total_counter": count,
            "sql_full_query": sql_text,
            "last_query_timestamp": ltimestamp,
        }
        self.sql_node = Neo4jNode(
            name="query_" + str(id), label=Labels.SQL, props=props, existing_id=id
        )

    def add_table_to_query(self, table_node, table_name: str):
        if not isinstance(table_node, Query):
            self.tables_ids.append(table_node.id)
        if (table_name, table_node) not in self.tables:
            self.tables.append((table_name, table_node))
        self.nodes_counter += 1

    def get_tables_ids(self) -> list[str]:
        return list(set(self.tables_ids))

    def get_nodes_counter(self) -> int:
        return self.nodes_counter

    def get_edges(self) -> list:
        return self.edges
