from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels


class Query:
    def __init__(
        self,
        schemas,
        id,
        q,
        utterance,
        ltimestamp,
        count,
        sql_node=None,
        default_schema=None,
        dialect=None,
        query_tag=None,
    ):
        self.id = id
        self.tables: list = []
        self.tables_ids: list[str] = []
        self.reached_columns_ids: list[str] = []
        self.edges: list = []
        self.nodes_counter: int = 0

        if sql_node is None:
            month = ltimestamp.month
            year = ltimestamp.year
            string_query = q.replace('"', '\\"')
            props = {
                "name": f"query_{str(id)}",
                f"cnt_{month}_{year}": count,
                "total_counter": count,
                "sql_full_query": string_query,
                "last_query_timestamp": ltimestamp,
                "query_tag": query_tag,
            }
            if utterance is not None:
                props["description"] = utterance.replace('"', '\\"')
            sql_node = Neo4jNode(
                name="query_" + str(id), label=Labels.SQL, props=props, existing_id=id
            )
        self.sql_node = sql_node

    def add_table_to_query(self, table_node, table_name: str):
        if not isinstance(table_node, Query):
            self.tables_ids.append(table_node.id)
        if (table_name, table_node) not in self.tables:
            self.tables.append((table_name, table_node))
        self.nodes_counter += 1

    def get_tables_ids(self) -> list[str]:
        return list(set(self.tables_ids))

    def get_reached_columns_ids(self) -> list[str]:
        return list(set(self.reached_columns_ids))

    def set_reached_columns_ids(self, columns_ids: list[str]):
        self.reached_columns_ids = columns_ids

    def get_nodes_counter(self) -> int:
        return self.nodes_counter

    def get_edges(self) -> list:
        return self.edges
