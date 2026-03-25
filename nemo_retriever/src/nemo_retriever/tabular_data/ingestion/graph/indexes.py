from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.tabular_data.ingestion.graph.model.reserved_words import Labels

conn = get_neo4j_conn()


def add_indices():
    parameters = {}

    for c in Labels.LIST_OF_ALL:
        query_create = f"""CREATE CONSTRAINT constraint_on_{c.lower()}_id IF NOT EXISTS FOR (n: {c})
                        REQUIRE (n.id) IS UNIQUE """
        conn.query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_name IF NOT EXISTS FOR (n: {c}) ON(n.name)
                        """
        conn.query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_id IF NOT EXISTS FOR (n: {c}) ON(n.id)
                                            """
        conn.query_write(query_create, parameters)
