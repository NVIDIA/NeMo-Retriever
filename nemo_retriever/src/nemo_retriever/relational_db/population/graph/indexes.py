from nemo_retriever.relational_db.neo4j_connection import get_neo4j_conn
from nemo_retriever.relational_db.population.graph.model.reserved_words import Labels

conn = get_neo4j_conn()


def add_indices():
    parameters = {}

    # if len(index_names) == 0:
    query_create = """CREATE CONSTRAINT constraint_on_connection_id_and_account IF NOT EXISTS FOR (n: Connection)
                    REQUIRE (n.id, n.account_id) IS UNIQUE """
    conn.query_write(query_create, parameters)

    for c in Labels.LIST_OF_ALL:
        query_create = f"""CREATE CONSTRAINT constraint_on_{c.lower()}_id_and_account IF NOT EXISTS FOR (n: {c})
                        REQUIRE (n.id, n.account_id) IS UNIQUE """
        conn.query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_name IF NOT EXISTS FOR (n: {c}) ON(n.name)
                        """
        conn.query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_account IF NOT EXISTS FOR (n: {c}) ON(n.account_id)
                                    """
        conn.query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_id IF NOT EXISTS FOR (n: {c}) ON(n.id)
                                            """
        conn.query_write(query_create, parameters)

        labels_str = "|".join([n for n in Labels.LIST_OF_ALL if n != Labels.SQL])
        query_fulltext_name_create = f"""CREATE FULLTEXT INDEX name_index IF NOT EXISTS FOR
                                    (n:{labels_str}) ON EACH [n.name]
                                    OPTIONS {{indexConfig: {{`fulltext.eventually_consistent`: true,
                                    `fulltext.analyzer`: 'standard-no-stop-words'}}}}"""

        conn.query_write(query_fulltext_name_create, parameters)
