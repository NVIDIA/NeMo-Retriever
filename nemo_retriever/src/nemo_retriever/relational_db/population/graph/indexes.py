from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.model.reserved_words import Labels

conn = get_neo4j_conn()

description_index = [
    *Labels.LIST_OF_SEMANTIC,
    *Labels.LIST_OF_BI,
    Labels.DB,
    Labels.SCHEMA,
    Labels.TABLE,
    Labels.COLUMN,
]


# account_id is not really used, needed to get neo4j connection
def add_indices(account_id):
    parameters = {"account_id": account_id}
    # index_names = conn.query_read_only(
    #     query="SHOW INDEXES where type<>'VECTOR'", parameters=parameters
    # )
    # index_names = [x["name"] for x in index_names]
    # constraint_names = conn.query_read_only(
    #     query="SHOW CONSTRAINTS", parameters=parameters
    # )
    # constraint_names = [x["name"] for x in constraint_names]
    # for i in constraint_names:
    #     query = f"DROP CONSTRAINT {i} IF EXISTS"
    #     conn.query_write(query, parameters=parameters)
    # for i in index_names:
    #     query = f"DROP INDEX {i} IF EXISTS"
    #     conn.query_write(query, parameters=parameters)

    # if len(index_names) == 0:
    query_create = """CREATE CONSTRAINT constraint_on_connection_id_and_account IF NOT EXISTS FOR (n: connection)
                    REQUIRE (n.id, n.account_id) IS UNIQUE """
    conn.query_write(query_create, parameters)

    for c in Labels.LIST_OF_ALL:
        query_create = f"""CREATE CONSTRAINT constraint_on_{c.lower()}_id_and_account IF NOT EXISTS FOR (n: {c})
                        REQUIRE (n.id, n.account_id) IS UNIQUE """
        conn.query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_name IF NOT EXISTS FOR (n: {c}) ON(n.name) 
                        """
        conn.query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_clean_name IF NOT EXISTS FOR (n: {c}) ON(n.clean_name) 
                                    """
        conn.query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_account IF NOT EXISTS FOR (n: {c}) ON(n.account_id) 
                                    """
        conn.query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_id IF NOT EXISTS FOR (n: {c}) ON(n.id) 
                                            """
        conn.query_write(query_create, parameters)

        for c in [
            Labels.FUNCTION,
            Labels.ALIAS,
            Labels.CONSTANT,
            Labels.COMMAND,
            Labels.OPERATOR,
        ]:
            query_create = f"""CREATE INDEX index_on_{c.lower()}_sql_id IF NOT EXISTS FOR (n: {c}) ON(n.sql_id) 
                            """
            conn.query_write(query_create, parameters)

        labels_str = "|".join([n for n in Labels.LIST_OF_ALL if n != Labels.SQL])
        query_fulltext_name_create = f"""CREATE FULLTEXT INDEX name_index IF NOT EXISTS FOR 
                                    (n:{labels_str}) ON EACH [n.name] 
                                    OPTIONS {{indexConfig: {{`fulltext.eventually_consistent`: true, 
                                    `fulltext.analyzer`: 'standard-no-stop-words'}}}}"""
        query_fulltext_description_create = f"""CREATE FULLTEXT INDEX description_index IF NOT EXISTS FOR 
                                        (n:{"|".join(description_index)}) ON EACH [n.description]
                                        OPTIONS {{indexConfig: {{`fulltext.eventually_consistent`: true, 
                                        `fulltext.analyzer`: 'standard-no-stop-words'}}}}"""
        query_fulltext_terms_create = f"""CREATE FULLTEXT INDEX name_terms_index IF NOT EXISTS FOR 
                                        (n:{Labels.BT}|{Labels.ATTR}) ON EACH [n.name] 
                                        OPTIONS {{indexConfig: {{`fulltext.eventually_consistent`: true, 
                                        `fulltext.analyzer`: 'standard-no-stop-words'}}}}"""

        conn.query_write(query_fulltext_name_create, parameters)
        conn.query_write(query_fulltext_description_create, parameters)
        conn.query_write(query_fulltext_terms_create, parameters)
