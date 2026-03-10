from enum import Enum

# TODO how to do the connection
from nemo_retriever.relational_db.infra.Neo4jConnection import get_neo4j_conn
from nemo_retriever.relational_db.population.graph.model.reserved_words import Labels

conn = get_neo4j_conn()


class LabelsWithDescriptionSuggestion(str, Enum):
    TERM = Labels.BT
    ANALYSIS = Labels.ANALYSIS
    ATTRIBUTE = Labels.ATTR
    SNIPPET = "snippet"
    QUERY = Labels.SQL
    SCHEMA = Labels.SCHEMA
    TABLE = Labels.TABLE
    COLUMN = Labels.COLUMN

   


def insert_description_suggestion(
    account_id: str,
    label: LabelsWithDescriptionSuggestion,
    node_id: str,
    description_suggestion: str,
):
    if label == LabelsWithDescriptionSuggestion.SNIPPET:
        return insert_snippet_description_suggestion(
            account_id, node_id, description_suggestion
        )

    suggestion = description_suggestion.strip().capitalize()
    label_str = str(label.value)
    query = f"""match(n:{label_str} {{account_id: $account_id, id: $node_id}})
                set n.description_suggestion = $description_suggestion"""
    result = conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "node_id": node_id,
            "description_suggestion": suggestion,
        },
    )
    return result


def insert_snippet_description_suggestion(
    account_id: str, snippet_id: str, description_suggestion: str
):
    suggestion = description_suggestion.strip().capitalize()
    query = """
        MATCH (:attribute|metric{account_id:$account_id})-[n:attr_of|metric_sql{sql_snippet_id: $snippet_id}]-() 
        SET n.description_suggestion = $description_suggestion
    """
    return conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "snippet_id": snippet_id,
            "description_suggestion": suggestion,
        },
    )
