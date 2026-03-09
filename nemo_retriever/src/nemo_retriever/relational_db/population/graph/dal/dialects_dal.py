from shared.graph.parsers.connection_type_to_dialect_mapper import conn_to_dialect_map
from shared.graph.model.reserved_words import BiConnectors
from infra.Neo4jConnection import get_neo4j_conn
import logging

logger = logging.getLogger("dialects_dal.py")
conn = get_neo4j_conn()


def get_dialects_by_account_id(account_id: str, is_omni: bool = False):
    query = """
        MATCH (n:connection {account_id: $account_id})
        WHERE not n.type in $bi_connectors
        RETURN collect(distinct n.type) AS connection_types
    """
    res = conn.query_read_only(
        query, {"account_id": account_id, "bi_connectors": BiConnectors.ALL}
    )
    if len(res) == 0:
        dialects = ["generic", "ansi"]
    else:
        dialects = set([conn_to_dialect_map[x] for x in res[0]["connection_types"]])
        dialects = list(dialects)
        if not is_omni:
            dialects.extend(["generic", "ansi"])
        # second_choice_dialects = set(ALL_DIALECTS) - dialects
        # dialects = list(dialects)
        # dialects.extend(list(second_choice_dialects))
    return dialects
