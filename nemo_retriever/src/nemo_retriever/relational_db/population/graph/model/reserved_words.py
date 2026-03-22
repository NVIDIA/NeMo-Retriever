class Labels:
    COMMAND = "Command"
    OPERATOR = "Operator"
    FUNCTION = "Function"
    CONSTANT = "Constant"
    ALIAS = "Alias"
    SQL = "Sql"
    COLUMN = "Column"
    TEMP_COLUMN = "TempColumn"
    TABLE = "Table"
    TEMP_TABLE = "TempTable"
    SCHEMA = "Schema"
    TEMP_SCHEMA = "TempSchema"
    DB = "Db"
    CONNECTION = "Connection"

    LIST_OF_ALL = [
        DB,
        SCHEMA,
        TABLE,
        COLUMN,
        TEMP_SCHEMA,
        TEMP_TABLE,
        TEMP_COLUMN,
        SQL,
        COMMAND,
        OPERATOR,
        FUNCTION,
        CONSTANT,
        CONNECTION,
    ]


labels_to_types = {
    Labels.TABLE: "base table",
}


class Props:
    """Edge/node property keys (used by utils_dal, node)."""

    JOIN = "join"
    SOURCE_SQL_ID = "source_sql_id"
    UNION = "union"
    SQL_ID = "sql_id"


# Labels that have no parent owner in the graph (used by get_entity_before_update).
entities_without_owners = []


class RelTypes:
    """Neo4j relationship type names (used in Cypher)."""

    CONTAINS = "CONTAINS"
    CONNECTING = "CONNECTING"
    FOREIGN_KEY = "FOREIGN_KEY"


# Relationship types for owner traversal (used by get_node_parent_owner_by_id).
data_relationships = [RelTypes.CONTAINS]


def label_to_type(label: str) -> str:
    if label in labels_to_types:
        return labels_to_types[label]
    return label
