from nemo_retriever.tabular_data.connectors.duckdb import DuckDB
from nemo_retriever.tabular_data.ingestion.graph.utils import (
    normalize_fks,
    normalize_pks,
    normalize_tables,
    normalize_columns,
)


def create_dataframe(settings):
    duck = DuckDB(settings.get("connection_string", "./spider2.duckdb"))

    queries = duck.get_queries()
    tables = duck.get_tables()
    columns = duck.get_columns()
    views = duck.get_views()
    pks = duck.get_pks()
    fks = duck.get_fks()

    return tables, columns, views, queries, pks, fks


def data_for_populate_tabular(settings):
    """Build the `data` dict expected by populate_tabular_data() from create_dataframe output."""
    tables, columns, views, queries, pks, fks = create_dataframe(settings)
    tables = normalize_tables(tables)
    columns = normalize_columns(columns)
    pks = normalize_pks(pks)
    fks = normalize_fks(fks)
    data = {
        "tables": tables,
        "columns": columns,
        "views": views,
        "pks": pks,
        "fks": fks,
    }
    # queries is not used by populate_tabular_data(); include if needed elsewhere
    return data


def extract_tabular_db_data(params=None):
    """Step 1 — Pull schema entities from the relational DB into a data dict.

    Args:
        params: TabularExtractParams instance. When provided,
            ``params.connection_string`` overrides the default database path.

    Returns:
        data dict with keys: tables, columns, views, pks, fks.
    """
    db_path = (
        params.connection_string if params is not None and params.connection_string is not None else "./spider2.duckdb"
    )
    settings = {"connection_string": db_path}
    return data_for_populate_tabular(settings)


def store_relational_db_in_neo4j(data, neo4j_conn=None):
    """Step 2 — Write the extracted data dict as graph nodes into Neo4j.

    Args:
        data:       Data dict returned by extract_tabular_db_data().
        neo4j_conn: Active Neo4jConnectionManager instance (unused directly here;
                    populate_tabular_data uses its own DAL connection, but
                    accepted for API consistency with the other ingest steps).
    """
    from nemo_retriever.tabular_data.ingestion.write_to_graph import (
        populate_tabular_data,
    )

    populate_tabular_data(
        data,
        num_workers=4,
        dialect="duckdb",
    )
