import pandas as pd
from nemo_retriever.relational_db.connectors.duckdb import DuckDB
from nemo_retriever.relational_db.population.graph.utils import (
    load_fks,
    load_pks,
    load_tables,
    load_columns,
)


def create_dataframe(settings):
    duckdb_connector = DuckDB(settings.get("connection_properties", {"database": "./spider2.duckdb"}))

    queries = duckdb_connector.get_queries()

    schema = duckdb_connector.get_schemas()
    tables = schema[0]
    columns = schema[1]

    views = duckdb_connector.get_views()

    pull_df = pd.DataFrame(duckdb_connector.db_schemas).explode("schemas")
    pull_df = pull_df.rename({"db_name": "database", "schemas": "schema"}, axis=1)

    tables = tables.merge(pull_df)
    columns = columns.merge(pull_df)
    views = views.merge(pull_df)

    pks = duckdb_connector.get_pks()
    fks = duckdb_connector.get_fks()

    return tables, columns, views, queries, pks, fks


def data_for_populate_tabular(settings):
    """Build the `data` dict expected by populate_tabular_data from create_dataframe output."""
    tables, columns, views, queries, pks, fks = create_dataframe(settings)
    tables = load_tables(tables)
    columns = load_columns(columns)
    pks = load_pks(pks)
    fks = load_fks(fks)
    data = {
        "tables": tables,
        "columns": columns,
        "views": views,
        "pks": pks,
        "fks": fks,
    }
    # queries is not used by populate_tabular_data; include if needed elsewhere
    return data


def extract_relational_db_data(params=None):
    """Step 1 — Pull schema entities from the relational DB into a data dict.

    Args:
        params: TabularExtractParams instance. When provided,
            ``params.db_connection_string`` overrides the default database path.

    Returns:
        data dict with keys: tables, columns, views, pks, fks.
    """
    db_path = (
        params.db_connection_string
        if params is not None and params.db_connection_string is not None
        else "./spider2.duckdb"
    )
    settings = {"connection_properties": {"database": db_path}}
    return data_for_populate_tabular(settings)


def store_relational_db_in_neo4j(data, neo4j_conn=None):
    """Step 2 — Write the extracted data dict as graph nodes into Neo4j.

    Args:
        data:       Data dict returned by extract_relational_db_data().
        neo4j_conn: Active Neo4jConnectionManager instance (unused directly here;
                    populate_tabular_data uses its own DAL connection, but
                    accepted for API consistency with the other ingest steps).
    """
    from nemo_retriever.relational_db.population.populate_data import (
        populate_structured_data,
    )

    populate_structured_data(
        data,
        num_workers=4,
        dialect="duckdb",
    )
