import sys
import pendulum
import pandas as pd
from structured_data import DuckDBEngine


def create_dataframe(settings):
    duckdb_connector = DuckDBEngine(settings["connection_properties"])
    queries = duckdb_connector.get_queries(
        settings["data_interval_start"], settings["data_interval_end"]
    )

    schema = duckdb_connector.get_schemas()
    tables = schema[0]
    columns = schema[1]

    views = duckdb_connector.get_views()

    pull_df = pd.DataFrame(duckdb_connector.pull_info).explode("schemas")
    pull_df = pull_df.rename(
        {"db_name": "database", "schemas": "schema"}, axis=1
    )

    tables = tables.merge(pull_df)
    columns = columns.merge(pull_df)
    views = views.merge(pull_df)

    pks = duckdb_connector.get_pks()
    fks = duckdb_connector.get_fks()

    return tables, columns, views, queries, pks, fks


def parse_param(argv):
    ret = {}
    ret["data_interval_start"] = pendulum.parse(argv[1])
    ret["data_interval_end"] = pendulum.parse(argv[2])
    ret["connection_properties"] = argv[3]
    return ret


settings = parse_param(sys.argv)
create_dataframe(settings)
