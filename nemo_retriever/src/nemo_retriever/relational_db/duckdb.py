import pandas as pd
from connectors.connector import DBConnector


class DuckDB(DBConnector):
    def __init__(self, connection):
        super().__init__(connection)
        database = self.connection_properties.get("database", ":memory:")
        read_only = self.connection_properties.get("read_only", False)
        self.conn = duckdb.connect(database=database, read_only=read_only)

    def _query(self, query):
        return self.conn.execute(query).df()

    def get_databases(self):
        result = self._query("SELECT DISTINCT catalog_name FROM information_schema.schemata")
        return result["catalog_name"].tolist()

    def get_schemas_of_database(self, db):
        result = self._query(
            f"SELECT DISTINCT schema_name FROM information_schema.schemata "
            f"WHERE catalog_name = '{db}'"
        )
        return result["schema_name"].tolist()

    def test(self):
        to_return = []
        databases = self.get_databases()
        for db in databases:
            schemas = self.get_schemas_of_database(db)
            to_return.append({"db_name": db, "schemas": schemas})
        return to_return

    def get_schemas(self):
        def get_columns():
            q = """
            SELECT
                table_catalog  AS `database`,
                table_schema   AS `schema`,
                table_name     AS `table_name`,
                column_name    AS `column_name`,
                ordinal_position AS `ordinal_position`,
                data_type      AS `data_type`,
                is_nullable    AS `is_nullable`
            FROM information_schema.columns
            ORDER BY table_catalog, table_schema, table_name, ordinal_position
            """
            return self._query(q)

        def get_tables():
            q = """
            SELECT
                table_catalog AS `database`,
                table_schema  AS `schema`,
                table_name    AS `table_name`,
                table_type    AS `table_type`,
                NULL          AS `created`
            FROM information_schema.tables
            ORDER BY table_catalog, table_schema, table_name
            """
            return self._query(q)

        return get_tables(), get_columns()

    def get_queries(self, data_interval_start, data_interval_end):
        # DuckDB does not expose a built-in query history via information_schema.
        # Return an empty DataFrame matching the expected schema.
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self):
        q = """
        SELECT
            table_catalog AS `database`,
            table_schema  AS `schema`,
            table_name    AS `table_name`,
            view_definition AS `view_definition`
        FROM information_schema.views
        ORDER BY table_catalog, table_schema, table_name
        """
        return self._query(q)
