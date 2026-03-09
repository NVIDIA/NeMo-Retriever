"""Database client."""

import os
import psycopg2
import logging
from psycopg2.extras import Json
from psycopg2.extensions import register_adapter
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager

from .SecretsManager import SecretsManager

register_adapter(dict, Json)


class PostgresConnection:
    """PostgreSQL Database class."""

    def __init__(self, creds):
        self.__logger = logging.getLogger("postgres-conn")

        # pool define with 10 live connections
        self.connectionpool = SimpleConnectionPool(
            1,
            2,
            host=creds["host"],
            user=creds["username"],
            password=creds["password"],
            port=creds["port"],
            dbname=creds["database"],
        )
        self.connection_string = self.build_pgvector_connection_string(creds)

    def build_pgvector_connection_string(self, creds: dict) -> str:
        return (
            f"postgresql+psycopg2://{creds['username']}:{creds['password']}"
            f"@{creds['host']}:{creds['port']}/{creds['database']}"
        )

    @contextmanager
    def getcursor(self):
        con = self.connectionpool.getconn()
        con.autocommit = True
        try:
            yield con.cursor()
        finally:
            # Closing the connection Everytime, because we had connectivity erroes when pg is hosted in the demo cluster.
            # psycopg2.OperationalError: server closed the connection unexpectedly
            self.connectionpool.putconn(con, None, close=True)

    def execute_query(self, query, params=None):
        """Run a SQL query and return the results."""
        with self.getcursor() as cur:
            cur.execute(query, params or ())
            try:
                return cur.fetchall()  # Returns list of rows
            except psycopg2.ProgrammingError:
                # No results to fetch (e.g., for INSERT/UPDATE)
                return None

    def delete_rows(self, query, values):
        with self.getcursor() as cur:
            cur.execute(query, values)
            return f"{cur.rowcount} rows affected."

    def execute_values(self, query, values, template):
        """Run a SQL query to insert rows in table."""
        with self.getcursor() as cur:
            psycopg2.extras.execute_values(cur, query, values, template)
            return f"{cur.rowcount} inserted."

    def execute_query_with_returning(self, query, values, template):
        ## only use this if you have RETURNING in your query, otherwise use execute_values
        """Run a SQL query to insert rows in table."""
        returned_values = []
        with self.getcursor() as cur:
            res = psycopg2.extras.execute_values(
                cur, query, values, template, 100, True
            )
            for returned_value in res:
                returned_values.append(returned_value[0])
        return returned_values

    def install_vector_extension(self):
        """Install the vector extension if it doesn't exist."""
        try:
            with self.getcursor() as cur:
                # Check if vector extension exists
                cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                if not cur.fetchone():
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except psycopg2.errors.UndefinedFile as e:
            self.__logger.error(f"pg vector extension not available: {e}")
            raise e
        except Exception as e:
            self.__logger.error(f"pg vector extension error: {e}")
            raise e


if os.environ["LMX_ENV"] != "development":
    creds = SecretsManager().get_secret_dict(os.environ["LMX_POSTGRES_SECRET_NAME"])
else:
    creds = {
        "host": os.environ["POSTGRES_HOST"],
        "port": os.environ["POSTGRES_PORT"],
        "username": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "database": os.environ["POSTGRES_DATABASE"],
    }
logger = logging.getLogger("postgres-conn")
conn = PostgresConnection(creds)

# Install vector extension on initialization
try:
    conn.install_vector_extension()
except Exception as e:
    logger.warning(f"Could not install vector extension: {e}")


# One day, we will have database per account, and here there will be some logic
def get_postgres_conn():
    return conn
