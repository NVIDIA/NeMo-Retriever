import os
from neo4j import GraphDatabase, exceptions, WRITE_ACCESS, READ_ACCESS

from .SecretsManager import SecretsManager
import logging


logger = logging.getLogger("Neo4jConnection")


class Neo4jConnection:
    def __init__(self, uri, username, password):
        self.__uri = uri
        self.__username = username
        self.__password = password
        self.__driver = None

        try:
            self.__driver = GraphDatabase.driver(
                self.__uri,
                auth=(self.__username, self.__password),
                max_connection_lifetime=290,
                liveness_check_timeout=4,
                notifications_min_severity="OFF",  # or 'OFF' to disable entirely
            )
        except Exception as e:
            print("Failed to create the driver: ", e)

    def verify_connectivity(self):
        return self.__driver.verify_connectivity()

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(
        self,
        query,
        parameters=None,
        default_access_mode=WRITE_ACCESS,
        ret_type="data",
    ):
        assert self.__driver is not None, "Driver not initialized!"
        db = "neo4j"
        session = None
        response = None
        tries = 3
        for i in range(tries):
            try:
                session = self.__driver.session(
                    database=db,
                    default_access_mode=default_access_mode,
                )
                result = session.run(query, parameters)
                response = result.data() if ret_type == "data" else result.graph()
            except exceptions.TransientError as te:
                if i < tries - 1:  # i is zero indexed
                    # logger.info(f"Retrying for ({i + 1} out of {tries}): {query} ")
                    continue
                else:
                    # logger.error(
                    #     f"CYPHER QUERY FAILED {i} times with TransientError: {query}, parameters: {parameters}"
                    # )
                    logger.error(f"CYPHER QUERY FAILED {i} times with TransientError")
                    if session is not None:
                        session.close()
                    raise te
            except Exception as e:
                logger.error(f"CYPHER QUERY FAILED: {query}, parameters: {parameters}")
                if session is not None:
                    session.close()
                raise e
            break
        if session is not None:
            session.close()
        return response


class Neo4jConnectionManager:
    def __init__(self):
        if os.environ["LMX_ENV"] != "development":
            creds = SecretsManager().get_secret_dict(
                os.environ["LMX_NEO4J_SECRET_NAME"]
            )
            self.conn = Neo4jConnection(
                creds["uri"], creds["username"], creds["password"]
            )

        else:
            self.conn = Neo4jConnection(
                os.environ["NEO4J_URI"],
                os.environ["NEO4J_USERNAME"],
                os.environ["NEO4J_PASSWORD"],
            )

        logger.info("Verify Connectivity for default Neo4j")
        self.conn.verify_connectivity()

    def _get_connection_for_account(self, parameters):
        if parameters is None:
            raise ValueError("Paremeters (with account_id) must be sent to Neo4j")
        if "account_id" not in parameters:
            raise ValueError("account_id is missing")

        return self.conn

    def verify_connectivity(self):
        # Checking for all customers only (at the moment)
        return self.conn.verify_connectivity()

    def query_write(self, query, parameters):
        conn = self._get_connection_for_account(parameters)
        return conn.query(query, parameters)

    def query_read_only(self, query, parameters):
        conn = self._get_connection_for_account(parameters)
        return conn.query(query, parameters, READ_ACCESS)

    def query_graph(self, query, parameters):
        conn = self._get_connection_for_account(parameters)
        return conn.query(query, parameters, ret_type="graph")


manager = Neo4jConnectionManager()


def get_neo4j_conn():
    return manager
