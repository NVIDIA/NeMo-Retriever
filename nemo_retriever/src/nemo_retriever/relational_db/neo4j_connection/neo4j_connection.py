# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Neo4j connection management for the relational_db stack.
"""

import os
import logging

from neo4j import GraphDatabase, Result, RoutingControl

logger = logging.getLogger(__name__)


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
                notifications_min_severity="OFF",
            )
        except Exception as e:
            print("Failed to create the driver: ", e)

    def verify_connectivity(self):
        return self.__driver.verify_connectivity()

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def query(
        self,
        query,
        parameters=None,
        routing=RoutingControl.WRITE,
        ret_type="data",
    ):
        assert self.__driver is not None, "Driver not initialized!"
        try:
            if ret_type == "data":
                records, _, _ = self.__driver.execute_query(
                    query,
                    parameters_=parameters,
                    routing_=routing,
                    database_="neo4j",
                )
                return [dict(record) for record in records]
            else:
                return self.__driver.execute_query(
                    query,
                    parameters_=parameters,
                    routing_=routing,
                    database_="neo4j",
                    result_transformer_=Result.graph,
                )
        except Exception as e:
            logger.error(f"CYPHER QUERY FAILED: {query}, parameters: {parameters}")
            raise e

    def query_write(self, query, parameters=None):
        """Run a write query."""
        return self.query(query, parameters)

    def query_read(self, query, parameters=None):
        """Run a read-only query."""
        return self.query(query, parameters, routing=RoutingControl.READ)

    def query_graph(self, query, parameters=None):
        """Run a query and return the graph."""
        return self.query(query, parameters, ret_type="graph")


_conn = None


def get_neo4j_conn() -> Neo4jConnection:
    """Return the shared Neo4j connection (singleton)."""
    global _conn
    if _conn is None:
        _conn = Neo4jConnection(
            os.environ["NEO4J_URI"],
            os.environ["NEO4J_USERNAME"],
            os.environ["NEO4J_PASSWORD"],
        )
        logger.info("Verify connectivity for Neo4j")
        _conn.verify_connectivity()
    return _conn
