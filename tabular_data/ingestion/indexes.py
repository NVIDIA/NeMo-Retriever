# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels


def add_indices():
    parameters = {}

    for c in Labels.LIST_OF_ALL:
        query_create = f"""CREATE CONSTRAINT constraint_on_{c.lower()}_id IF NOT EXISTS FOR (n: {c})
                        REQUIRE (n.id) IS UNIQUE """
        get_neo4j_conn().query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_name IF NOT EXISTS FOR (n: {c}) ON(n.name)
                        """
        get_neo4j_conn().query_write(query_create, parameters)
        query_create = f"""CREATE INDEX index_on_{c.lower()}_id IF NOT EXISTS FOR (n: {c}) ON(n.id)
                                            """
        get_neo4j_conn().query_write(query_create, parameters)
