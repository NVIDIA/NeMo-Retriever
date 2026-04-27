# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
import pandas as pd
import numpy as np
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels

pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


class Schema:
    def __init__(
        self,
        db_node: Neo4jNode = None,
        schema_tables_df: pd.DataFrame = None,
        schema_columns_df: pd.DataFrame = None,
        schema_name: str = None,
        is_creation_mode: bool = True,
    ):
        self.db_node = db_node
        self.schema_node = None
        self.schema_name = ""

        self.tables_to_columns = {}  # key - table node object, value - list of column names (strings) in lower case
        self.tables_columns_pos = {}  # key - tuple of the form (table name, ordinal position), value - column name
        self.table_nodes = {}  # key - table name in lower case, value - table node object
        self.column_nodes = (
            {}
        )  # key - tuple of the form (column name in lower case, table name in lower case), value - column node object
        self.id_to_node = {}  # key - node id, value - column or table "full" names (schema_name.table_name.column_name

        self.tables_df = schema_tables_df
        self.columns_df = schema_columns_df
        if schema_tables_df is not None and schema_columns_df is not None:
            table_names = self.tables_df.table_name.unique()
            self.columns_df = self.columns_df.loc[self.columns_df["table_name"].isin(table_names)]

            self.tables_df["table_name_lower"] = self.tables_df["table_name"].apply(lambda x: x.lower())
            if "id" not in self.tables_df.columns:
                self.tables_df["id"] = None
            mask = self.tables_df["id"].isna()
            self.tables_df.loc[mask, "id"] = [str(uuid.uuid4()) for _ in range(mask.sum())]

            self.tables_df["full_name"] = self.tables_df.apply(
                lambda x: f"{x.table_schema}.{x.table_name}",
                axis=1,
            )

            self.id_to_node.update(self.tables_df.set_index("id")["full_name"].to_dict())

            self.columns_df["table_name_lower"] = self.columns_df["table_name"].apply(lambda x: x.lower())

            self.columns_df["column_name_lower"] = self.columns_df["column_name"].apply(lambda x: x.strip('"').lower())
            if "id" not in self.columns_df.columns:
                self.columns_df["id"] = None
            mask = self.columns_df["id"].isna()
            self.columns_df.loc[mask, "id"] = [str(uuid.uuid4()) for _ in range(mask.sum())]
            self.columns_df["data_type"] = self.columns_df["data_type"].str.strip('"')
            self.columns_df["full_name"] = self.columns_df.apply(
                lambda x: f"{x.table_schema}.{x.table_name}.{x.column_name}",
                axis=1,
            )
            self.id_to_node.update(self.columns_df.set_index("id")["full_name"].to_dict())

            # This is for improving is_column_in_table
            self.columns_map = {}
            for index, row in self.columns_df.iterrows():
                if row["table_name_lower"] not in self.columns_map:
                    self.columns_map[row["table_name_lower"]] = {}
                self.columns_map[row["table_name_lower"]][row["column_name_lower"]] = True

            if is_creation_mode:  # ingestion mode
                self.reset_tables_props()
                self.reset_columns_props()
            else:
                # retrieval mode
                self.schema_name = schema_name
                self.reset_slim_columns_props()
                self.reset_slim_tables_props()

    def reset_tables_props(self):
        self.tables_df["props"] = self.tables_df.apply(
            lambda x: {
                "name": x["table_name"],
                "created": None if pd.isna(x["created"]) else x["created"],
                "description": None if pd.isna(x["description"]) else x["description"],
                "id": x.id,
            },
            axis=1,
        )
        self.tables_df["match_props"] = self.tables_df.apply(
            lambda x: {"id": x.id},
            axis=1,
        )

    def reset_columns_props(self):
        self.columns_df["props"] = self.columns_df.apply(
            lambda x: {
                "name": x["column_name"].strip('"'),
                "data_type": (None if pd.isna(x["data_type"]) else x["data_type"].strip('"')),
                "is_nullable": x["is_nullable"],
                "ordinal_position": x["ordinal_position"],
                "description": None if pd.isna(x["description"]) else x["description"],
                "id": x.id,
            },
            axis=1,
        )
        self.columns_df["match_props"] = self.columns_df.apply(
            lambda x: {"id": x.id},
            axis=1,
        )

    def reset_slim_tables_props(self):
        self.tables_df["props"] = self.tables_df.apply(
            lambda x: {
                "name": x["name"],
                "id": x["id"],
            },
            axis=1,
        )
        self.tables_df["match_props"] = self.tables_df.apply(
            lambda x: {"id": x["id"]},
            axis=1,
        )

    def reset_slim_columns_props(self):
        self.columns_df["props"] = self.columns_df.apply(
            lambda x: {
                "name": x["name"].strip('"'),
                "data_type": (x["data_type"] or "").strip('"'),
                "id": x["id"],
            },
            axis=1,
        )
        self.columns_df["match_props"] = self.columns_df.apply(
            lambda x: {"id": x["id"]},
            axis=1,
        )

    def get_schema_name(self):
        return self.schema_name

    def get_db_node(self):
        return self.db_node

    def get_db_name(self):
        return self.db_node.get_name()

    def get_table_nodes(self):
        if self.tables_df is not None:
            return self.tables_df.to_dict(orient="records")
        return []

    def get_column_nodes(self):
        if self.columns_df is not None:
            return self.columns_df.to_dict(orient="records")
        return []

    def get_column_nodes_by_table_name(self, table_name: str):
        if self.columns_df is not None:
            columns_names = list(self.get_table_columns_by_table_name(table_name))
            columns_nodes = [self.get_column_node(column_name, table_name) for column_name in columns_names]
            return columns_nodes
        return []

    def get_table_columns_by_table_name(self, table_name):
        table_name_lower = table_name.lower()
        columns_df = self.columns_df.loc[self.columns_df["table_name_lower"] == table_name_lower].replace(np.nan, None)
        return columns_df.column_name.unique()

    def get_table_columns(self, table_node):
        table_name_lower = table_node.name.lower()
        if self.columns_df is not None:
            columns_df = self.columns_df.loc[self.columns_df["table_name_lower"] == table_name_lower].replace(
                np.nan, None
            )
            if not columns_df.empty:
                return columns_df.column_name.unique()
        return self.tables_to_columns[table_node]

    def add_column_to_table(self, table_node, column_node, ordinal_position=None):
        column_name_lower = column_node.get_name().lower()
        if table_node not in self.tables_to_columns.keys():
            self.tables_to_columns[table_node] = []
        if column_name_lower not in self.tables_to_columns[table_node]:
            self.tables_to_columns[table_node].append(column_name_lower)
            self.tables_columns_pos.update({(table_node.name.lower(), ordinal_position): column_name_lower})

    def get_db_schema_edge(self):
        if self.db_node is None:
            raise Exception("No db node exists in the given schema object.")
        else:
            db_node = self.db_node
        if self.schema_node is None:
            raise Exception("No schema node exists in the given schema object.")
        else:
            schema_node = self.schema_node
        edge = (db_node, schema_node, {})
        return edge

    def get_schema_to_tables_edges(self):
        if self.tables_df is not None:
            edges_to_tables = [
                {
                    "vid": str(self.schema_node.get_id()),
                    "uid": table_id,
                    "props": {},
                }
                for table_id in [x["id"] for x in list(self.tables_df["props"])]
            ]
            return edges_to_tables
        return []

    def get_edges_per_table(self):
        edges_per_table = []

        if self.tables_df is not None:
            unique_table_names = self.tables_df.table_name.unique()
            for table_name in unique_table_names:
                table_name_lower = table_name.lower()
                tables_df = self.tables_df.loc[self.tables_df["table_name_lower"] == table_name_lower].replace(
                    np.nan, None
                )
                table_id = tables_df.iloc[0]["props"]["id"]
                column_df = self.columns_df.loc[self.columns_df["table_name_lower"] == table_name_lower].replace(
                    np.nan, None
                )
                table_edges = [
                    {
                        "vid": table_id,
                        "uid": column_id,
                        "props": {},
                    }
                    for column_id in [x["id"] for x in list(column_df["props"])]
                ]
                edges_per_table.append(table_edges)
        return edges_per_table

    def create_column_node(
        self,
        column_name,
        table_name="",
        data_type=None,
        id=None,
        description=None,
        is_nullable=None,
        ordinal_position=None,
    ):
        column_name_lower = column_name.lower()
        table_name_lower = table_name.lower()
        props = {"name": column_name}
        match_props = {"id": id}
        props = self.update_column_props_by_arguments(
            props,
            data_type,
            description,
            is_nullable,
            ordinal_position,
        )
        column_node = Neo4jNode(
            column_name,
            label=Labels.COLUMN,
            props=props,
            existing_id=id,
            match_props=match_props,
        )
        self.column_nodes.update({(column_name_lower, table_name_lower): column_node})
        if str(column_node.get_id()) not in self.id_to_node:
            self.id_to_node.update({str(column_node.get_id()): f"{self.schema_name}.{table_name}.{column_name}"})

    def get_column_node_props(self, column_name, table_name):
        column_name_lower = column_name.lower()
        table_name_lower = table_name.lower()
        column_df = self.columns_df.loc[
            (self.columns_df["table_name_lower"] == table_name_lower)
            & (self.columns_df["column_name_lower"] == column_name_lower)
        ].replace(np.nan, None)

        if column_df.empty:
            raise Exception(f"Column {column_name} is not in table {table_name}.")

        column = column_df.to_dict(orient="records")[0]
        return column["props"]

    def get_column_node_match_props(self, column_name, table_name):
        column_name_lower = column_name.lower()
        table_name_lower = table_name.lower()
        column_df = self.columns_df.loc[
            (self.columns_df["table_name_lower"] == table_name_lower)
            & (self.columns_df["column_name_lower"] == column_name_lower)
        ].replace(np.nan, None)

        if column_df.empty:
            raise Exception(f"Column {column_name} is not in table {table_name}.")

        column = column_df.to_dict(orient="records")[0]
        return column["match_props"]

    def get_column_node(self, column_name, table_name):
        column_name_lower = column_name.lower()
        table_name_lower = table_name.lower()
        if (column_name_lower, table_name_lower) not in self.column_nodes:
            column_df = self.columns_df.loc[
                (self.columns_df["table_name_lower"] == table_name_lower)
                & (self.columns_df["column_name_lower"] == column_name_lower)
            ].replace(np.nan, None)

            if column_df.empty:
                raise ValueError(f"Column {column_name} is not in table {table_name}.")

            id = column_df.iloc[0]["props"]["id"]
            data_type = None if pd.isna(column_df.iloc[0]["data_type"]) else column_df.iloc[0]["data_type"].strip('"')
            column_name = column_df.iloc[0]["column_name"].strip('"')
            description = (
                None
                if "description" not in column_df.iloc[0] or pd.isna(column_df.iloc[0]["description"])
                else column_df.iloc[0]["description"]
            )
            is_nullable = (
                None
                if "is_nullable" not in column_df.iloc[0] or pd.isna(column_df.iloc[0]["is_nullable"])
                else column_df.iloc[0]["is_nullable"]
            )
            ordinal_position = (
                None if ("ordinal_position" not in column_df.iloc[0]) else column_df.iloc[0]["ordinal_position"]
            )
            self.create_column_node(
                column_name,
                column_df.iloc[0]["table_name"],
                data_type,
                id=id,
                description=description,
                is_nullable=is_nullable,
                ordinal_position=ordinal_position,
            )
            column_node = self.get_column_node(column_df.iloc[0]["column_name"], table_name)
            table_node = self.get_table_node(table_name)
            self.add_column_to_table(table_node, column_node, ordinal_position)
        return self.column_nodes[(column_name_lower, table_name_lower)]

    def table_exists(self, table_name):
        table_name_lower = table_name.lower()
        if self.tables_df is not None:
            result = self.tables_df.loc[self.tables_df["table_name_lower"] == table_name_lower].replace(np.nan, None)
            if result.empty:
                return table_name.lower() in self.table_nodes
            return True
        return table_name_lower in self.table_nodes

    def create_table_node(
        self,
        table_name,
        id=None,
        created=None,
        description=None,
    ):
        table_name_lower = table_name.lower()
        props = {"name": table_name}
        match_props = {"id": id}
        props = self.update_table_props_by_arguments(
            props,
            created,
            description,
        )
        table_node = Neo4jNode(
            name=table_name,
            label=Labels.TABLE,
            props=props,
            existing_id=id,
            match_props=match_props,
        )
        self.table_nodes.update({table_name_lower: table_node})
        if str(table_node.get_id()) not in self.id_to_node:
            self.id_to_node.update({str(table_node.get_id()): f"{self.schema_name}.{table_name}"})

    def get_table_node_props(self, table_name):
        table_name_lower = table_name.lower()
        table_df = self.tables_df.loc[self.tables_df["table_name_lower"] == table_name_lower].replace(np.nan, None)

        if table_df.empty:
            raise Exception(f"Table {table_name} is not in schema {self.schema_name}.")

        table = table_df.to_dict(orient="records")[0]
        return table["props"]

    def get_table_node_match_props(self, table_name):
        table_name_lower = table_name.lower()
        table_df = self.tables_df.loc[self.tables_df["table_name_lower"] == table_name_lower].replace(np.nan, None)

        if table_df.empty:
            raise Exception(f"Table {table_name} is not in schema {self.schema_name}.")

        table = table_df.to_dict(orient="records")[0]
        return table["match_props"]

    def get_table_node(self, table_name):
        table_name_lower = table_name.lower()
        if table_name_lower not in self.table_nodes:
            table_df = self.tables_df.loc[self.tables_df["table_name_lower"] == table_name_lower].replace(np.nan, None)

            if table_df.empty:
                raise ValueError(f"Table {table_name} is not in schema {self.schema_name}.")

            id = table_df.iloc[0]["props"]["id"]
            created = (
                None
                if "created" not in table_df.iloc[0] or pd.isna(table_df.iloc[0]["created"])
                else table_df.iloc[0]["created"]
            )
            description = (
                None
                if "description" not in table_df.iloc[0] or pd.isna(table_df.iloc[0]["description"])
                else table_df.iloc[0]["description"]
            )
            self.create_table_node(
                table_df.iloc[0]["table_name"],
                id,
                created,
                description,
            )
        return self.table_nodes[table_name_lower]

    def get_schema_node(self):
        return self.schema_node

    def create_schema_node(self, schema_name, id=None):
        if self.schema_node is None:
            self.schema_name = schema_name
            props = {"name": schema_name}
            match_props = {
                "db_name": self.get_db_name(),
                "name": schema_name,
            }
            self.schema_node = Neo4jNode(
                name=schema_name,
                label=Labels.SCHEMA,
                props=props,
                existing_id=id,
                match_props=match_props,
            )

    def is_column_in_table(self, table_node, column_name):
        column_name_lower = column_name.lower()
        table_name_lower = table_node.name.lower()

        if self.columns_df is not None:
            return table_name_lower in self.columns_map and column_name_lower in self.columns_map[table_name_lower]
        else:
            if table_node in self.tables_to_columns:
                if column_name_lower in self.tables_to_columns[table_node]:
                    return True
            return False

    def replace_id(self, old_id, new_id):
        if old_id in self.id_to_node:
            name = self.id_to_node.pop(old_id)
            self.id_to_node.update({new_id: name})
        # the replace will work only for the dataframe that contains the old_id value
        table = self.tables_df.loc[self.tables_df["id"] == old_id].replace(np.nan, None)
        if not table.empty:
            table_props = table.iloc[0]["props"]
            table_props["id"] = new_id
            self.tables_df.replace(old_id, new_id, inplace=True)
        column = self.columns_df.loc[self.columns_df["id"] == old_id].replace(np.nan, None)
        if not column.empty:
            column_props = column.iloc[0]["props"]
            column_props["id"] = new_id
            self.columns_df.replace(old_id, new_id, inplace=True)

    def get_node_by_id(self, id):
        if id in self.id_to_node.keys():
            return self.id_to_node[id]
        return None

    def update_table_props_by_arguments(
        self,
        props,
        created,
        description,
    ):
        if created:
            props.update({"created": created})
        if description:
            props.update({"description": description})
        return props

    def update_column_props_by_arguments(
        self,
        props,
        data_type,
        description,
        is_nullable,
        ordinal_position,
    ):
        if data_type:
            props.update({"data_type": data_type})
        if description:
            props.update({"description": description})
        if is_nullable:
            props.update({"is_nullable": is_nullable})
        if not pd.isna(ordinal_position):
            props.update({"ordinal_position": ordinal_position})
        return props
