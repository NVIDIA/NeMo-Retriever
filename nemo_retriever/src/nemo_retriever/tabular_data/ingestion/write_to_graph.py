from datetime import datetime, timezone
import logging
import time

from nemo_retriever.tabular_data.ingestion.graph.dal.db_dal import (
    db_exists,
    update_node_property,
    delete_schema,
    update_diff_from_existing_schema,
)
from nemo_retriever.tabular_data.ingestion.graph.indexes import add_indices
from concurrent.futures import ThreadPoolExecutor
from nemo_retriever.tabular_data.ingestion.graph.parsers import schemas_parser
from nemo_retriever.tabular_data.ingestion.graph.dal.schemas_dal import (
    get_schemas_ids_and_names,
    add_fks,
    add_pks,
    delete_old_fks,
    reset_pks,
)
from nemo_retriever.tabular_data.ingestion.graph.services.schema import add_schema

logger = logging.getLogger(__name__)


def populate_tabular_data(data, num_workers, dialect):
    logger.info("Using Dialect: " + dialect)

    add_indices()

    all_schemas = {}

    tables_df = data["tables"]
    columns_df = data["columns"]

    unique_databases = tables_df.database.unique()
    for database in unique_databases:
        sub_tables_df = tables_df.loc[tables_df["database"] == database]
        sub_columns_df = columns_df.loc[columns_df["database"] == database]
        logger.info(f"Started parsing db {database}.")
        schemas = populate_db(
            sub_tables_df,
            sub_columns_df,
            num_workers,
        )
        all_schemas.update(schemas)

    if "fks" in data:
        populate_fks(fks=data["fks"])
    if "pks" in data:
        populate_pks(pks=data["pks"])

    return []


def populate_db(tables_df, columns_df, num_workers):
    schemas, db_node = schemas_parser.parse_df(tables_df, columns_df)
    existing_db_id, loaded = db_exists(db_node)

    latest_timestamp = datetime.now(timezone.utc).replace(microsecond=0)

    if existing_db_id is None or not loaded:
        if existing_db_id is not None:
            db_node.replace_id(existing_db_id)

        before_adding_schemas = time.time()
        for schema_name, schema in schemas.items():
            add_schema(schema, latest_timestamp, num_workers)
            logger.info(f"Added schema {schema_name} to db.")

        update_node_property("db", str(db_node.get_id()), {"pulled": latest_timestamp})

        logger.info(f"Time took to add schemas:{time.time() - before_adding_schemas}")
        return schemas

    before_adding_schema = time.time()
    existing_schemas = get_schemas_ids_and_names(existing_db_id)
    existing_schema_names = [s["schema_name"].lower() for s in existing_schemas]
    new_schemas = schemas.keys()
    schemas_to_add = [
        schema[1] for schema in schemas.items() if schema[0] in (set(new_schemas) - set(existing_schema_names))
    ]
    for schema in schemas_to_add:
        schema.get_db_node().replace_id(existing_db_id)
        add_schema(schema, latest_timestamp, num_workers)
        logger.info(f"Added schema {schema.get_schema_name()} to db.")

    schemas_to_update = [
        schema[1]
        for schema in schemas.items()
        if schema[0] in (set(new_schemas) - (set(new_schemas) - set(existing_schema_names)))
    ]
    for schema in schemas_to_update:
        schema.get_db_node().replace_id(existing_db_id)
    with ThreadPoolExecutor(num_workers) as executor:
        executor.map(
            lambda schema: _update_schema(schema, latest_timestamp),
            schemas_to_update,
        )

    # delete existing - new
    schemas_to_delete = [
        schema_name.lower()
        for schema_name in existing_schema_names
        if schema_name.lower() in (set(existing_schema_names) - set(new_schemas))
    ]
    schemas_ids_to_delete = [s["schema_id"] for s in existing_schemas if s["schema_name"].lower() in schemas_to_delete]
    schemas_props_to_delete = [
        {"id": s["schema_id"], "name": s["schema_name"]}
        for s in existing_schemas
        if s["schema_name"].lower() in schemas_to_delete
    ]
    logger.info(f"Deleting schemas: {[s['name'] for s in schemas_props_to_delete]}")
    for schema_id in schemas_ids_to_delete:
        delete_schema(schema_id)

    logger.info(f"Time took to update schemas:{time.time() - before_adding_schema}")

    update_node_property("db", existing_db_id, {"pulled": latest_timestamp})
    return schemas


def populate_fks(fks):
    logger.info("Adding FKs.")
    last_seen = datetime.now()
    add_fks(fks, last_seen)
    delete_old_fks(last_seen)


def populate_pks(pks):
    logger.info("Adding PKs.")
    reset_pks()
    add_pks(pks)


def _update_schema(schema, latest_timestamp):
    update_diff_from_existing_schema(schema, latest_timestamp)
    logger.info(f"Updated schema {schema.get_schema_name()} to db.")
