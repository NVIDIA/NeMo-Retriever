from datetime import datetime, timezone
import logging
import time

logger = logging.getLogger(__name__)

from db.dal import (
    db_exists,
    update_node_property,
    delete_schema,
    update_disconnected_sqls,
    update_diff_from_existing_schema,
)
from graph.indexes import add_indices
from concurrent.futures import ThreadPoolExecutor

from graph.parsers import schemas_parser
from graph.dal.schemas_dal import (
    get_schemas_ids_and_names,
    add_fks,
    add_pks,
    delete_old_fks,
    reset_pks,
)
from graph.parsers.sql.parse_queries_df import populate_views
from graph.services.schema import add_schema

def populate_structured_data(
    data,
    account_id,
    num_workers,
    dialect,
    keep_string_values
):
    logger.info("Using Dialect: " + dialect)

    # Make sure that the indices exist in the graph database
    add_indices()

    all_schemas = {}

    tables_df = data["tables"]
    columns_df = data["columns"]

    # if in a single population there is more than one DB, then the temp schema will be created
    # only for the first one (temp_schema_creation_flag is passed to populate_db function)
    temp_schema_creation_flag = True

    unique_databases = tables_df.database.unique()
    for database in unique_databases:
        sub_tables_df = tables_df.loc[tables_df["database"] == database]
        sub_columns_df = columns_df.loc[columns_df["database"] == database]
        logger.info(f"Started parsing db {database}.")
        schemas, _, added_or_modified_tables = populate_db(
            sub_tables_df,
            sub_columns_df,
            account_id,
            num_workers,
            temp_schema_creation_flag,
        )
        # Temp hack for now
        all_schemas.update(schemas)
        temp_schema_creation_flag = (
            False if temp_schema_creation_flag else temp_schema_creation_flag
        )
    # Do garbage collection after updating the graph:
    # Search for SQLs that got disconnected from the DB tree
    update_disconnected_sqls(account_id)

    if "fks" in data:
        populate_fks(account_id, fks=data["fks"])
    if "pks" in data:
        populate_pks(account_id, pks=data["pks"])

    failed_views = []
    if "views" in data:
        failed_views = populate_views(
            all_schemas,
            data["views"],
            account_id,
            num_workers,
            dialect,
            keep_string_values,
        )
        logger.info(f"Failed views: {len(failed_views)}")

    return failed_views


def populate_db(
    tables_df, columns_df, account_id, num_workers, temp_schema_creation_flag=True
):
    added_or_modified_tables = []
    schemas, db_node = schemas_parser.parse_df(
        tables_df,
        columns_df,
        account_id,
        temp_schema_creation_flag=temp_schema_creation_flag,
    )
    existing_db_id, loaded = db_exists(account_id, db_node)

    latest_timestamp = datetime.now(timezone.utc).replace(microsecond=0)

    if existing_db_id is None or not loaded:
        if existing_db_id is not None:
            db_node.replace_id(existing_db_id)

        before_adding_schemas = time.time()
        for schema_name, schema in schemas.items():
            added_or_modified_tables_dict = {
                "db": str(db_node.id),
                "schema": str(schema.schema_node.name),
            }
            added_or_modified_tables_dict = add_schema(
                schema,
                account_id,
                latest_timestamp,
                num_workers,
                added_or_modified_tables_dict,
            )
            if added_or_modified_tables_dict:
                # if temporary schema, then added_or_modified_tables_dict is None
                added_or_modified_tables.append(added_or_modified_tables_dict)
            logger.info(f"Added schema {schema_name} to db.")

        update_node_property(
            account_id, "db", str(db_node.get_id()), {"pulled": latest_timestamp}
        )

        logger.info(f"Time took to add schemas:{time.time() - before_adding_schemas}")
        return schemas, db_node, added_or_modified_tables

    before_adding_schema = time.time()
    existing_schemas = get_schemas_ids_and_names(account_id, existing_db_id)
    existing_schema_names = [s["schema_name"].lower() for s in existing_schemas]
    new_schemas = schemas.keys()
    schemas_to_add = [
        schema[1]
        for schema in schemas.items()
        if schema[0] in (set(new_schemas) - set(existing_schema_names))
    ]
    for schema in schemas_to_add:
        schema.get_db_node().replace_id(existing_db_id)
        added_or_modified_tables_dict = {
            "db": str(db_node.id),
            "schema": str(schema.schema_node.name),
        }
        added_or_modified_tables_dict = add_schema(
            schema,
            account_id,
            latest_timestamp,
            num_workers,
            added_or_modified_tables_dict,
        )
        added_or_modified_tables.append(added_or_modified_tables_dict)
        logger.info(f"Added schema {schema.get_schema_name()} to db.")

    schemas_to_update = [
        schema[1]
        for schema in schemas.items()
        if schema[0]
        in (set(new_schemas) - (set(new_schemas) - set(existing_schema_names)))
    ]
    for schema in schemas_to_update:
        schema.get_db_node().replace_id(existing_db_id)
    with ThreadPoolExecutor(num_workers) as executor:
        for r in executor.map(
            lambda schema: _update_schema(schema, account_id, latest_timestamp),
            schemas_to_update,
        ):
            if len(r["tables"]) > 0:
                r["db"] = str(db_node.id)
                added_or_modified_tables.append(r)

    # delete existing - new
    schemas_to_delete = [
        schema_name.lower()
        for schema_name in existing_schema_names
        if schema_name.lower() in (set(existing_schema_names) - set(new_schemas))
    ]
    schemas_ids_to_delete = [
        s["schema_id"]
        for s in existing_schemas
        if s["schema_name"].lower() in schemas_to_delete
    ]
    schemas_props_to_delete = [
        {"id": s["schema_id"], "name": s["schema_name"]}
        for s in existing_schemas
        if s["schema_name"].lower() in schemas_to_delete
    ]
    logger.info(f"Deleting schemas: {[s['name'] for s in schemas_props_to_delete]}")
    for schema_id in schemas_ids_to_delete:
        delete_schema(schema_id, account_id)

    logger.info(f"Time took to update schemas:{time.time() - before_adding_schema}")

    update_node_property(account_id, "db", existing_db_id, {"pulled": latest_timestamp})
    return schemas, db_node, added_or_modified_tables


def populate_fks(account_id, fks):
    logger.info("Adding FKs.")
    last_seen = datetime.now()
    add_fks(account_id, fks, last_seen)
    delete_old_fks(account_id, last_seen)


def populate_pks(account_id, pks):
    logger.info("Adding PKs.")
    reset_pks(account_id)
    add_pks(account_id, pks)


def _update_schema(schema, account_id, latest_timestamp):
    added_or_modified_tables = update_diff_from_existing_schema(
        schema, account_id, latest_timestamp
    )
    logger.info(f"Updated schema {schema.get_schema_name()} to db.")
    return added_or_modified_tables
