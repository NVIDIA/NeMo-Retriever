from functools import reduce
import traceback
from concurrent.futures import ThreadPoolExecutor
from shared.graph.model.reserved_words import Labels
from shared.graph.dal.utils_dal import get_node_properties_by_id
from .lineage_dal import (
    get_lineage_base,
    get_lineage_data,
    children_relationship,
    get_account_bi_connectors,
    get_bi_lineage_of_table,
    get_datasources_lineage_of_table,
    handle_node_type,
)
import logging

DEFAULT_CHUNK_SIZE = 100
logger = logging.getLogger("lineage_service")


def to_chunk(chunks: list[list[str]], current: dict[str, str]):
    if not len(chunks):
        chunks.append([current["id"]])
    else:
        last_chunk = chunks[len(chunks) - 1]
        if len(last_chunk) > DEFAULT_CHUNK_SIZE:
            chunks.append([current["id"]])
        else:
            last_chunk.append(current["id"])
    return chunks


def get_lineage(
    account: str,
    root_id: str,
    node_labels: list[Labels],
    child_id: str = None,
    level: int = None,
    disable_upstream: bool = False,
):
    base = get_lineage_base(account, root_id, node_labels)
    props = get_node_properties_by_id(account, root_id, base["label"])
    data = get_lineage_data(account, root_id, props["label"], base["connector_type"])
    root = base | props | data
    if base["label"] in children_relationship:
        upstream = []
        downstream = []
        if not disable_upstream:
            try:
                upstream = get_lineage_tree(
                    account,
                    root_id,
                    base["label"],
                    data["columns"],
                    "upstream",
                    level,
                    child_id,
                )
            except Exception:
                logger.error(traceback.format_exc())
                upstream = {}
        try:
            downstream = get_lineage_tree(
                account,
                root_id,
                base["label"],
                data["columns"],
                "downstream",
                level,
                child_id,
            )
        except Exception:
            logger.error(traceback.format_exc())
            downstream = {}
    if upstream and "upstream" in upstream:
        root["upstream"] = upstream["upstream"]
    else:
        root["upstream"] = []
    if downstream and "downstream" in downstream:
        root["downstream"] = downstream["downstream"]
    else:
        root["downstream"] = []

    upstream_cols = (
        []
        if not root["upstream"] or "columns_ids" not in upstream
        else upstream["columns_ids"]
    )
    downstream_cols = (
        []
        if not root["downstream"] or "columns_ids" not in downstream
        else downstream["columns_ids"]
    )
    upstream_sql_ids = (
        [] if not root["upstream"] or "sql_id" not in upstream else upstream["sql_id"]
    )
    root["columns_ids"] = list(set(upstream_cols + downstream_cols))
    root["sql_id"] = upstream_sql_ids
    handle_node_type(root)
    return root


def get_lineage_single_direction(
    account_id: str,
    root_id: str,
    root_label: str,
    direction: str,
    columns_ids: list[str],
    level: int = None,
):
    connections = get_account_bi_connectors(account_id)
    datasources_lineage_matrix = get_datasources_lineage_of_table(
        account_id, root_id, root_label, direction, columns_ids, level
    )
    if (
        root_label == "table" or root_label == "temp_table"
    ) and direction == "upstream":
        return datasources_lineage_matrix
    for connector in connections:
        datasources_lineage_matrix = get_bi_lineage_of_table(
            account_id,
            root_id,
            root_label,
            connector,
            direction,
            level,
            datasources_lineage_matrix if datasources_lineage_matrix else [],
            columns_ids,
        )
    return datasources_lineage_matrix


def get_lineage_tree(
    account: str,
    root_id: str,
    root_label: str,
    columns_of_root: list[dict[str, str]],
    direction: str,
    level: int = None,
    root_column: str = None,
):
    if root_column:
        chunks = [[root_column]]
    elif len(columns_of_root) <= DEFAULT_CHUNK_SIZE:
        columns_ids = [c["id"] for c in columns_of_root]
        chunks = [columns_ids]
    else:
        chunks = reduce(to_chunk, columns_of_root, [])
    direction_matrix = []
    with ThreadPoolExecutor(2) as executor:
        for matrix in executor.map(
            lambda chunk: get_lineage_single_direction(
                account, root_id, root_label, direction, chunk, level
            ),
            chunks,
        ):
            if len(matrix):
                direction_matrix = merge_matrixes(direction_matrix, matrix, direction)
    if direction_matrix:
        return create_lineage_tree_from_lineage_matrix(
            direction_matrix, direction, root_id
        )
    else:
        return {}


def get_matrix_count(matrix: list[dict[str, str]]):
    return reduce(
        lambda count, current_level: count + len(current_level.keys()), matrix, 0
    )


def merge_matrixes(
    matrix_to: list[dict[str, str]], matrix_from: list[dict[str, str]], direction: str
):
    if not matrix_to:
        return matrix_from
    for level, tables_in_level in enumerate(matrix_from):
        if level > len(matrix_to) - 1:
            matrix_to.append(tables_in_level)
            continue
        for table_id in tables_in_level.keys():
            if table_id in matrix_to[level]:
                children = matrix_to[level][table_id][direction]
                matrix_to[level][table_id][direction] = list(
                    set(children + tables_in_level[table_id][direction])
                )
                columns = matrix_to[level][table_id]["columns_ids"]
                matrix_to[level][table_id]["columns_ids"] = list(
                    set(columns + tables_in_level[table_id]["columns_ids"])
                )
                sql_ids = matrix_to[level][table_id]["sql_id"]
                matrix_to[level][table_id]["sql_id"] = list(
                    set(sql_ids + tables_in_level[table_id]["sql_id"])
                )
            else:
                matrix_to[level][table_id] = tables_in_level[table_id]
    return matrix_to


def create_lineage_tree_from_lineage_matrix(
    matrix: list[dict[str, dict[str, str | list[str]]]], direction: str, root_id: str
):
    max_level = len(matrix) - 1

    def create_tree_node(current_id: str, current_level: int):
        node = matrix[current_level][current_id].copy()
        if current_level == max_level:
            node[direction] = []
            return node
        else:
            direction_ids = list(set(node[direction]))
            node[direction] = [
                create_tree_node(child_id, current_level + 1)
                for child_id in direction_ids
            ]
            return node

    tree = create_tree_node(root_id, 0)
    return tree
