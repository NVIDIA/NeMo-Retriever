import logging
from enum import StrEnum
from itertools import groupby
import os
import re

from langchain_community.vectorstores import PGVector
from langchain_nvidia_ai_endpoints import ChatNVIDIA


logger = logging.getLogger(__name__)

# Load .env from current working directory so LLM_API_KEY, LLM_INVOKE_URL are set (run from repo root)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ==================== CONSTANTS ====================

# Hard ceiling on how many candidate snippets we want to reason over for a single question.
# Larger numbers tend to confuse the LLM and increase latency.
MAX_CALCULATION_CANDIDATES = 15

# k for each get_semantic_candidates_information call (qnv, qwv, per-entity).
SEMANTIC_CANDIDATES_K = 10

# Per-attempt batch size when pulling semantic candidates. We keep this modest
# to balance recall with response time and to leave room for deduping.
ATTR_CANDIDATE_BATCH_SIZE = 6

# Maximum number of batch pulls before we give up looking for fresh attributes;
# empirically, three rounds either fill the quota or show the backend is stuck on duplicates.
ATTR_CANDIDATE_MAX_ATTEMPTS = 3

# Budget for entity-specific candidate searches. This limits how many candidates we fetch
# per entity when ensuring entity coverage. Distributed evenly across entities (at least 1 per entity).
# This prevents over-fetching when there are many entities while ensuring each gets coverage.
ENTITY_CANDIDATE_BUDGET = 4


QUERIES_USAGE_PERCENTILE = "queries_usage_percentile"
TABLES_USAGE_PERCENTILE = "tables_usage_percentile"
COLUMNS_USAGE_PERCENTILE = "columns_usage_percentile"





def _make_llm() -> ChatNVIDIA:
    # Prefer LLM_API_KEY; fall back to NVIDIA_API_KEY (used by LangChain NVIDIA docs)
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    return ChatNVIDIA(
        base_url=os.environ.get("LLM_INVOKE_URL"),
        api_key=api_key,
        model=os.environ.get("LLM_MODEL", "meta/llama-3.1-70b-instruct"),
    )

    
class Labels(StrEnum):
    """Semantic labels used by omni-lite candidate retrieval."""

    CUSTOM_ANALYSIS = "custom_analysis"
    COLUMN = "column"

def store_usage_percentiles(
    percentiles_type_name: str,
    usage_percentile_25: int,
    usage_percentile_75: int,
):
    query = """
            MATCH (n:db)
            WITH n
            CALL apoc.create.setProperties(n, [$percentiles_type_name_25, $percentiles_type_name_75], [$usage_percentile_25, $usage_percentile_75]) 
            YIELD node
            RETURN n
            """
    conn.query_write(
        query=query,
        parameters={
            "percentiles_type_name_25": f"{percentiles_type_name}_25",
            "percentiles_type_name_75": f"{percentiles_type_name}_75",
            "usage_percentile_25": usage_percentile_25,
            "usage_percentile_75": usage_percentile_75,
        },
    )



def get_stored_usage_percentiles(percentiles_type_name: str):
    query = f"""
                MATCH (n:db)
                RETURN n.{f"{percentiles_type_name}_25"} as usage_percentile_25, n.{f"{percentiles_type_name}_75"} as usage_percentile_75
                """
    results = conn.query_read_only(
        query=query,
        parameters={
        },
    )
    return results

def get_count_str_by_month(alias: str):
    current = date.today().replace(day=1)
    count_3_month = []

    for i in range(0, 3):
        count_3_month.append(f"coalesce({alias}.cnt_{current.month}_{current.year}, 0)")
        prev = current - timedelta(days=1)
        current = prev.replace(day=1)

    count_str = "+".join(count_3_month)
    return count_str

def init_queries_usage_percentiles():
    count_string = get_count_str_by_month("n")
    query_all = f"""match(n:sql{{is_sub_select:FALSE}})
                     return collect({count_string}) as usages """
    usages_result = conn.query_read_only(
        query=query_all, parameters={}
    )
    usages = usages_result[0]["usages"]
    if len(usages) == 0:
        return 0, 0
    usage_percentile_25 = np.percentile(usages, 25)
    usage_percentile_75 = np.percentile(usages, 75)
    store_usage_percentiles(
        QUERIES_USAGE_PERCENTILE, usage_percentile_25, usage_percentile_75
    )
    return usage_percentile_25, usage_percentile_75


def get_usage_percentiles():
    stored_percentiles = get_stored_usage_percentiles(
        QUERIES_USAGE_PERCENTILE
    )
    if len(stored_percentiles) == 0 or (
        stored_percentiles[0]["usage_percentile_25"] is None
    ):
        usage_percentile_25, usage_percentile_75 = init_queries_usage_percentiles()
    else:
        usage_percentile_25 = stored_percentiles[0]["usage_percentile_25"]
        usage_percentile_75 = stored_percentiles[0]["usage_percentile_75"]

    return usage_percentile_25, usage_percentile_75


def get_queries_usage_percentiles(node_str="node"):
    count_str = get_count_str_by_month(node_str)
    usage_percentile_25, usage_percentile_75 = get_usage_percentiles()
    return usage_percentile_25, usage_percentile_75, count_str


def clean_results(raw_candidates: list[dict]) -> list[dict]:
    """
    Normalize raw semantic hits: require id, dedupe by (label, id), preserve order.
    """
    out: list[dict] = []
    seen: set[tuple[str | None, str]] = set()
    for c in raw_candidates or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if cid is None:
            continue
        key = (c.get("label"), str(cid))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def update_candidate_properties(candidate: dict, props_by_id: dict) -> None:
    """Merge graph-expanded properties from ``expand_info`` into ``candidate``."""
    cid = candidate.get("id")
    if cid is None:
        return
    extra = props_by_id.get(cid) or props_by_id.get(str(cid))
    if isinstance(extra, dict):
        # values in ``extra`` may be list-wrapped from groupBy; take first dict if so
        candidate.update(extra)


def expand_info(ids_and_labels):
    # define a function for key
    def key_func(k):
        return k["label"]

    results = {}

    (
        queries_percentile_25,
        queries_percentile_75,
        _cnt_str,
    ) = get_queries_usage_percentiles("sql_node")

    for label, ids in groupby(ids_and_labels, key_func):
        label_id_pairs_for_current_label = list(ids)
        query = f"""UNWIND $label_id_pairs as label_id
                    MATCH (n:{label} {{id:label_id.id}})
                    CALL apoc.case([
                        n:custom_analysis,
                            'MATCH(n)-[:analysis_of]->(sql:sql)
                            WITH n, collect(distinct {{sql_code: sql.sql_full_query}}) as sql
                            RETURN apoc.map.setKey(properties(n), "sql", sql) as item',
                        n:column,
                            'MATCH(n)<-[:schema]-(parent)
                            RETURN apoc.map.setPairs(properties(n),[["table_name", parent.name],["table_type", parent.type], ["parent_id", parent.id]]) as item'
                        ],
                        'with n RETURN n{{ .*}} as item ',
                        {{n:n, sql_type: $sql_type, usage_percentile_25: $usage_percentile_25,
                        usage_percentile_75: $usage_percentile_75, {queries_for_columns_params_keys} }}
                        )
                    YIELD value as response
                    WITH collect(response.item) as all_items
                    RETURN apoc.map.groupBy(all_items,'id') as ids_to_props
                    """
        params = {
            "label_id_pairs": label_id_pairs_for_current_label,
            "sql_type": SQLType.QUERY,
            "usage_percentile_25": queries_percentile_25,
            "usage_percentile_75": queries_percentile_75,
        }
        params.update(queries_for_columns_params)
        result = conn.query_read_only(
            query=query,
            parameters=params,
        )
        if len(result) > 0:
            results = results | result[0]["ids_to_props"]

    return results


def search_vector_index(
    index_name,
    allowed_ids,
    user_question,
    k=30,
    label_filter: list[str] | None = None,
):
    vectorstore = PGVector.from_existing_index(
        embedding=get_embeddings(account_id),
        collection_name=index_name,
        connection_string=pg_conn.connection_string,
    )

    filter_kwargs = {}
    if label_filter:
        filter_kwargs["label"] = {"in": label_filter}

    # PGVector returns distance here; convert to a similarity-style score where
    # higher is better by applying (1 - distance).
    result = vectorstore.similarity_search_with_score(
        user_question,
        k=k,
        filter=filter_kwargs,
    )
    # double check for account_id
    return result


def get_semantic_candidates_information(
    entity: str,
    k: int = 5,
    threshold: float = 0,
    list_of_semantic: list | None = None,
):
    """
    Vector search over indexed nodes, then merge graph properties from ``expand_info``.

    Matches the call shape used by ``extract_candidates``:
    ``get_semantic_candidates_information(text, k=..., list_of_semantic=[...])``.
    """
    if list_of_semantic is None:
        list_of_semantic = [Labels.CUSTOM_ANALYSIS, Labels.COLUMN]
    labels = list_of_semantic
    results: list[dict] = []

    nodes_results = search_vector_index(
        "nodes_vector_store",
        entity,
        k=k,
        label_filter=labels,
    )
    results.extend([item["metadata"] for item in nodes_results])

    ids_and_labels = [{"label": x["label"], "id": x["id"]} for x in results]
    props_by_id = expand_info(ids_and_labels)
    for c in results:
        cid = c.get("id")
        if cid is None:
            continue
        extra = props_by_id.get(cid) or props_by_id.get(str(cid))
        if isinstance(extra, dict):
            c.update(extra)

    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results


def _dedupe_best_score_sort_cap(combined: list[dict]) -> list[dict]:
    """Deduplicate by (label, id), keep best score, sort descending, cap."""
    best_by_key: dict[tuple[str | None, str], dict] = {}
    for c in combined:
        cid = c.get("id")
        if cid is None:
            continue
        key = (c.get("label"), str(cid))
        score = float(c.get("score") or 0)
        prev = best_by_key.get(key)
        if prev is None or score > float(prev.get("score") or 0):
            best_by_key[key] = c

    unique = list(best_by_key.values())
    unique.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return unique[:MAX_CALCULATION_CANDIDATES]


def extract_candidates(
    entities: list[str],
    query_no_values: str,
    query_with_values: str = "",
) -> tuple[list[dict], list[dict]]:
    """
    One semantic search per string: ``query_no_values``, ``query_with_values`` (if distinct),
    and each entity name. For each string, pull custom analyses and columns via
    ``get_semantic_candidates_information``. Merge streams, dedupe by (label, id) keeping best score,
    sort by score, cap at ``MAX_CALCULATION_CANDIDATES`` per stream.

    Returns:
        ``(custom_analysis_candidates, column_candidates)``
    """
    qnv_text = (query_no_values or "").strip()
    qwv_text = (query_with_values or "").strip()

    pulls: list[str] = []
    if qnv_text:
        pulls.append(qnv_text)
    if qwv_text and qwv_text != qnv_text:
        pulls.append(qwv_text)
    for ent in entities or []:
        t = (ent or "").strip()
        if t:
            pulls.append(t)

    combined_custom: list[dict] = []
    combined_columns: list[dict] = []
    for text in pulls:
        t = (text or "").strip()
        if not t:
            continue
        combined_custom.extend(
            get_semantic_candidates_information(
                t,
                k=SEMANTIC_CANDIDATES_K,
                list_of_semantic=[Labels.CUSTOM_ANALYSIS],
            )
            or []
        )
        combined_columns.extend(
            get_semantic_candidates_information(
                t,
                k=SEMANTIC_CANDIDATES_K,
                list_of_semantic=[Labels.COLUMN],
            )
            or []
        )

    out_custom = _dedupe_best_score_sort_cap(combined_custom)
    out_columns = _dedupe_best_score_sort_cap(combined_columns)

    logger.info(
        f"extract_candidates: {len(out_custom)} custom_analysis, {len(out_columns)} column "
        f"(max {MAX_CALCULATION_CANDIDATES} each), {len(pulls)} pulls"
    )

    return out_custom, out_columns



def get_semantic_entities_ids(items):
    """
    Filter semantic items by classification flag and return their IDs.

    Args:
        items: list of ItemScore objects (or dicts) like
            [{'id': '...', label: 'custom_analysis', 'classification': true}, ...]
            or ItemScore(id=..., label=..., classification=True)

    Returns:
        list of dictionaries with 'id' and 'label' for items where classification is True
    """
    if not items:
        return []

    def _get(obj, key, default=None):
        """Safe getter for both Pydantic-style objects and plain dicts."""
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    classified_ids_and_labels = []
    for item in items:
        is_relevant = bool(_get(item, "classification", False))
        if not is_relevant:
            continue
        item_id = _get(item, "id")
        item_label = _get(item, "label")
        if item_id and item_label:
            classified_ids_and_labels.append({"id": item_id, "label": item_label})

    return classified_ids_and_labels



def extract_entities_with_id_name_label(data):
    result = {}

    def recurse(obj):
        if isinstance(obj, dict):
            # Main entity case: id + name + (type or label)
            if "id" in obj and "name" in obj and ("type" in obj or "label" in obj):
                if "type" in obj:
                    label = type_to_labels(obj["type"])
                    final_label = label[0] if label else obj["type"]
                else:
                    final_label = obj["label"]

                result[obj["id"]] = (
                    obj["name"],
                    final_label,
                    obj.get("parent_id"),
                )

            # explicitly capture table inside column
            if "table" in obj and isinstance(obj["table"], dict):
                table = obj["table"]
                if "id" in table and "name" in table:
                    result[table["id"]] = (table["name"], "table", None)

            # Continue recursion
            for value in obj.values():
                recurse(value)

        elif isinstance(obj, list):
            for item in obj:
                recurse(item)

    recurse(data)
    return result



def highlight_entity(items_present: dict, text: str, account_id: str) -> str:
    """
    Processes [[[entity]]] patterns in the text.

    Supported formats:
    - [[[name/id]]]
    - [[[label/id|display_name]]]

    Replaces valid ones with hyperlinks using `prepare_link`.
    Falls back to bolding just the entity name if invalid or not found.
    """

    def replace_entity(match):
        raw = match.group(1)
        cleaned = re.sub(r"\s*/\s*", "/", raw.strip())  # Remove whitespace around slash

        # Handle display name (e.g., [[[label/id|display_name]]])
        if "|" in cleaned:
            entity_part, display_name = cleaned.split("|", 1)
            display_name = display_name.strip()
        else:
            entity_part, display_name = cleaned, None

        # Now parse entity part → name/ID
        if "/" in entity_part:
            name_or_label, eid = entity_part.split("/", 1)
            name_or_label = name_or_label.strip()
            eid = eid.strip()

            # Lookup by ID
            entity = items_present.get(eid)
            if entity and (
                entity[0].lower() == name_or_label.lower()
                or (display_name and entity[0].lower() == display_name.lower())
            ):
                shown_name = display_name or name_or_label
                return f"<{prepare_link(shown_name, eid, entity[1], entity[2])}>"
            elif entity:
                logger.warning(
                    "ASSUMPTION: the name in link is not found correctly by llm, take it by id from candidates"
                )
                return f"<{prepare_link(entity[0], eid, entity[1], entity[2])}>"

            # no entity in candidates
            try:
                item = get_item_by_id(account_id, eid, name_or_label)
            except Exception:
                logger.error("Something not ok with id, error raised")
                return f"*{display_name or name_or_label}*"

            if item:
                return f"<{prepare_link(item['name'], eid, name_or_label)}>"
            else:
                logger.warning(
                    f"Entity ID mismatch or not found: {name_or_label}/{eid}"
                )
                return f"*{display_name or name_or_label}*"
        else:
            logger.warning(f"No ID found in entity: {cleaned}")
            return f"*{cleaned}*"

    return re.sub(r"\[\[\[(.*?)\]\]\]", replace_entity, text)



def format_response(account_id, candidates, response):
    final_response_formatted = response.replace("%%%", "```").replace("**", "*")
    final_response_formatted = re.sub(r"(\\+n|\n)", "\n ", final_response_formatted)
    all_entities_present = extract_entities_with_id_name_label(candidates)

    try:
        final_response_highlighted = highlight_entity(
            all_entities_present, final_response_formatted, account_id
        )
    except Exception:
        return final_response_formatted
    return final_response_highlighted


# TODO how to get fks from duckdb?
def get_relevant_fks(tables_ids, account_id):
    # Build a connected graph by expanding from target tables through FK relationships
    query = """ 
    // Start with target tables and expand outward to find connected tables
    WITH $tables_ids as current_ids
    
    // Level 1: Find tables connected via FK
    OPTIONAL MATCH (t0:table{account_id:$account_id} WHERE t0.id IN current_ids)
          -[:schema]->(:column)-[:fk]-(:column)<-[:schema]-(t1:table{account_id:$account_id})
    WITH current_ids, collect(DISTINCT t1.id) as new_ids_1
    WITH current_ids + new_ids_1 as level_1_ids
    
    // Level 2
    OPTIONAL MATCH (t1:table{account_id:$account_id} WHERE t1.id IN level_1_ids)
          -[:schema]->(:column)-[:fk]-(:column)<-[:schema]-(t2:table{account_id:$account_id})
    WITH level_1_ids, collect(DISTINCT t2.id) as new_ids_2
    WITH level_1_ids + new_ids_2 as level_2_ids
    
    // Level 3
    OPTIONAL MATCH (t2:table{account_id:$account_id} WHERE t2.id IN level_2_ids)
          -[:schema]->(:column)-[:fk]-(:column)<-[:schema]-(t3:table{account_id:$account_id})
    WITH level_2_ids, collect(DISTINCT t3.id) as new_ids_3
    WITH level_2_ids + new_ids_3 as all_table_ids
    
    // Get all FK relationships between these tables
    MATCH (t1:table{account_id:$account_id})-[:schema]->(col1:column)-[:fk]-(col2:column)<-[:schema]-(t2:table{account_id:$account_id})
    WHERE t1.id IN all_table_ids AND t2.id IN all_table_ids
      AND t1.id < t2.id  // Avoid duplicates by keeping only one direction
    
    RETURN collect(DISTINCT {
        table1: t1.schema_name + '.' + t1.name, 
        column1: col1.name, 
        column1_datatype: coalesce(col1.data_type, 'None'), 
        table2: t2.schema_name + '.' + t2.name, 
        column2: col2.name, 
        column2_datatype: coalesce(col2.data_type, 'None')
    }) as list_of_foreign_keys
    """
    results = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "tables_ids": tables_ids,
        },
    )
    if len(results) > 0:
        result_fks = results[0]["list_of_foreign_keys"]
    else:
        result_fks = []

    # Build a connected graph by expanding from target tables through FK relationships
    query = """ 
    // Start with target tables and expand outward to find connected tables
    
    // Level 1: Find tables connected via FK
    OPTIONAL MATCH (t0:table{account_id:$account_id} WHERE t0.id IN $tables_ids)-[:join]-(t1:table{account_id:$account_id})
    WITH collect(DISTINCT t1.id) as new_ids_1
    WITH $tables_ids + new_ids_1 as level_1_ids
    
    // Level 2
    OPTIONAL MATCH (t1:table{account_id:$account_id} WHERE t1.id IN level_1_ids)-[:join]-(t2:table{account_id:$account_id})
    WITH level_1_ids, collect(DISTINCT t2.id) as new_ids_2
    WITH level_1_ids + new_ids_2 as level_2_ids
    
    // Level 3
    OPTIONAL MATCH (t2:table{account_id:$account_id} WHERE t2.id IN level_2_ids)-[:join]-(t3:table{account_id:$account_id})
    WITH level_2_ids, collect(DISTINCT t3.id) as new_ids_3
    WITH level_2_ids + new_ids_3 as all_table_ids
    
    // Get all join relationships between these tables and parse the join property
    MATCH (t1:table{account_id:$account_id})-[rel:join]-(t2:table{account_id:$account_id})
    WHERE t1.id IN all_table_ids AND t2.id IN all_table_ids
      AND t1.id < t2.id  // Avoid duplicates by keeping only one direction
      AND rel.join IS NOT NULL
    
    // Parse the join property: split by operators and extract left/right sides
    WITH t1, t2, rel,
         trim(apoc.text.split(rel.join, '<=|>=|=|<|>')[0]) as left_side,
         trim(apoc.text.split(rel.join, '<=|>=|=|<|>')[1]) as right_side
    
    // Parse left side: SCHEMA.TABLE.COLUMN (handle potential whitespace)
    WITH t1, t2, rel, left_side, right_side,
         trim(split(left_side, '.')[0]) as left_schema,
         trim(split(left_side, '.')[1]) as left_table,
         trim(split(left_side, '.')[2]) as left_column,
         trim(split(right_side, '.')[0]) as right_schema,
         trim(split(right_side, '.')[1]) as right_table,
         trim(split(right_side, '.')[2]) as right_column
    WHERE left_schema IS NOT NULL AND left_table IS NOT NULL AND left_column IS NOT NULL
      AND right_schema IS NOT NULL AND right_table IS NOT NULL AND right_column IS NOT NULL
    
    // Match the actual column nodes for left side
    OPTIONAL MATCH (s1:schema{account_id:$account_id, name: left_schema})-[:schema]->(tbl1:table{account_id:$account_id, name: left_table})-[:schema]->(col1:column{account_id:$account_id, name: left_column})
    
    // Match the actual column nodes for right side
    OPTIONAL MATCH (s2:schema{account_id:$account_id, name: right_schema})-[:schema]->(tbl2:table{account_id:$account_id, name: right_table})-[:schema]->(col2:column{account_id:$account_id, name: right_column})
    
    // Return the structured format
    RETURN collect(DISTINCT {
        table1: t1.schema_name + '.' + t1.name, 
        column1: coalesce(col1.name, left_column), 
        column1_datatype: coalesce(col1.data_type, 'None'), 
        table2: t2.schema_name + '.' + t2.name, 
        column2: coalesce(col2.name, right_column), 
        column2_datatype: coalesce(col2.data_type, 'None')
    }) as list_of_foreign_keys
    """
    results = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "tables_ids": tables_ids,
        },
    )
    if len(results) > 0:
        result_joins = results[0]["list_of_foreign_keys"]
    else:
        result_joins = []
    results = result_fks + result_joins

    # Convert to JSON strings, use set to remove duplicates, then convert back
    unique_strings = set(json.dumps(d, sort_keys=True) for d in results)
    unique_results = [json.loads(s) for s in unique_strings]

    key_order = [
        "table1",
        "column1",
        "column1_datatype",
        "table2",
        "column2",
        "column2_datatype",
    ]
    sorted_results = [{key: d[key] for key in key_order} for d in unique_results]
    return sorted_results




def get_relevant_tables(
    account_d,
    initial_question,
    k=15,
    exclude_ids: list[str] | None = None,
):
    account_simple_str = account_id.replace("-", "_") # TODO no account
    try:
        if exclude_ids:
            allowed_tables_ids = [
                table_id
                for table_id in allowed_tables_ids
                if table_id not in exclude_ids
            ]
        relevant_tables = search_vector_index(
            account_id,
            index_name=f"{account_simple_str}_tables_info_vector_store",
            allowed_ids=allowed_tables_ids,
            user_question=initial_question,
            k=k,
            # label_filter=[Labels.TABLE],    add the filter? reduce runtime?
        )
    except Exception:
        relevant_tables = []

    def parse_table_text(text: str) -> dict:
        """Parse db_name, schema_name, and columns from table text."""
        parsed = {}
        try:
            import re

            # Ensure text is a string
            if not isinstance(text, str):
                return parsed

            # Extract db_name
            db_match = re.search(r"db_name:\s*([^,]+)", text)
            if db_match:
                parsed["db_name"] = db_match.group(1).strip()

            # Extract schema_name
            schema_match = re.search(r"schema_name:\s*([^,]+)", text)
            if schema_match:
                parsed["schema_name"] = schema_match.group(1).strip()

            # Extract columns
            columns_match = re.search(r"columns:\s*(.+)$", text)
            if columns_match:
                columns_str = columns_match.group(1).strip()
                # Parse column blocks like {name: X, data_type: Y, description: Z}
                column_pattern = r"\{name:\s*([^,}]+)(?:,\s*data_type:\s*([^,}]+))?(?:,\s*description:\s*([^}]+))?\}"
                columns = []
                for match in re.finditer(column_pattern, columns_str):
                    column = {
                        "name": match.group(1).strip(),
                    }
                    if match.group(2):
                        column["data_type"] = match.group(2).strip()
                    if match.group(3):
                        desc = match.group(3).strip()
                        if desc != "null":
                            column["description"] = desc
                    columns.append(column)
                if columns:
                    parsed["columns"] = columns
        except Exception:
            # If parsing fails, return empty dict (fallback to table_info)
            pass

        return parsed

    relevant_tables_list = [
        {
            "name": f"{table['metadata']['name']}",
            "label": f"{table['metadata']['label']}",
            "id": f"{table['metadata']['id']}",
            "table_info": f"{table['text']}",
            **(
                {"primary_key": table["metadata"]["pk"]}
                if "pk" in table["metadata"]
                else {}
            ),
            **parse_table_text(table["text"]),
        }
        for table in relevant_tables
    ]
    relevant_fks = get_relevant_fks([x["id"] for x in relevant_tables_list], account_id)
    for table in relevant_tables_list:
        for fk in relevant_fks:
            if table["name"] == fk["table1"]:
                table["foreign_key"] = (
                    f"'{table['name']}.{fk['column1']}' = '{fk['table2']}.{fk['column2']}'"
                )

    return relevant_tables_list, relevant_fks

