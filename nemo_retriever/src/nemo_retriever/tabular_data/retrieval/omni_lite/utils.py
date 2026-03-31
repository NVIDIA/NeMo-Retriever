from enum import StrEnum
from itertools import groupby

# ==================== CONSTANTS ====================

# Hard ceiling on how many candidate snippets we want to reason over for a single question.
# Larger numbers tend to confuse the LLM and increase latency.
MAX_CALCULATION_CANDIDATES = 9

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



class Labels(StrEnum):
    """Semantic labels used by omni-lite candidate retrieval."""

    CUSTOM_ANALYSIS = "custom_analysis"
    ATTR = "attr"
    METRIC = "metric"
    ANALYSIS = "analysis"
    BT = "bt"


def expand_info(ids_and_labels):
    # define a function for key
    def key_func(k):
        return k["label"]

    results = {}

    (
        queries_percentile_25,
        queries_percentile_75,
        cnt_str,
    ) = get_queries_usage_percentiles("sql_node")

    for label, ids in groupby(ids_and_labels, key_func):
        label_id_pairs_for_current_label = list(ids)
        query = f"""UNWIND $label_id_pairs as label_id
                    MATCH (n:{label} {{id:label_id.id}})
                    CALL apoc.case([
                        n:term,
                            'MATCH (n)-[:term_of]->(attr:attribute)
                            WITH n, collect(properties(attr)) as attributes
                            RETURN apoc.map.setKey(properties(n), "attributes", attributes) as item',
                        n:attribute,
                            'MATCH(term:term)-[:term_of]->(n)
                             CALL {{
                                WITH n
                                MATCH (n)-[snippet:attr_of]->(item:column|error_node|sql|alias)
                                WITH n, snippet, item,
                                     CASE WHEN item:sql OR item:alias THEN TRUE ELSE FALSE END AS complex_snippet
                                CALL apoc.case([
                                        item:column, "match(item)<-[:schema]-(table:table) {get_queries_for_column("table", "item", cnt_str)} return  usage",
                                        item:sql or item:alias , "{get_sql_or_alias_roots(cnt_str)}"
                                        ],
                                    "return 0 AS usage",
                                    {{item:item, account_id:$account_id, usage_percentile_25: $usage_percentile_25,
                                      usage_percentile_75: $usage_percentile_75, {queries_for_columns_params_keys} }})
                                    YIELD value as usage
                                WITH n, snippet, usage.usage as usage, complex_snippet
                                RETURN collect(distinct {{sql_code: snippet.sql_snippet, snippet_id: snippet.sql_snippet_id, usage: usage}}) as sql,
                                       reduce(res=FALSE, flag IN collect(complex_snippet) |res OR flag) as complex_attribute
                                UNION
                                WITH n
                                OPTIONAL MATCH (n)-[:attr_of]->(dummy:column|error_node|sql|alias)
                                WITH n, count(dummy) as snippet_count
                                WHERE snippet_count = 0
                                RETURN [] as sql, false as complex_attribute
                             }}
                             RETURN apoc.map.setPairs(properties(n),[["owner_id", term.owner_id],["term_name", term.name], ["sql", sql], ["complex_attribute", complex_attribute]]) as item',
                        n:analysis,
                            'MATCH(n)-[:analysis_of]->(sql:sql)
                            WITH n, collect(distinct {{sql_code: sql.sql_full_query}}) as sql
                            RETURN apoc.map.setKey(properties(n), "sql", sql) as item',
                        n:metric,
                            'OPTIONAL MATCH(n)-[metric_sql:metric_sql]->(attribute:attribute)
                            WITH n, collect(distinct {{sql_code: metric_sql.sql}}) as sql
                            RETURN apoc.map.setKey(properties(n), "sql", sql) as item',
                        n:field,
                            'MATCH(n)<-[:{"|".join(fields_relationships)}]-(parent) 
                            RETURN apoc.map.setPairs(properties(n),[["table_name", parent.name],["table_type", parent.type], ["parent_id", parent.id]]) as item',
                        n:column,
                            'MATCH(n)<-[:schema]-(parent) 
                            RETURN apoc.map.setPairs(properties(n),[["table_name", parent.name],["table_type", parent.type], ["parent_id", parent.id]]) as item'
                        ],
                        
                        'with n RETURN n{{ .*}} as item ',
                        {{n:n, account_id:$account_id, sql_type: $sql_type, usage_percentile_25: $usage_percentile_25,
                        usage_percentile_75: $usage_percentile_75, {queries_for_columns_params_keys} }}
                        )
                    YIELD value as response
                    WITH collect(response.item) as all_items
                    RETURN apoc.map.groupBy(all_items,'id') as ids_to_props
                    """
        params = {
            "account_id": account_id,
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
            # Add additional info
            for id, item in result[0]["ids_to_props"].items():
                if item["type"] == Labels.BT:
                    term_tables = get_term_tables(
                        account_id, user_participants, item["id"], 0, 100
                    )
                    item["tables"] = term_tables
                elif item["type"] == Labels.ATTR:
                    attr_columns = get_attr_columns(
                        account_id, user_participants, item["id"], 0, 100
                    )
                    item["columns"] = attr_columns

            results = results | result[0]["ids_to_props"]

    return results


def get_semantic_candidates_information(
    entity: str,
    k: int = 5,
    threshold: float = 0,
    list_of_semantic: list = Labels.LIST_OF_SEMANTIC,
    exclude_ids: list[str] | None = None,
):
    labels = list_of_semantic
    results = []
    if exclude_ids:
        exclude_set = set(exclude_ids)
        allowed_ids = [node_id for node_id in allowed_ids if node_id not in exclude_set]

    # TODO: use Retriever class to search semantic vector index
    # nodes_results = search_vector_index(
    #     "nodes_vector_store",
    #     allowed_ids,
    #     entity,
    #     k=k,
    #     label_filter=labels,
    # )
    # results.extend([item["metadata"] for item in nodes_results])

    ids_and_labels = [{"label": x["label"], "id": x["id"]} for x in results]
    candidates_properties = expand_info(ids_and_labels)
    for c in results:
        update_candidate_properties(account_id, c, candidates_properties)
    results = [
        c
        for c in results
        if (c["label"] == Labels.ATTR and "sql" in c) or c["label"] != Labels.ATTR
    ]
    # score is similarity, higher is better
    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results


def _fetch_unique_attr_candidates(
    query: str,
    semantic_labels: list[str],
    max_results: int,
) -> list[dict]:
    """
    Repeatedly pull batches of semantic candidates until we have `max_results`
    distinct items (by ID) while also de-duplicating attributes that share the
    same (name, single snippet) pair. This avoids returning multiple copies of
    the same metric/attribute while still keeping the search responsive by
    capping the number of backend calls.

    Args:
        account_id: Account ID
        user_participants: User participants list
        query: Search query string
        semantic_labels: List of semantic labels to search for (e.g., [Labels.ATTR, Labels.METRIC])
        max_results: Maximum number of results to return

    Returns:
        List of unique candidate dictionaries
    """
    results: list[dict] = []
    seen_attr_keys: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    attempts = 0

    while len(results) < max_results and attempts < ATTR_CANDIDATE_MAX_ATTEMPTS:
        batch_size = max(ATTR_CANDIDATE_BATCH_SIZE, max_results - len(results))
        chunk = get_semantic_candidates_information(
            query,
            k=batch_size,
            list_of_semantic=semantic_labels,
            exclude_ids=list(seen_ids) if seen_ids else None,
        )
        attempts += 1
        if not chunk:
            break

        added_this_round = False
        for candidate in chunk:
            candidate_id = candidate.get("id")
            if candidate_id in seen_ids:
                continue

            seen_ids.add(candidate_id)
            results.append(candidate)
            added_this_round = True

            if len(results) >= max_results:
                break

        if not added_this_round:
            break

    return results[:max_results]


def extract_candidates(
    entities: list[str],
    query_no_values: str,
) -> list[dict]:
    """
    Extract candidates for calculation queries.

    This function:
    1. Gets top candidates for the full question
    2. Ensures at least one candidate for each required entity
    3. Combines and deduplicates results
    4. Processes attribute candidates to keep best snippets per zone
    5. Adds relevant tables information to each candidate

    Args:
        user_participants: User participants list
        entities: List of entity names to ensure coverage for
        query_no_values: Query without values
    Returns:
        List of dictionaries with structure:
        {
            "candidate": candidate dictionary with relevant_tables information,
            "entity": entity name this candidate is related to (or None if not entity-specific)
        }
    """
    # 1. Get top candidates for the full question
    calculation_labels = [Labels.CUSTOM_ANALYSIS]

    # Question candidates from query_no_values search
    question_candidates = _fetch_unique_attr_candidates(
        user_participants,
        query_no_values,
        semantic_labels=calculation_labels,
        max_results=MAX_CALCULATION_CANDIDATES,
    )
    logger.info(
        f"Found {len(question_candidates)} candidates from question search"
    )

    # 2. Ensure at least one candidate for each required entity
    # Use question candidates for entity coverage check
    all_question_candidates = question_candidates
    question_candidate_names: dict[str, list[str]] = {}
    question_candidate_ids = {
        c.get("id") for c in all_question_candidates if c.get("id")
    }
    for c in all_question_candidates:
        name = c.get("name")
        if name:
            name_lower = name.lower()
            candidate_id = c.get("id")
            if candidate_id:
                if name_lower not in question_candidate_names:
                    question_candidate_names[name_lower] = []
                question_candidate_names[name_lower].append(candidate_id)

    # Determine how many candidates to fetch per entity
    # Distribute ENTITY_CANDIDATE_BUDGET evenly across entities (at least 1 per entity)
    k = max(1, ENTITY_CANDIDATE_BUDGET // len(entities)) if entities else 0

    entities_to_search = []
    for entity in entities:
        entity_lower = entity.lower()
        # Check if entity is already covered in question candidates (exact match)
        if entity_lower in question_candidate_names:
            entity_ids = question_candidate_names[entity_lower]
            # Check if any of the IDs for this name are already in candidates
            if any(eid in question_candidate_ids for eid in entity_ids):
                logger.info(
                    f"Entity '{entity}' already covered in question candidates, skipping search"
                )
                continue

        # Check if any candidate name contains this entity (fuzzy match)
        entity_covered = any(
            entity_lower in name or name in entity_lower
            for name in question_candidate_names.keys()
        )
        if entity_covered:
            logger.info(
                f"Entity '{entity}' appears to be covered in question candidates, skipping search"
            )
            continue

        entities_to_search.append(entity)

    # Only search for entities not already covered
    # Track which entity each candidate came from to ensure each entity gets at least one slot
    entity_candidates_with_source = []  # List of (candidate, entity_name) tuples
    for entity in entities_to_search:
        entity_specific = _fetch_unique_attr_candidates(
            user_participants,
            entity,
            semantic_labels=[Labels.CUSTOM_ANALYSIS],
            max_results=max(k, 1),
        )
        for candidate in entity_specific:
            if candidate.get("id") in question_candidate_ids:
                question_candidate_ids.remove(candidate.get("id"))
                # Remove from all_question_candidates by ID
                all_question_candidates = [
                    c
                    for c in all_question_candidates
                    if c.get("id") != candidate.get("id")
                ]

            entity_candidates_with_source.append((candidate, entity))

    # 3. Combine candidates with priority: entity candidates ensuring 1 per entity + question candidates
    #
    # Strategy:
    # - Reserve slots: 1 per entity + 1 for question candidates
    # - Ensure each entity gets at least one candidate
    # - Ensure question candidates get at least one candidate
    # - Fill remaining slots with remaining entity and question candidates
    # - Don't sort by score (different extraction logic, scores not comparable)

    # Calculate required slots for entities and question candidates
    num_entities_to_cover = len(entities_to_search)
    slots_needed_for_entities = num_entities_to_cover
    slots_needed_for_question = 1 if question_candidates else 0
    reserved_slots = slots_needed_for_entities + slots_needed_for_question

    # Track candidates with their entity relationships: list of (candidate, entity_name or None)
    candidates_with_entities = []
    existing_ids = set()

    logger.info(
        f"Reserved {reserved_slots} slots for entities ({slots_needed_for_entities}) and question ({slots_needed_for_question}), "
        f"using {len(candidates_with_entities)} initial candidates"
    )

    # Track which entities have been covered
    question_candidate_added = False

    # First pass: ensure each entity gets at least one candidate
    for cand, entity in entity_candidates_with_source:
        cand_id = cand.get("id")
        if cand_id:
            candidates_with_entities.append((cand, entity))
            if cand_id not in existing_ids:
                existing_ids.add(cand_id)

    # Second pass: ensure question candidates get at least one slot
    if question_candidates and not question_candidate_added:
        for cand in question_candidates:
            if len(candidates_with_entities) >= MAX_CALCULATION_CANDIDATES:
                break
            cand_id = cand.get("id")
            if cand_id and cand_id not in existing_ids:
                candidates_with_entities.append((cand, None))
                existing_ids.add(cand_id)
                question_candidate_added = True
                break

    # Third pass: fill remaining slots with remaining entity and question candidates
    remaining_entity_candidates = [
        (cand, entity)
        for cand, entity in entity_candidates_with_source
        if cand.get("id") not in existing_ids
    ]
    remaining_question_candidates = [
        cand for cand in question_candidates if cand.get("id") not in existing_ids
    ]

    # Alternate between question and entity candidates to fill remaining slots
    question_idx = 0
    entity_idx = 0

    while len(candidates_with_entities) < MAX_CALCULATION_CANDIDATES:
        added_any = False

        # Try to add entity candidate
        if entity_idx < len(remaining_entity_candidates):
            cand, entity = remaining_entity_candidates[entity_idx]
            cand_id = cand.get("id")
            if cand_id and cand_id not in existing_ids:
                candidates_with_entities.append((cand, entity))
                existing_ids.add(cand_id)
                entity_idx += 1
                added_any = True

        # Try to add question candidate
        if question_idx < len(remaining_question_candidates):
            cand = remaining_question_candidates[question_idx]
            cand_id = cand.get("id")
            if cand_id and cand_id not in existing_ids:
                candidates_with_entities.append((cand, None))
                existing_ids.add(cand_id)
                question_idx += 1
                added_any = True

        # If we couldn't add any, break
        if not added_any:
            break

    # 4. Collect all tables' info from snippets
    # For each attribute candidate, retain only the highest-usage SQL snippet.
    for candidate, entity in candidates_with_entities:
        if (
            candidate.get("label") == Labels.CUSTOM_ANALYSIS
            and len(candidate.get("sql", [])) + len(candidate.get("documents", [])) > 1
        ):
            sql_snippets = candidate.get("sql", [])
            if sql_snippets:
                best_sql_snippet = max(sql_snippets, key=lambda s: s.get("usage", 0))
                candidate["sql"] = [best_sql_snippet]

        candidates_relevant_tables = (
            get_item_tables(account_id, user_participants, candidate)
            if candidate.get("label") != Labels.BT
            else candidate.get("tables", {}).get("tables", [])
        )

        candidate["relevant_tables"] = candidates_relevant_tables

    # Return list of dicts with candidate and entity structure
    return [
        {"candidate": candidate, "entity": entity}
        for candidate, entity in candidates_with_entities
    ]
