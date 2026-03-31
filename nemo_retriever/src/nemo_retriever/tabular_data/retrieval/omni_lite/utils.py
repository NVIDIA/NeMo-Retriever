from enum import StrEnum


class Labels(StrEnum):
    """Semantic labels used by omni-lite candidate retrieval."""

    CUSTOM_ANALYSIS = "custom_analysis"
    ATTR = "attr"
    METRIC = "metric"
    ANALYSIS = "analysis"
    BT = "bt"


def extract_candidates(
    user_participants: list,
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
