from shared.graph.dal.usage_dal import (
    get_columns_usage_percentiles,
    get_node_usage_percentiles,
    get_tables_usage_percentiles,
    get_usage_percentiles,
    usage_entities,
)
from shared.graph.model.reserved_words import (
    SQLType,
    Labels,
    UsageDesc,
    label_to_type,
    type_to_labels,
    bi_roots,
)
import logging
from nltk.corpus import wordnet  # nltk.download("wordnet")

from shared.graph.dal.metrics_dal import get_metrics_related_zones_cypher
from shared.graph.dal.attribute_dal import get_term_attr_zones, get_analysis_zones

# Init the wordnet module during startup
_ = wordnet.synsets("temp")


business_domain = "business company sales customer market"

logger = logging.getLogger("search.py")


def find_syn_index(x, search_synonyms):
    for idx, search_syn in enumerate(search_synonyms):
        if search_syn.lower() in x.lower():
            return idx


class SearchObjectOptions:
    TERM = "term"
    ANALYSIS = "analysis"
    METRIC = "metric"
    ATTRIBUTE = "attribute"
    DOCUMENT = "document"
    SCHEMA = "schema"
    BASE_TABLE = "base table"
    VIEW = "view"
    COLUMN = "column"
    ONLY_COLUMN = "only column"
    ONLY_TABLE = "only table"
    FIELD = "field"
    TAG = "tag"
    DB = "db"
    WORKBOOK = "workbook"
    DASHBOARD = "dashboard"
    PROJECT = "project"
    SHEET = "sheet"
    SISENSE_DATAMODEL = "sisense_datamodel"
    QUERY = "query"
    SITE = "site"


def _documents_allowed(filters: dict) -> bool:
    """Check if documents are part of the requested object filters (or no filter)."""
    if "objects" not in filters or not filters["objects"]:
        return True
    return SearchObjectOptions.DOCUMENT in filters["objects"]


dashboards_labels = type_to_labels("dashboard") + [Labels.POWERBI_REPORT]
visuals_labels = type_to_labels("sheet")
search_object_to_label = {
    SearchObjectOptions.TERM: "term",
    SearchObjectOptions.ANALYSIS: "analysis",
    SearchObjectOptions.METRIC: "metric",
    SearchObjectOptions.ATTRIBUTE: "attribute",
    SearchObjectOptions.DOCUMENT: "document",
    SearchObjectOptions.SCHEMA: "schema",
    SearchObjectOptions.BASE_TABLE: f"table|sisense_table|{'|'.join(Labels.POWERBI_TABLES_LABELS)}",
    SearchObjectOptions.VIEW: "table",
    SearchObjectOptions.COLUMN: "column|field",
    SearchObjectOptions.ONLY_COLUMN: "column",
    SearchObjectOptions.ONLY_TABLE: "table",
    SearchObjectOptions.TAG: "tag",
    SearchObjectOptions.DB: "db",
    SearchObjectOptions.PROJECT: "project",
    SearchObjectOptions.WORKBOOK: "workbook",
    SearchObjectOptions.DASHBOARD: "|".join(dashboards_labels),
    SearchObjectOptions.SHEET: "|".join(visuals_labels),
    SearchObjectOptions.SISENSE_DATAMODEL: f"sisense_datamodel|qs_dataset|published_datasource|embedded_datasource|{Labels.POWERBI_DATAFLOW}|{Labels.POWERBI_SEMANTIC_MODEL}",
    SearchObjectOptions.QUERY: "sql",
    SearchObjectOptions.SITE: "|".join(bi_roots.values()) + f"|{Labels.PROJECT}",
}

powerbi_tables_labels = " or ".join(
    [f"n:{label}" for label in Labels.POWERBI_TABLES_LABELS]
)
dashboards_nodes_labels = " or ".join([f"n:{label}" for label in dashboards_labels])
visuals_nodes_labels = " or ".join([f"n:{label}" for label in visuals_labels])
roots_labels = " or ".join([f"n:{label}" for label in bi_roots.values()])
object_type_to_string = {
    SearchObjectOptions.TERM: "n:term",
    SearchObjectOptions.ANALYSIS: "(n:analysis and not coalesce(n.recommended, false))",
    SearchObjectOptions.METRIC: "(n:metric and not coalesce(n.recommended, false))",
    SearchObjectOptions.ATTRIBUTE: "n:attribute",
    SearchObjectOptions.DOCUMENT: "n:document",
    SearchObjectOptions.SCHEMA: "n:schema",
    SearchObjectOptions.BASE_TABLE: f"""(n:table or n:sisense_table or {powerbi_tables_labels}) and (NOT ( n:table and (n.table_type="view" or n.table_type="materialized view")))""",
    SearchObjectOptions.VIEW: """(n:table and (n.table_type="view" or n.table_type="materialized view"))""",
    SearchObjectOptions.COLUMN: "n:column or n:field",
    SearchObjectOptions.ONLY_COLUMN: "n:column",
    SearchObjectOptions.ONLY_TABLE: "n:table",
    SearchObjectOptions.TAG: "n:tag",
    SearchObjectOptions.DB: "n:db",
    SearchObjectOptions.PROJECT: "n:project",
    SearchObjectOptions.WORKBOOK: "n:workbook",
    SearchObjectOptions.DASHBOARD: dashboards_nodes_labels,
    SearchObjectOptions.SHEET: visuals_nodes_labels,
    SearchObjectOptions.SISENSE_DATAMODEL: f"n:sisense_datamodel or n:qs_dataset or n:published_datasource or n:embedded_datasource OR n:{Labels.POWERBI_DATAFLOW} OR n:{Labels.POWERBI_SEMANTIC_MODEL}",
    SearchObjectOptions.QUERY: "(n:sql and not n.is_sub_select and n.sql_type=$sql_type)",
    SearchObjectOptions.SITE: roots_labels,
}


def is_in_object_type_to_string(x):
    return x in object_type_to_string


def filter_last_updated(all_filters):
    has_last_updated_filter = "last_updated" in all_filters
    if has_last_updated_filter:
        if (
            all_filters["last_updated"] == 0
        ):  # When user selects Never all semantic elements return
            semantic_entities = [
                label_to_type(Labels.ATTR),
                label_to_type(Labels.ANALYSIS),
                label_to_type(Labels.METRIC),
                label_to_type(Labels.BT),
            ]
            arr_to_string = '","'.join(semantic_entities)
            return (
                'AND (n.type IN ["' + arr_to_string + '"]) AND n.last_modified IS NULL'
            )
        else:
            base = "AND (n.last_modified IS NOT NULL AND n.last_modified > dateTime() - duration"
            time_param = "({hours:" + str(all_filters["last_updated"]) + "}))"
            return base + time_param
    return ""


def filter_by_label(all_filters):
    search_options_to_query = set(search_object_to_label)
    if "objects" in all_filters:
        search_options_to_query = search_options_to_query & set(
            filter(is_in_object_type_to_string, all_filters["objects"])
        )
    if "usage" in all_filters:
        search_options_to_query = search_options_to_query & set(
            [
                SearchObjectOptions.TERM,
                SearchObjectOptions.ANALYSIS,
                SearchObjectOptions.METRIC,
                SearchObjectOptions.ATTRIBUTE,
                SearchObjectOptions.COLUMN,
                SearchObjectOptions.QUERY,
                SearchObjectOptions.BASE_TABLE,
                SearchObjectOptions.VIEW,
            ]
        )
    if "data" in all_filters or "status" in all_filters:
        search_options_to_query = search_options_to_query & set(
            [
                SearchObjectOptions.BASE_TABLE,
                SearchObjectOptions.COLUMN,
                SearchObjectOptions.VIEW,
            ]
        )
    if "data_types" in all_filters or "nullable" in all_filters:
        search_options_to_query = search_options_to_query & set(
            [SearchObjectOptions.COLUMN]
        )
    if "columns_count" in all_filters or "duplicated" in all_filters:
        search_options_to_query = search_options_to_query & set(
            [SearchObjectOptions.BASE_TABLE, SearchObjectOptions.VIEW]
        )
    if "visuals_count" in all_filters:
        search_options_to_query = search_options_to_query & set(
            [SearchObjectOptions.DASHBOARD]
        )
    if "last_updated" in all_filters or "certification" in all_filters:
        search_options_to_query = search_options_to_query & set(
            [
                SearchObjectOptions.TERM,
                SearchObjectOptions.ANALYSIS,
                SearchObjectOptions.METRIC,
                SearchObjectOptions.ATTRIBUTE,
            ]
        )
    if "attributes_count" in all_filters:
        search_options_to_query = search_options_to_query & set(
            [SearchObjectOptions.TERM]
        )
    if (
        "tags" in all_filters
        or "owner_ids" in all_filters
        or "has_owner" in all_filters
        or "documentation" in all_filters
    ):
        search_options_to_query.discard(SearchObjectOptions.QUERY)
    if "zones" in all_filters:
        search_options_to_query = search_options_to_query & set(
            [
                SearchObjectOptions.TERM,
                SearchObjectOptions.METRIC,
                SearchObjectOptions.ANALYSIS,
                SearchObjectOptions.ATTRIBUTE,
                SearchObjectOptions.DB,
                SearchObjectOptions.SCHEMA,
                SearchObjectOptions.BASE_TABLE,
                SearchObjectOptions.COLUMN,
            ]
        )
    result = "|".join(search_object_to_label[obj] for obj in search_options_to_query)
    return ":" + result if result else result


def filter_by_object(all_filters):
    has_objects_filter = "objects" in all_filters
    if has_objects_filter:
        relevant_obj_filters = filter(
            is_in_object_type_to_string, all_filters["objects"]
        )
        result = " or ".join(object_type_to_string[obj] for obj in relevant_obj_filters)
        result = "and (" + result + ")"
        return result
    return f"""and
                 (
                    n:term or
                    (n:analysis and not coalesce(n.recommended, false)) or
                    (n:metric and not coalesce(n.recommended, false)) or
                    n:attribute or
                    n:document or
                    n:db or
                    n:schema or
                    n:table or
                    n:column or
                    n:tag or
                    n:field or
                    // tableau
                    n:site or
                    n:project or
                    n:workbook or
                    n:dashboard or
                    n:sheet or
                    n:embedded_datasource or
                    n:published_datasource or
                    // quicksight
                    n:quicksight or
                    n:qs_dashboard or n:qs_analysis or
                    n:qs_visualization or
                    n:qs_dataset or
                    // sisense
                    n:sisense or
                    n:sisense_dashboard or
                    n:sisense_widget or
                    n:sisense_table or
                    n:sisense_datamodel or
                    // powerBI
                    n:{Labels.POWERBI_DATAFLOW_TABLE} or
                    n:{Labels.POWERBI_COPIED_DATAFLOW_TABLE} or
                    n:{Labels.POWERBI_SEMANTIC_TABLE} or
                    n:{Labels.POWERBI_VISUALIZATION} or
                    n:{Labels.POWERBI_DASHBOARD} or
                    n:{Labels.POWERBI_DATAFLOW} or
                    n:{Labels.POWERBI_SEMANTIC_MODEL} or
                    n:{Labels.POWERBI_REPORT} or
                    // looker
                    n:{Labels.LOOKER} or
                    n:{Labels.LOOKER_PROJECT} or
                    n:{Labels.LOOKER_BOARD} or
                    n:{Labels.LOOKER_DASHBOARD} or
                    n:{Labels.LOOKER_FOLDER} or
                    n:{Labels.LOOKER_VIEW} or
                    n:{Labels.LOOKER_EXPLORE} or
                    n:{Labels.LOOKER_LOOK} or
                    n:{Labels.LOOKER_VISUAL} or
                    (n:sql and not n.is_sub_select and n.sql_type=$sql_type)
                )"""


def filter_by_owner(all_filters):
    has_owners_ids_filter = "owner_ids" in all_filters
    if has_owners_ids_filter:
        if not all_filters["owner_ids"]:
            return "AND ((parent IS NULL AND n.owner_id IS NULL) OR (parent IS NOT NULL AND parent.owner_id IS NULL))"

        owners_condition = "AND ((n.owner_id IS NOT NULL AND n.owner_id IN $owners) OR (parent IS NOT NULL AND parent.owner_id in $owners))"
        return owners_condition
    return ""


def filter_by_has_owner(all_filters):
    has_owner_filter = "has_owner" in all_filters
    if has_owner_filter:
        has_owner = all_filters["has_owner"] is True
        if has_owner:
            return "AND (n.owner_id IS NOT NULL)"
        # TODO: Move doesn't have owner from filter_by_owner
    return ""


def filter_by_tag(all_filters):
    has_tags_filter = "tags" in all_filters
    if has_tags_filter:
        tags_condition = "AND (any(tag IN assigned_tag_ids WHERE tag IN $tags)"
        if "(blanks)" in all_filters["tags"]:
            tags_condition += " OR (NOT EXISTS {(n)<-[:tag_of]-(:tag)} AND NOT EXISTS {(n)<-[:applies_to]-(:rule)})"
        tags_condition += ")"
        return tags_condition
    return ""


def filter_by_data_type(all_filters):
    has_data_types_filter = "data_types" in all_filters
    if has_data_types_filter:
        data_types_condition = (
            """AND (n.type = "column" AND n.data_type IN $data_types)"""
        )
        return data_types_condition
    return ""


def filter_by_documentation(all_filters):
    has_documentation_filter = "documentation" in all_filters
    if has_documentation_filter:
        documented = all_filters["documentation"] == "documented"
        if documented:
            return """AND (labels(n)[0] <> "sql" and n.description IS NOT NULL and n.description <> "")"""
        return """AND (labels(n)[0] <> "sql" and n.description IS NULL or n.description = "")"""
    return ""


def filter_by_semantic_status(all_filters):
    has_semantic_status_filter = "semantic_status" in all_filters
    if has_semantic_status_filter:
        semantic_status = all_filters["semantic_status"]
        return f"""AND coalesce(n.invalid, false) = {semantic_status}"""
    return ""


def filter_by_data_status(all_filters):
    has_status_filter = "status" in all_filters
    if has_status_filter:
        deleted = all_filters["status"] == "deleted"
        return f"""AND coalesce(n.deleted, false) = {deleted}"""
    return ""


def filter_by_certification(all_filters):
    has_certification_filter = "certification" in all_filters
    if has_certification_filter:
        certified_metric_props = (
            "n.certified_name, n.certified_description, n.certified_formula"
        )
        certified_analysis_props = (
            "n.certified_name, n.certified_description, n.certified_query"
        )
        certified_term_props = "n.certified_name, n.certified_description"
        has_uncertified_attributes = (
            "exists((n)-[:term_of]->(:attribute{certified:FALSE}))"
        )
        has_certified_attributes = (
            "exists((n)-[:term_of]->(:attribute{certified:TRUE}))"
        )

        condition_arr = []
        data_certification_condition = "AND ("
        if "pending" in all_filters["certification"]:
            pending_data_certification_condition = f"""(
                (n.type = "{label_to_type(Labels.ATTR)}" AND n.certified = FALSE) OR 
                (n.type = "{label_to_type(Labels.METRIC)}" AND all(x IN [{certified_metric_props}] WHERE x IS NULL or x=FALSE)) OR 
                (n.type = "{label_to_type(Labels.ANALYSIS)}" AND all(x IN [{certified_analysis_props}] WHERE x IS NULL or x=FALSE)) OR 
                (n.type = "{label_to_type(Labels.BT)}" AND (all(x IN [{certified_term_props}] WHERE x IS NULL or x=FALSE) AND NOT {has_certified_attributes}))
            )"""
            condition_arr.append(pending_data_certification_condition)

        if "partial" in all_filters["certification"]:
            partial_data_certification_condition = f"""(
                (n.type = "{label_to_type(Labels.METRIC)}" AND 
                    (any(x IN [{certified_metric_props}] WHERE x=True) AND NOT all(x IN [{certified_metric_props}] WHERE x=TRUE))) OR 
                (n.type = "{label_to_type(Labels.ANALYSIS)}" AND 
                    (any(x IN [{certified_analysis_props}] WHERE x=True) AND NOT all(x IN [{certified_analysis_props}] WHERE x=TRUE))) OR 
                (n.type = "{label_to_type(Labels.BT)}" AND 
                    (any(x IN [{certified_term_props}] WHERE x=True) AND NOT all(x IN [{certified_term_props}] WHERE x=TRUE)) OR ({has_certified_attributes} AND {has_uncertified_attributes}))
            )"""
            condition_arr.append(partial_data_certification_condition)

        if "certified" in all_filters["certification"]:
            certified_data_certification_condition = f"""(
                (n.type = "{label_to_type(Labels.ATTR)}" AND n.certified = TRUE) OR 
                (n.type = "{label_to_type(Labels.METRIC)}" AND all(x IN [{certified_metric_props}] WHERE x=TRUE)) OR 
                (n.type = "{label_to_type(Labels.ANALYSIS)}" AND all(x IN [{certified_analysis_props}] WHERE x=TRUE)) OR 
                (n.type = "{label_to_type(Labels.BT)}" AND (all(x IN [{certified_term_props}] WHERE x=TRUE) AND NOT {has_uncertified_attributes}))
            )"""
            condition_arr.append(certified_data_certification_condition)

        data_certification_condition += " OR ".join(condition_arr) + ")"
        return data_certification_condition
    return ""


def filter_by_duplications(all_filters):
    has_duplications_filter = "duplicated" in all_filters
    if has_duplications_filter:
        duplicated = all_filters["duplicated"] is True
        if duplicated:
            return """AND (n.type = "base table" AND (n.deleted<>True AND dup_tbl.deleted<>True))"""
    return ""


def filter_by_nullable(all_filters):
    has_nullable_filter = "nullable" in all_filters
    if has_nullable_filter:
        nullable = all_filters["nullable"] is True
        if nullable:
            return """AND (n.type = "column" AND (n.is_nullable = "YES"))"""
    return ""


def get_synonyms(search_term, all_filters):
    try:
        disable_synonyms = (
            "synonyms" in all_filters and all_filters["synonyms"] is False
        )
        if disable_synonyms or not search_term:
            return f"*{search_term}*", [search_term]

        search_synonyms = [search_term]
        synsets = wordnet.synsets(f"{search_term}")

        for syn in synsets:
            for lemma in syn.lemmas():
                if lemma.name() not in search_synonyms:
                    search_synonyms.append(
                        lemma.name()
                        .replace("_", " ")
                        .replace("-", " ")
                        .replace("'", "")
                        .replace("/", "//")
                    )
        search_synonyms_list = search_synonyms.copy()
        search_synonyms[0] = (
            " ".join(
                [f"*{item}*" for item in search_synonyms[0].split()]
            )  # Complete only the required word/expression (held in search_synonyms[0]). Completing synonyms could be confusing, for example, search term:     "card", a possible synonym is "wit".
            # Searching for "wit*" results in all words starting with "wit" (including with, withdraw, etc.), which are unwanted search results.
            if len(search_synonyms[0].split()) > 1
            else f"*{search_synonyms[0]}*"
        )
        search_synonyms = [
            " AND ".join([f"{item}" for item in item.split()])
            for item in search_synonyms
        ]  # put and between serach words in single expression (business AND id -> business entity id)

        search_expr = ") OR (".join(
            search_synonyms
        )  # put or between words for synonyms search (customer OR client)
        return search_expr, search_synonyms_list
    except Exception as e:
        logger.error(f"synonyms failed: {str(e)}", exc_info=e)
        logger.info("synonyms failed for " + search_term)
        return f"*{search_term}*", [search_term]


def filter_by_attribute_count(all_filters):
    has_attributes_count_filter = "attributes_count" in all_filters
    if has_attributes_count_filter:
        reference_number = all_filters["attributes_count"]["reference_number"]
        map_sign_to_string = {
            "less": f"<{reference_number}",
            "greater": f">{reference_number}",
            "greater_or_equal": f">={reference_number}",
            "less_or_equal": f"<={reference_number}",
            "equal": f"={reference_number}",
        }

        if "range_number" in all_filters["attributes_count"]:
            range_number = all_filters["attributes_count"]["range_number"]
            map_sign_to_string["between"] = (
                f">{reference_number} AND count_attr<{range_number}"
            )

        attributes_count_condition = f"""AND (n.type = "term" AND count_attr {map_sign_to_string[all_filters["attributes_count"]["relation"]]})"""
        return attributes_count_condition
    return ""


def filter_by_column_count(all_filters):
    has_columns_count_filter = "columns_count" in all_filters
    if has_columns_count_filter:
        reference_number = all_filters["columns_count"]["reference_number"]
        map_sign_to_string = {
            "less": f"<{reference_number}",
            "greater": f">{reference_number}",
            "greater_or_equal": f">={reference_number}",
            "less_or_equal": f"<={reference_number}",
            "equal": f"={reference_number}",
        }

        if "range_number" in all_filters["columns_count"]:
            range_number = all_filters["columns_count"]["range_number"]
            map_sign_to_string["between"] = (
                f">{reference_number} AND count_col<{range_number}"
            )

        columns_count_condition = f"""AND (labels(n)[0] = "{Labels.TABLE}" AND count_col {map_sign_to_string[all_filters["columns_count"]["relation"]]})"""
        return columns_count_condition
    return ""


def filter_by_visual_count(all_filters):
    has_visuals_count_filter = "visuals_count" in all_filters
    if has_visuals_count_filter:
        reference_number = all_filters["visuals_count"]["reference_number"]
        map_sign_to_string = {
            "less": f"<{reference_number}",
            "greater": f">{reference_number}",
            "greater_or_equal": f">={reference_number}",
            "less_or_equal": f"<={reference_number}",
            "equal": f"={reference_number}",
        }

        if "range_number" in all_filters["visuals_count"]:
            range_number = all_filters["visuals_count"]["range_number"]
            map_sign_to_string["between"] = (
                f">{reference_number} AND count_vis<{range_number}"
            )

        visuals_count_condition = f"""AND (n.type = "{Labels.DASHBOARD}" AND count_vis {map_sign_to_string[all_filters["visuals_count"]["relation"]]})"""
        return visuals_count_condition
    return ""


def get_query_params(account_id, filters, search_term):
    params = {
        "account_id": account_id,
        "search_term": search_term,
        "sql_type": SQLType.QUERY,
    }

    if "owner_ids" in filters:
        params["owners"] = filters["owner_ids"]

    if "tags" in filters:
        params["tags"] = filters["tags"]

    if "zones" in filters:
        params["zones"] = filters["zones"]

    if "data_types" in filters:
        params["data_types"] = filters["data_types"]

    if "data" in filters:
        params["data_entities_ids"] = list(map(lambda x: x["id"], filters["data"]))

    if "usage" in filters:
        term_percentile_25, term_percentile_75 = get_node_usage_percentiles(
            account_id, Labels.BT
        )
        metric_percentile_25, metric_percentile_75 = get_node_usage_percentiles(
            account_id, Labels.METRIC
        )
        attr_percentile_25, attr_percentile_75 = get_node_usage_percentiles(
            account_id, Labels.ATTR
        )
        table_percentile_25, table_percentile_75 = get_tables_usage_percentiles(
            account_id
        )
        column_percentile_25, column_percentile_75 = get_columns_usage_percentiles(
            account_id
        )
        sql_percentile_25, sql_percentile_75 = get_usage_percentiles(account_id)
        analysis_percentile_25, analysis_percentile_75 = get_usage_percentiles(
            account_id
        )

        params = {
            **params,
            "term_percentile_25": term_percentile_25,
            "term_percentile_75": term_percentile_75,
            "metric_percentile_25": metric_percentile_25,
            "metric_percentile_75": metric_percentile_75,
            "attribute_percentile_25": attr_percentile_25,
            "attribute_percentile_75": attr_percentile_75,
            "table_percentile_25": table_percentile_25,
            "table_percentile_75": table_percentile_75,
            "column_percentile_25": column_percentile_25,
            "column_percentile_75": column_percentile_75,
            "sql_percentile_25": sql_percentile_25,
            "sql_percentile_75": sql_percentile_75,
            "analysis_percentile_25": analysis_percentile_25,
            "analysis_percentile_75": analysis_percentile_75,
        }
    return params


def get_text_match_condition(text_match_option):
    if text_match_option == "starts_with":
        return "AND (toLower(n.name) STARTS WITH toLower($search_term))"
    if text_match_option == "ends_with":
        return "AND (toLower(n.name) ENDS WITH toLower($search_term))"
    if text_match_option == "not_contain":
        return "AND (NOT toLower(n.name) CONTAINS toLower($search_term))"
    return ""


def filter_by_data(all_filters):
    has_data_filter = "data" in all_filters
    if has_data_filter:
        data_tree_condition = f"""AND (
            (labels(n)[0] IN ["{Labels.TABLE}", "{Labels.COLUMN}", "{Labels.SCHEMA}"]) AND 
            ((parent_schema.id IN $data_entities_ids) OR (parent_db.id IN $data_entities_ids))
            )
        """
        return data_tree_condition
    return ""


usage_group_strings = {
    UsageDesc.HIGH: lambda entity_type: f"""(labels(n)[0] = "{entity_type}" AND n.usage > ${entity_type}_percentile_75)""",
    UsageDesc.MEDIUM: lambda entity_type: f"""(labels(n)[0] = "{entity_type}" AND n.usage < ${entity_type}_percentile_75 and n.usage > ${entity_type}_percentile_25)""",
    UsageDesc.LOW: lambda entity_type: f"""(labels(n)[0] = "{entity_type}" AND n.usage < ${entity_type}_percentile_25 and n.usage > 0)""",
    UsageDesc.UNUSED: lambda entity_type: f"""(labels(n)[0] = "{entity_type}" AND n.usage = 0)""",
}


def filter_by_usage(all_filters):
    has_usage_filter = "usage" in all_filters
    if not has_usage_filter:
        return ""
    active_usage_filters = []
    for usage_filter in all_filters["usage"]:
        for entity in usage_entities:
            active_usage_filters.append(usage_group_strings[usage_filter](entity))

    usage_str = " OR ".join(active_usage_filters)
    usage_str = "AND(" + usage_str + ")"
    return usage_str


def filter_by_zones(all_filters):
    has_zones_filter = "zones" in all_filters
    if has_zones_filter:
        zones_condition = (
            "AND (any(zone_id IN assigned_zones_ids WHERE zone_id IN $zones)"
        )
        if "(blanks)" in all_filters["zones"]:
            zones_condition += "OR size(coalesce(assigned_zones_ids, [])) = 0"
        return zones_condition + ")"
    return ""


def search_in_tags(all_filters: str) -> str:
    has_search_in_tags_filter = "search_tags" in all_filters
    if has_search_in_tags_filter and all_filters["search_tags"] is True:
        return """
            optional match p=(n:tag{account_id: $account_id})-[:tag_of]->(tagged_by_n)
            optional match rp=(n:tag{account_id: $account_id})<-[:rule_of]-(:rule)-[:applies_to]->(tagged_by_rn)
            with p, rp, n, collect(tagged_by_n) as tagged_by_n, collect(tagged_by_rn) as tagged_by_rn
            with case
                    when p is null and rp is null then [n]
                    when p is null then tagged_by_rn + [n]
                    when rp is null then tagged_by_n + [n]
                    else tagged_by_n + tagged_by_rn + [n]
                 end as items
            with apoc.coll.toSet(apoc.coll.flatten(items)) as search_matched_nodes
            UNWIND search_matched_nodes as n
        """
    return ""


def filter_special_characters(input_string: str) -> str:
    for ch in [
        "\\",
        "/",
        "#",
        "%",
        "*",
        ",",
        '"',
        "$",
        "&",
        "?",
        "!",
        "@",
        "^",
        "<",
        ">",
        "|",
        "+",
        ":",
        ";",
        "~",
    ]:
        if ch in input_string:
            input_string = input_string.replace(ch, " ")
    return input_string


def get_discovery_base(search_term, text_match_option, filters):
    """name
    id
    type - (attribute, term, metric, analysis, column, table, schema, dashboard, sheet, field, tag)
    certified - if type == term / metric / attribute / analysis
    term_name - if type == attribute
    db_name - if type == schema or table or column
    schema_name - if type == table or column
    parent_id - if type == field or column
    parent_name - if type == field or column"""

    search_term = filter_special_characters(search_term)
    search_expr, search_synonyms_list = get_synonyms(search_term, filters)
    search_in_tags_query = search_in_tags(filters)
    objects_filter = filter_by_object(filters)
    labels_filter = filter_by_label(filters)
    owners_filter = filter_by_owner(filters)
    has_owner_filter = filter_by_has_owner(filters)
    last_updated_filter = filter_last_updated(filters)
    data_type_filter = filter_by_data_type(filters)
    documentation_filter = filter_by_documentation(filters)
    data_status_filter = filter_by_data_status(filters)
    semantic_status_filter = filter_by_semantic_status(filters)
    certification_filter = filter_by_certification(filters)
    duplications_filter = filter_by_duplications(filters)
    nullables_filter = filter_by_nullable(filters)
    attribute_count_filter = filter_by_attribute_count(filters)
    column_count_filter = filter_by_column_count(filters)
    visual_count_filter = filter_by_visual_count(filters)
    tags_filter = filter_by_tag(filters)
    data_filter = filter_by_data(filters)
    usage_filter = filter_by_usage(filters)
    zones_filter = filter_by_zones(filters)

    text_match_condition = get_text_match_condition(text_match_option)

    optional_parent_match = (
        "",
        """optional match (n:column|attribute{account_id: $account_id})<-[:schema|term_of]-(parent:table|term)""",
    )["owner_ids" in filters or "has_owner" in filters]
    optional_parent_param = ("", ", parent")[
        "owner_ids" in filters or "has_owner" in filters
    ]
    optional_attr_match = (
        "",
        """optional match (n:term)-[:term_of]->(attr:attribute{account_id: $account_id})""",
    )["attributes_count" in filters]
    optional_attr_count = ("", ", count(attr) as count_attr")[
        "attributes_count" in filters
    ]
    optional_col_match = (
        "",
        """optional match (n:table)-[:schema]->(col:column{account_id: $account_id})""",
    )["columns_count" in filters]
    optional_col_count = ("", ", count(col) as count_col")["columns_count" in filters]
    optional_vis_match = (
        "",
        "optional match (n:dashboard)-[:bi_rel]->(vis:sheet)",
    )["visuals_count" in filters]
    optional_vis_count = ("", ", count(vis) as count_vis")["visuals_count" in filters]
    optional_description_include = (
        "",
        f"""union 
                    CALL db.index.fulltext.queryNodes("description_index", "({search_expr})") YIELD node as n, score
                    return n
                    """,
    )["description" in filters and filters["description"] is True]
    documents_union_search = ""
    if (
        search_term != ""
        and text_match_option == "contains"
        and _documents_allowed(filters)
    ):
        documents_union_search = """
            UNION
            MATCH (doc:document {account_id: $account_id})
            WHERE toLower(coalesce(doc.display_name, doc.name)) CONTAINS toLower($search_term)
            RETURN doc as n
        """
    optional_search_text = (
        f"""call (){{CALL db.index.fulltext.queryNodes("name_index", "({search_expr})") YIELD node as n, score
            return n {optional_description_include}
            {documents_union_search}
        }}""",
        f"match(n{labels_filter} {{account_id: $account_id}})",
    )[search_term == "" or text_match_option != "contains"]
    optional_tag_match = (
        "",
        f"""optional match (n{labels_filter})<-[:tag_of]-(tag:tag{{account_id: $account_id}})
            optional match (n{labels_filter})<-[:applies_to]-(:rule)-[:rule_of]->(rule_tag:tag{{account_id: $account_id}})
        """,
    )["tags" in filters]
    optional_tag_param = (
        "",
        ", apoc.coll.toSet(collect(tag.id) + collect(rule_tag.id)) as assigned_tag_ids",
    )["tags" in filters]

    optional_zone_match = (
        "",
        f""" CALL (n){{
        CALL apoc.case(
            [
                n:table,
                'RETURN [(n)<-[:zone_of]-(zone:zone {{account_id: $account_id}}) | zone.id] as zone_ids',

                n:db,
                'RETURN [(n)-[:schema]-(:schema)-[:schema]-(:table)<-[:zone_of]-(zone:zone {{account_id: $account_id}}) | zone.id] as 
                zone_ids',

                n:schema,
                'RETURN [(n)-[:schema]->(table:table)<-[:zone_of]-(zone:zone {{account_id: $account_id}}) | zone.id] as zone_ids',

                n:column,
                'RETURN [(n)<-[:schema]-(table:table)<-[:zone_of]-(zone:zone {{account_id: $account_id}}) | zone.id] as zone_ids',

                n:term,
                'RETURN [(n)-[:term_of]->(attr:attribute)-[:reaching|attr_of]->(col:column)<-[:schema]-(:table)<-[:zone_of]-(zone:zone{{account_id: $account_id}}) | zone.id] as zone_ids',

                n:analysis,
                '{get_analysis_zones("n")} return zone_ids',

                n:attribute,
                '{get_term_attr_zones("n")} return zone_ids',

                n:metric,
                '{get_metrics_related_zones_cypher("n")} return zone_ids'
            ],
            'RETURN [] as zone_ids',
            {{n:n, account_id: $account_id}}
            ) YIELD value
            RETURN value.zone_ids AS zone_ids }}
        """,
    )["zones" in filters]
    optional_zone_param = ("", ",zone_ids as assigned_zones_ids")["zones" in filters]
    optional_duplication_match = (
        "",
        """optional match (n)-[:similar_to]-(dup_tbl:table{account_id: $account_id})""",
    )["duplicated" in filters]
    optional_duplication_param = ("", ", dup_tbl")["duplicated" in filters]

    optional_data_schema_match = (
        "",
        """optional match(n)<-[:schema*1..2]-(parent_schema:schema)<-[:schema]-(parent_db:db{account_id: $account_id})""",
    )["data" in filters]
    optional_data_schema_param = ("", ", parent_schema, parent_db")["data" in filters]

    query = f"""{optional_search_text}
                with distinct n
                where n.account_id = $account_id
                {search_in_tags_query}
                with n where 
                n.account_id = $account_id {objects_filter}
                with n
                {optional_parent_match}
                {optional_attr_match}
                {optional_col_match}
                {optional_vis_match}
                {optional_tag_match}
                {optional_zone_match}
                {optional_duplication_match}
                {optional_data_schema_match}
                with n{optional_parent_param}{optional_attr_count}{optional_col_count}{optional_vis_count}{optional_tag_param}{optional_zone_param}{optional_duplication_param}{optional_data_schema_param}
                where 
                n.account_id = $account_id
                {text_match_condition}
                {owners_filter}
                {has_owner_filter}
                {last_updated_filter}
                {data_type_filter}
                {documentation_filter}
                {data_status_filter}
                {semantic_status_filter}
                {certification_filter}
                {duplications_filter}
                {nullables_filter}
                {attribute_count_filter}
                {column_count_filter}
                {visual_count_filter}
                {tags_filter}
                {zones_filter}
                {data_filter}
                {usage_filter}
                """

    return query, search_synonyms_list
