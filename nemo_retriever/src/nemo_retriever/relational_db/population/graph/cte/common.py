from shared.graph.cte.dal import (
    create_cte,
    get_fields_without_cte,
    delete_ctes,
    get_constant_fields,
    get_ctes_by_field_id,
    get_next_level_of_ctes,
    get_root_cte_columns,
    get_root_cte_custom_sql,
    is_cte_a_table_or_custom_query,
    get_field_relevant_columns,
)
import datetime
import re
from tqdm import tqdm
import logging


logger = logging.getLogger("common.py")


def create_sql_from_relevant_columns(
    schema_name: str, table_name: str, columns_names: list[str]
):
    columns_names.sort()
    columns = ",\n".join(columns_names)
    return f"SELECT {columns} FROM {schema_name}.{table_name}"


def construct_sql_from_field(account_id: str, field_id: str):
    root_ctes_map_to_columns = get_field_relevant_columns(account_id, field_id)

    def get_cte_sql(cte_name: str, original_cte_sql: str):
        if cte_name not in root_ctes_map_to_columns:
            return original_cte_sql
        schema_name = root_ctes_map_to_columns[cte_name]["schema_name"]
        table_name = root_ctes_map_to_columns[cte_name]["table_name"]
        columns_names = root_ctes_map_to_columns[cte_name]["columns_names"]
        return create_sql_from_relevant_columns(schema_name, table_name, columns_names)

    result = get_ctes_by_field_id(account_id, field_id)
    if len(result) == 0:
        return None
    current_cte: dict = result["cte"]
    parent_ctes: list[dict] = result["sources"]
    sql: str = get_cte_sql(current_cte["name"], current_cte["sql"])
    if not parent_ctes:
        return sql
    ctes = []
    sqls = []
    aliases = {}
    while parent_ctes:
        for cte in parent_ctes:
            cte_sql = get_cte_sql(cte["name"], cte["sql"])
            if cte_sql in sqls:
                continue
            alias = cte["name"]
            if alias in aliases:
                aliases[alias] += 1
                alias_count = aliases[alias]
                alias = alias + str(alias_count)
                if alias_count > 1:
                    replacing = alias + str(alias_count - 1)
                else:
                    replacing = cte["name"]
            else:
                aliases[alias] = 0
                replacing = None

            if replacing:
                for i, c in enumerate(ctes):
                    if replacing not in c:
                        continue
                    from_list = c.split(" FROM ")
                    if from_list and replacing in from_list[-1]:
                        from_list[-1] = from_list[-1].replace(replacing, alias)
                        ctes[i] = " FROM ".join(from_list)

            cte_formatting = f" {alias} AS (\n{cte_sql}\n) "
            ctes = [cte_formatting] + ctes
            sqls.append(cte_sql)
        parents_ctes_ids = [cte["id"] for cte in parent_ctes]
        parent_ctes = get_next_level_of_ctes(account_id, parents_ctes_ids, field_id)
    sql = " WITH " + " , \n\n".join(ctes) + "\n\n" + sql
    return sql


def get_valid_sql_name(name: str):
    name_without_parenthesis = re.sub(r"[(){}&$#@*!?]+", "", name)
    name_with_underlines = re.sub(r"[ -/]+", "_", name_without_parenthesis)
    name = name_with_underlines.lower()
    return name


def if_field_outsource_value(field: dict):
    if "formula" not in field or not field["formula"]:
        return False
    try:
        formula = field["formula"]
        proxy_pattern = r"\[([^\]]+?\.[^\]]+?)\]"
        proxy_match = re.search(proxy_pattern, formula)
        if proxy_match:
            return True
        return False
    except Exception:
        return False


def calculate_fields_ctes(
    account_id: str, created_time=datetime.datetime.now(), processed_ids=[]
):
    delete_ctes(account_id)
    level_index = 0
    fields = get_fields_without_cte(account_id, processed_ids)
    constant_fields = get_constant_fields(account_id)
    fields = constant_fields + fields
    while fields:
        with tqdm(
            desc=f"Calculate CTEs for level {level_index}",
            total=len(fields),
            mininterval=10,
            maxinterval=10,
        ) as pbar:
            for field in fields:
                if if_field_outsource_value(field):
                    processed_ids.append(field["id"])
                else:
                    create_cte_from_field(
                        account_id, field, created_time, processed_ids
                    )
                pbar.update(1)
        fields = get_fields_without_cte(account_id, processed_ids)
        level_index += 1


def create_cte_from_field(
    account_id: str, field_and_ancestors: dict, created_time: str, processed_ids=[]
):
    match field_and_ancestors["field"]["type"]:
        case "calculated_field":
            create_cte_from_calculated_field(
                account_id, field_and_ancestors, created_time, processed_ids
            )
        case "custom_sql_field":
            create_cte_from_custom_sql_field(
                account_id, field_and_ancestors, created_time, processed_ids
            )
        case "referencing_field":
            connect_referencing_field(
                account_id, field_and_ancestors, created_time, processed_ids
            )
        case "constant_field":
            connect_constant_field_to_cte(
                account_id, field_and_ancestors, created_time, processed_ids
            )
        case "error_field":
            processed_ids.append(field_and_ancestors["field"]["id"])
    return


def get_inner_expression(expression: str):
    opening_indexes = []
    closing_indexes = []
    for index, char in enumerate(expression):
        if char == "(":
            opening_indexes.append(index)
        if char == ")":
            closing_indexes.append(index)
        diff = len(closing_indexes) - len(opening_indexes)
        if diff > 0:
            break
    if diff > 0:
        last_index = closing_indexes[-diff]
        return expression[:last_index]
    return expression


# any expression pattern: a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n
lod_regex_pattern = r"({[\s]*(FIXED|EXCLUDE|INCLUDE|fixed|exclude|include) ([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n]+)[\s]*:[\s]*([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n]+)[\s]*})"


def is_lod(formula: str):
    return "{" in formula


def parse_lod_expression(expression: str):
    try:
        groups = re.findall(lod_regex_pattern, expression)
        matches: list[tuple[str, str, str]] = []
        for group in groups:
            full_expression = group[0]
            group_by = group[2]
            aggregation = group[3]
            matches.append((full_expression, group_by, aggregation))
        return matches
    except Exception:
        logger.error(f"error in expresssion {expression}")


def parse_LOD(formula: str, field_alias_name: str, from_tables: str, level: int = 0):
    new_formula = formula
    new_formula = replace_non_grouping_lod(new_formula)
    if not is_lod(new_formula):
        return new_formula
    matches = parse_lod_expression(new_formula)
    for full_expression, group_by, aggregation in matches:
        new_formula = new_formula.replace(
            full_expression,
            f"\n(SELECT {aggregation} FROM {from_tables} GROUP BY {group_by})\n",
        )
    if is_lod(new_formula):
        return parse_LOD(new_formula, field_alias_name, from_tables, level + 1)
    return new_formula


def replace_attr(formula: str):
    attr_pattern = r"(ATTR\(([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]'\\\r\n]+)\))"
    new_formula = formula
    groups = re.findall(attr_pattern, formula)
    for group in groups:
        full_expression = group[0]
        field_name = group[1]
        new_expression = f"""CASE WHEN MIN({field_name}) IS NULL THEN NULL \nWHEN MIN({field_name}) = MAX({field_name}) THEN MIN({field_name}) \nELSE {field_name} END"""
        new_formula = new_formula.replace(full_expression, new_expression)
    return new_formula


add_months_args_pattern = (
    r"([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n]+)[\s]*,[\s]*([0-9\-]+)"
)


def replace_add_months(formula: str):
    function_str: str = "add_months("
    formula_copy = formula
    last_index = formula_copy.lower().rfind(function_str)
    while last_index > -1:
        sub_string = formula_copy[last_index:]
        full_inner_expression = sub_string[len(function_str) :]
        inner_expression = get_inner_expression(full_inner_expression)
        groups = re.findall(add_months_args_pattern, inner_expression)
        if len(groups) != 1:
            return formula
        date = groups[0][0]
        months_count = groups[0][1]
        new_expression = f"date_add('month', {months_count}, {date}"
        from_index = last_index + len(inner_expression) + len(function_str)
        start = formula_copy[:last_index]
        end = formula_copy[from_index:]
        formula_copy = start + new_expression + end
        last_index = formula_copy.lower().rfind(function_str)
    return formula_copy


datepart_args_pattern = (
    r"\'([a-z]+)\'[\s]*,[\s]*([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n]+)"
)


def replace_date_part(formula: str):
    function_str: str = "datepart("
    formula_copy = formula
    last_index = formula_copy.lower().rfind(function_str)
    while last_index > -1:
        sub_string = formula_copy[last_index:]
        full_inner_expression = sub_string[len(function_str) :]
        inner_expression = get_inner_expression(full_inner_expression)
        groups = re.findall(datepart_args_pattern, inner_expression)
        if len(groups) != 1:
            return formula
        unit = groups[0][0]
        date = groups[0][1]
        new_expression = unit.lower() + "(" + date  ## week(somedate)
        from_index = last_index + len(inner_expression) + len(function_str)
        start = formula_copy[:last_index]
        end = formula_copy[from_index:]
        formula_copy = start + new_expression + end
        last_index = formula_copy.lower().rfind(function_str)
    return formula_copy


def replace_date_name(formula: str):
    function_str: str = "datename("
    formula_copy = formula
    last_index = formula_copy.lower().rfind(function_str)
    while last_index > -1:
        sub_string = formula_copy[last_index:]
        full_inner_expression = sub_string[len(function_str) :]
        inner_expression = get_inner_expression(full_inner_expression)
        groups = re.findall(datepart_args_pattern, inner_expression)
        if len(groups) != 1:
            return formula
        unit = groups[0][0]
        date = groups[0][1]
        new_expression = (
            "CAST(" + unit.lower() + "(" + date + ") AS VARCHAR"
        )  ## CAST(week(somedate) AS VARCHAR
        from_index = last_index + len(inner_expression) + len(function_str)
        start = formula_copy[:last_index]
        end = formula_copy[from_index:]
        formula_copy = start + new_expression + end
        last_index = formula_copy.lower().rfind(function_str)
    return formula_copy


def replace_makedate(formula: str):
    makedate_pattern = r"((MAKEDATE|makedate)\(([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n]+),([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n]+),([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n]+)\))"
    new_formula = formula
    groups = re.findall(makedate_pattern, formula)
    for group in groups:
        full_expression = group[0]
        year = group[2]
        month = group[3]
        day = group[4]
        new_expression = (
            f"""date(str({year}) || '-' || str({month}) || '-' || str({day}))"""
        )
        new_formula = new_formula.replace(full_expression, new_expression)
    return new_formula


def match_non_grouping_lod(formula: str) -> list[tuple[str, str]]:
    non_grouping_lod_regex_pattern_v1 = r"({[\s]*(FIXED|EXCLUDE|INCLUDE|fixed|exclude|include)[\s]*:[\s]*([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n]+)[\s]*})"
    non_grouping_lod_regex_pattern_v2 = (
        r"({([a-zA-Z0-9_$%\(ֿ\)\/\s\|\-\=\*\+<>\.,\"\[\]\'\\\r\n]+)})"
    )
    groups = re.findall(non_grouping_lod_regex_pattern_v1, formula)
    matches = []
    if groups:
        for group in groups:
            full_expression = group[0]
            aggregation = group[2]
            matches.append((full_expression, aggregation))
    else:
        groups = re.findall(non_grouping_lod_regex_pattern_v2, formula)
        for group in groups:
            full_expression = group[0]
            aggregation = group[1]
            matches.append((full_expression, aggregation))
    return matches


def replace_non_grouping_lod(formula: str):
    try:
        if not is_lod(formula):
            return formula
        new_formula = formula
        groups = match_non_grouping_lod(formula)
        for group in groups:
            full_expression = group[0]
            aggregation = group[1]
            new_formula = new_formula.replace(full_expression, aggregation)
        if is_lod(new_formula) and match_non_grouping_lod(formula):
            return replace_non_grouping_lod(new_formula)
        return new_formula
    except Exception:
        logger.error(f"replace_non_grouping_lod - error in formula {formula}")
        return formula


trailing_string_pattern = r"((\+)[\s]*(\"|\'))+"
leading_string_pattern = r"((\')[\s]*(\+))+"


## replace patterns like:
# state + "-" + city
# to: state || "-" || city
def replace_concatentation(formula: str):
    try:
        new_formula = formula
        groups = re.findall(trailing_string_pattern, formula)
        for group in groups:
            full_expression = group[0]
            _operator = group[1]
            quotation = group[2]
            new_formula = new_formula.replace(full_expression, " || " + quotation)
        groups = re.findall(leading_string_pattern, new_formula)
        for group in groups:
            full_expression = group[0]
            quotation = group[1]
            _operator = group[2]
            new_formula = new_formula.replace(full_expression, quotation + " || ")
        return new_formula
    except Exception:
        logger.error(f"replace_concatentation -error in formula {formula} ")
        return formula


def get_last_match_index(regex_exp, string):
    matches = []
    for match in re.finditer(regex_exp, string):
        matches.append(match.start())
    if len(matches) > 0:
        return matches[-1]
    return -1


## replace casting expressions
#  str([number of records] + [total revenue]) -> cast(([number of records] + [total revenue]) as str)
def replace_cast(
    formula: str, function_str: str = "str(", cast_to_type: str = "VARCHAR"
):
    formula_copy = formula
    function_pattern = get_insensitive_function_pattern(function_str)
    last_index = get_last_match_index(function_pattern, formula_copy.lower())
    while last_index > -1:
        sub_string = formula_copy[last_index:]
        full_inner_expression = sub_string[len(function_str) :]
        inner_expression = get_inner_expression(full_inner_expression)
        from_index = last_index + len(inner_expression) + len(function_str)
        start = formula_copy[:last_index]
        end = formula_copy[from_index:]
        formula_copy = start + f"CAST({inner_expression} AS {cast_to_type})" + end
        last_index = get_last_match_index(function_pattern, formula_copy.lower())
    return formula_copy


def replace_zn(formula: str, function_str: str = "zn("):
    formula_copy = formula
    last_index = formula_copy.lower().rfind(function_str)
    while last_index > -1:
        sub_string = formula_copy[last_index:]
        full_inner_expression = sub_string[len(function_str) :]
        inner_expression = get_inner_expression(full_inner_expression)
        from_index = last_index + len(inner_expression) + len(function_str)
        start = formula_copy[:last_index]
        end = formula_copy[from_index:]
        expression = f"(CASE WHEN({inner_expression}) IS NOT NULL THEN ({inner_expression}) ELSE 0 END "
        formula_copy = start + expression + end
        last_index = formula_copy.lower().rfind(function_str)
    return formula_copy


def extract_arguments_from_expression(expression: str, max_num_of_arguments: int = 2):
    expression_copy = get_inner_expression(expression)
    commas_indexes = [i for i, char in enumerate(expression_copy) if char == ","]
    if not commas_indexes:
        return [expression]
    commas_indexes.reverse()
    expressions = []
    i = 0
    while len(expressions) <= max_num_of_arguments and i < len(commas_indexes):
        current_comma_index = commas_indexes[i]
        current_part = expression_copy[current_comma_index + 1 :]
        inner_expression = get_inner_expression(current_part)
        is_full_expression = inner_expression == current_part
        if is_full_expression:
            expressions.append(inner_expression)
            expression_copy = expression_copy[:current_comma_index]
        i += 1
        # if i == len(commas_indexes) - 1:
    expressions.append(expression_copy)
    expressions.reverse()
    return expressions


def replace_coalesce(formula: str, function_str: str):
    formula_copy = formula
    last_index = formula_copy.lower().rfind(function_str)
    while last_index > -1:
        sub_string = formula_copy[last_index:]
        full_inner_expression = sub_string[len(function_str) :]
        inner_expression = get_inner_expression(full_inner_expression)
        from_index = last_index + len(inner_expression) + len(function_str)
        start = formula_copy[:last_index]
        end = formula_copy[from_index:]
        arguments = extract_arguments_from_expression(inner_expression)
        if len(arguments) < 2:
            return formula
        true = arguments[0]
        false = arguments[1]
        expression = f"(CASE WHEN({true}) IS NOT NULL THEN ({true}) ELSE ({false}) END "
        formula_copy = start + expression + end
        last_index = formula_copy.lower().rfind(function_str)
    return formula_copy


def replace_isnull(formula: str):
    formula_copy = formula
    function_str = "isnull("
    last_index = formula_copy.lower().rfind(function_str)
    while last_index > -1:
        sub_string = formula_copy[last_index:]
        full_inner_expression = sub_string[len(function_str) :]
        inner_expression = get_inner_expression(full_inner_expression)
        from_index = last_index + len(inner_expression) + len(function_str)
        start = formula_copy[:last_index]
        end = formula_copy[from_index:]
        expression = f"(CASE WHEN({inner_expression}) IS NULL THEN TRUE ELSE FALSE END "
        formula_copy = start + expression + end
        last_index = formula_copy.lower().rfind(function_str)
    return formula_copy


def replace_window(formula: str):
    window_pattern = r"([WINDOW|RUNNING|LOOKUP]+_([A-Z]+\())"
    new_formula = formula
    last_index = get_last_match_index(window_pattern, new_formula)
    while last_index > -1:
        group = re.findall(window_pattern, formula)[-1]
        full_window_function: str = group[0]  # WINDOW_SUM(
        aggregation: str = group[1]  # SUM( \ AVG(
        arguments_start_index = last_index + len(full_window_function)
        internal_expression = new_formula[arguments_start_index:]
        all_arguments_string = get_inner_expression(internal_expression)
        arguments = extract_arguments_from_expression(internal_expression, 3)
        start = new_formula[:last_index]
        after_expression_index = arguments_start_index + len(all_arguments_string)
        end = new_formula[after_expression_index:]
        new_formula = start + aggregation + arguments[0] + end
        last_index = get_last_match_index(window_pattern, new_formula)
    return new_formula


comment_pattern = r"(\/\/.+\n)"
oracle_comment_pattern = r"(\/\*(.[^\/\*]|\n)+\*\/)"


def remove_oracle_comments(formula: str):
    new_formula = formula
    groups = re.findall(oracle_comment_pattern, formula)
    for group in groups:
        comment = group[0]
        new_formula = new_formula.replace(comment, "")
    return new_formula


double_quotes_pattern = r'\"([^"]*)\"'


def convert_to_single_quotes(expression: str):
    new_expression = expression
    double_quotes = re.findall(double_quotes_pattern, expression)
    for match in double_quotes:
        match_with_escaped_quote = match.replace("'", "''")
        new_expression = new_expression.replace(
            f'"{match}"', f"'{match_with_escaped_quote}'"
        )
    return new_expression


def convert_oracle_functions_to_presto(formula: str):
    formula = remove_oracle_comments(formula)
    formula = convert_to_single_quotes(formula)
    formula = replace_insensitive(
        formula, "sysdate", "current_date", no_parenthesis=True
    )
    formula = replace_insensitive(formula, "stdev(", "STTDEV(")
    formula = replace_insensitive(formula, "datetrunc(", "date_trunc(")
    formula = replace_insensitive(formula, "dateadd(", "date_add(")
    formula = replace_insensitive(formula, "datediff(", "date_diff(")
    formula = replace_insensitive(formula, "cube", '"cube"', no_parenthesis=True)
    formula = replace_cast(formula, "to_number(", "INTEGER")
    formula = replace_cast(formula, "to_char(", "VARCHAR")
    formula = replace_add_months(formula)
    formula = replace_zn(formula)
    formula = replace_coalesce(formula, "ifnull(")
    formula = replace_coalesce(formula, "nvl(")
    formula = replace_isnull(formula)
    formula = replace_date_part(formula)
    formula = replace_date_name(formula)
    return formula


charactes_to_escape = "()[]?*+-|^$\/.&~#"


def handle_char_to_regex(char: str):
    if char.lower() != char.upper():
        return f"[{char.lower()}|{char.upper()}]"
    if char == " ":
        return "\s"
    if char in charactes_to_escape:
        return f"\{char}"
    return char


def get_insensitive_function_pattern(function_name: str, no_parenthesis: bool = False):
    chars = list(function_name.lower())
    chars_patterns = [handle_char_to_regex(char) for char in chars]
    regex_pattern = "".join(chars_patterns)

    if no_parenthesis:
        regex_pattern = rf"[^a-zA-Z_\"\']({regex_pattern})[^a-zA-Z_\"\']+|(^{regex_pattern})[^a-zA-Z_\"\']+"
    else:
        regex_pattern = rf"[^a-zA-Z_\"\']({regex_pattern})+|(^{regex_pattern})+"
    return regex_pattern


def sub_by_index(
    string: str,
    index_of_substr: int,
    substr: str,
    replace: str,
    no_parenthesis: bool = False,
):
    if not substr:
        return string
    from_index = len(substr) + index_of_substr
    start = (
        string[:index_of_substr]
        if index_of_substr == 0
        else string[: index_of_substr + 1]
    )
    end = string[from_index:] if index_of_substr == 0 else string[from_index + 1 :]
    new_string = start + replace + end
    return new_string


def replace_insensitive(
    formula: str, function_name: str, replace: str, no_parenthesis: bool = False
):
    new_formula = formula
    regex_pattern = get_insensitive_function_pattern(function_name, no_parenthesis)
    last_index = get_last_match_index(regex_pattern, new_formula)
    while last_index > -1:
        new_formula = sub_by_index(
            new_formula, last_index, function_name, replace, no_parenthesis
        )
        last_index = get_last_match_index(regex_pattern, new_formula)
    return new_formula


def parse_tableau_formula_to_sql(formula: str, field_alias_name: str, from_tables: str):
    formula = re.sub(comment_pattern, "\n", formula)
    formula = replace_non_grouping_lod(formula)
    is_formula_lod = is_lod(formula)
    if is_formula_lod:
        formula = parse_LOD(formula, field_alias_name, from_tables)
    formula = convert_oracle_functions_to_presto(formula)
    formula = replace_insensitive(formula, "left(", "substr(")
    formula = replace_insensitive(formula, "len(", "length(")
    formula = replace_insensitive(formula, "mid(", "substr(")
    formula = replace_insensitive(formula, "right(", "trail(")
    formula = replace_insensitive(formula, "total(", "(")
    formula = replace_insensitive(formula, "elseif", "WHEN ", no_parenthesis=True)
    formula = replace_insensitive(formula, "if", "CASE WHEN", no_parenthesis=True)
    formula = replace_insensitive(formula, "countd(", "COUNT( DISTINCT ")
    formula = replace_insensitive(formula, "isoweek", "WEEK", no_parenthesis=True)
    formula = replace_insensitive(
        formula, "today()", "current_date", no_parenthesis=True
    )
    formula = replace_insensitive(formula, "regex_replace(", "regexp_replace(")
    formula = replace_insensitive(formula, "regex_match(", "regexp_like(")
    formula = replace_insensitive(formula, "rank_unique(", "rank(")
    formula = replace_makedate(formula)
    formula = replace_cast(formula, "str(")
    formula = replace_cast(formula, "float(", "DECIMAL")
    formula = replace_cast(formula, "int(", "INTEGER")
    formula = replace_concatentation(formula)
    formula = replace_attr(formula)
    formula = replace_window(formula)
    if is_formula_lod:
        cte_sql = "SELECT " + formula + f" AS {field_alias_name} FROM {from_tables}"
    else:
        cte_sql = f"SELECT ({formula}) AS {field_alias_name} FROM {from_tables}"
    return cte_sql


def create_cte_from_calculated_field(
    account_id: str,
    field_and_ancestors: dict,
    created_time: str,
    processed_ids: list[str],
):
    ancestors = field_and_ancestors["ancestors"]
    field = field_and_ancestors["field"]
    processed_ids.append(field["id"])
    ancestors_names_to_aliases = [
        {
            "name": ancestor["name"],
            "alias": ancestor["cte_alias"],
            "cte_name": ancestor["cte"]["name"],
        }
        for ancestor in ancestors
    ]
    from_aliases = []
    for ancestor in ancestors_names_to_aliases:
        if is_cte_a_table_or_custom_query(ancestor["cte_name"]):
            alias = ancestor["cte_name"]
        else:
            alias = f"cte_{ancestor['alias']}"
        if alias not in from_aliases:
            from_aliases.append(alias)
    from_aliases.sort()
    from_aliases = ", ".join(from_aliases)
    formula: str = field["formula"]
    for ancestor_alias_name in ancestors_names_to_aliases:
        formula = formula.replace(
            f"[{ancestor_alias_name['name']}]", ancestor_alias_name["alias"]
        )

    field_sql_alias = get_valid_sql_name(field["name"])
    cte_name = f"cte_{field_sql_alias}"
    cte_sql = parse_tableau_formula_to_sql(formula, field_sql_alias, from_aliases)
    create_cte(
        account_id,
        cte_sql,
        cte_name,
        field_sql_alias,
        field["id"],
        created_time,
    )
    return


def connect_constant_field_to_cte(
    account_id: str,
    field: dict,
    created_time: str,
    processed_ids: list[str],
):
    field = field["field"]
    processed_ids.append(field["id"])
    formula: str = field["formula"]
    field_sql_alias = get_valid_sql_name(field["name"])
    cte_name = f"cte_{field_sql_alias}"
    cte_sql = f"SELECT {formula} \n AS {field_sql_alias}"
    create_cte(
        account_id,
        cte_sql,
        cte_name,
        field_sql_alias,
        field["id"],
        created_time,
        is_constant=True,
    )
    return


def create_cte_from_custom_sql_field(
    account_id: str,
    field_and_ancestors: dict,
    created_time: str,
    processed_ids: list[str],
):
    field = field_and_ancestors["field"]
    processed_ids.append(field["id"])
    field_sql_alias = get_valid_sql_name(field["name"])
    cte_sql, sql_name = get_root_cte_custom_sql(account_id, field["id"])
    cte_sql = convert_oracle_functions_to_presto(cte_sql)
    sql_name = get_valid_sql_name(sql_name)
    cte_name = "cte_custom_query"
    create_cte(
        account_id,
        cte_sql,
        cte_name,
        field_sql_alias,
        field["id"],
        created_time,
    )
    return


def connect_referencing_field(
    account_id: str,
    field_and_ancestor: dict,
    created_time: str,
    processed_ids: list[str],
):
    ancestor = field_and_ancestor["ancestors"][0]
    field = field_and_ancestor["field"]
    processed_ids.append(field["id"])
    if ancestor["type"] == "column":
        column_name = ancestor["name"]
        columns_names = get_root_cte_columns(account_id, field["id"])
        cte_sql = create_sql_from_relevant_columns(
            schema_name=ancestor["schema_name"],
            table_name=ancestor["table_name"],
            columns_names=columns_names,
        )
        create_cte(
            account_id,
            cte_sql,
            f"cte_table_{ancestor['schema_name']}_{ancestor['table_name']}".lower(),
            column_name,
            field["id"],
            created_time,
        )
    else:
        cte = ancestor["cte"]
        create_cte(
            account_id,
            cte["sql"],
            cte["name"],
            ancestor["cte_alias"],
            field["id"],
            created_time,
        )
    return
