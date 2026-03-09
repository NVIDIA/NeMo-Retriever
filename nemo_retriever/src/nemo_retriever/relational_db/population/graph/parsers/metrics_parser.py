import re
from functools import reduce
from shared.graph.dal.metrics_dal import (
    get_metric_snippets_combinations,
    get_child_metric_id,
    SnippetError,
    add_common_dimensions_to_metric,
    add_common_filters_to_metric,
    get_child_metrics,
)
from shared.graph.model.node import Node
from shared.graph.model.metric import Metric

from shared.graph.model.reserved_words import (
    Labels,
    Parser,
    SQL,
    SQLFunctions,
    DataTypes,
    get_types_families,
)
from shared.graph.parsers.sql.op_name_to_symbol import get_symbol

from sqloxide import parse_sql
import pandas as pd
import pendulum

import logging

logger = logging.getLogger("metrics_parser.py")


class UnbalancedError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class TypesError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def create_between_node(
    metric_obj: Metric,
    between_dict: dict,
    parent: Node | None,
    account_id: str,
    optional_edge_params=None,
):
    negated = between_dict["negated"]
    if negated:
        operator = "NotBetween"
    else:
        operator = "Between"
    between_node = Node(operator, label=Labels.OPERATOR, props={"name": operator})
    metric_obj.add_edge((parent, between_node, optional_edge_params), "formula")

    v_types_expr = handle_expression(
        metric_obj, between_dict["expr"], between_node, account_id, {"child_idx": 0}
    )

    v_types_low = handle_expression(
        metric_obj, between_dict["low"], between_node, account_id, {"child_idx": 1}
    )
    v_types_high = handle_expression(
        metric_obj, between_dict["high"], between_node, account_id, {"child_idx": 2}
    )
    common_data_types = filter_common_data_types(
        [v_types_expr, v_types_low, v_types_high], "Between", between_node, metric_obj
    )
    return common_data_types


def create_interval_node(
    metric_obj, interval_dict, parent, account_id, optional_edge_params=None
):
    interval_node = Node("Interval", label=Labels.OPERATOR, props={"name": "Interval"})
    metric_obj.add_edge((parent, interval_node, optional_edge_params), "formula")

    handle_expression(
        metric_obj, interval_dict["value"], interval_node, account_id, {"child_idx": 0}
    )
    leading_field_node = Node(
        name=interval_dict["leading_field"],
        label=Labels.CONSTANT,
        props={"name": interval_dict["leading_field"]},
    )
    leading_field_node.props.update({"data_type": "datetime"})
    metric_obj.add_edge(
        (interval_node, leading_field_node, {"child_idx": 1}), "formula"
    )
    return ["datetime"]


def create_case_node(
    metric_obj, case_dict, parent, account_id, optional_edge_params=None
):
    case_node = Node("Case", label=Labels.OPERATOR, props={"name": "Case"})
    metric_obj.add_edge((parent, case_node, optional_edge_params), "formula")

    conditions_list = case_dict["conditions"]
    results_list = case_dict["results"]
    else_result = case_dict["else_result"]

    if "Value" in conditions_list[0].keys():
        handle_expression(
            metric_obj,
            case_dict["operand"],
            case_node,
            account_id,
            {"case_part": "value"},
        )

    for i in range(len(conditions_list)):
        handle_expression(
            metric_obj,
            conditions_list[i],
            case_node,
            account_id,
            {"case_part": f"{i}_cond"},
        )
        v_types_res = handle_expression(
            metric_obj,
            results_list[i],
            case_node,
            account_id,
            {"case_part": f"{i}_res"},
        )

    if else_result is not None:
        _ = handle_expression(
            metric_obj, else_result, case_node, account_id, {"case_part": "else"}
        )

    case_node.props.update({"number_conditions": len(conditions_list)})
    return v_types_res


def create_like_node(
    metric_obj, like_dict, parent, account_id, optional_edge_params=None
):
    negated = like_dict["negated"]
    if negated:
        operator = "Not Like"
    else:
        operator = "Like"
    like_node = Node(operator, label=Labels.OPERATOR, props={"name": operator})
    metric_obj.add_edge((parent, like_node, optional_edge_params), "formula")

    v_types = handle_expression(
        metric_obj, like_dict["expr"], like_node, account_id, {"child_idx": 0}
    )
    _ = handle_expression(
        metric_obj, like_dict["pattern"], like_node, account_id, {"child_idx": 1}
    )
    return v_types


def create_isnull_node(
    metric_obj, isnull_dict, parent, account_id, optional_edge_params=None
):
    isnull_node = Node(name="isNull", label=Labels.OPERATOR, props={"name": "isNull"})
    metric_obj.add_edge((parent, isnull_node, optional_edge_params), "formula")
    v_types = handle_expression(
        metric_obj, isnull_dict, isnull_node, account_id, optional_edge_params
    )
    return v_types


def create_isnotnull_node(
    metric_obj, isnotnull_dict, parent, account_id, optional_edge_params=None
):
    isnotnull_node = Node(
        name="isNotNull", label=Labels.OPERATOR, props={"name": "isNotNull"}
    )
    metric_obj.add_edge((parent, isnotnull_node, optional_edge_params), "formula")
    v_types = handle_expression(
        metric_obj, isnotnull_dict, isnotnull_node, account_id, optional_edge_params
    )
    return v_types


def create_function_node(
    metric_obj: Metric,
    function_dict: dict,
    parent: Node | None,
    account_id: str,
    optional_edge_params=None,
):
    function_name = function_dict["name"][0]["value"].lower()
    if function_name.lower() not in ["filter", "over", "groupby"]:
        function_node = Node(
            name=function_name, label=Labels.FUNCTION, props={"name": function_name}
        )
        metric_obj.add_edge((parent, function_node, optional_edge_params), "formula")

    args_list = (
        function_dict["args"]
        if isinstance(function_dict["args"], list)
        else function_dict["args"]["List"]["args"]
    )
    if len(args_list) == 0:
        return_types = check_func_types_compatible(
            function_name, function_node, [""], metric_obj
        )

    elif len(args_list) == 1 and (
        function_name.lower() in ["filter", "over", "groupby"]
    ):
        arg = args_list[0]
        arg_type = next(iter(arg))
        v_types = handle_expression(
            metric_obj, arg[arg_type], parent, account_id, optional_edge_params
        )
        return_types = v_types

    elif function_name.lower() == "if":
        _ = handle_expression(
            metric_obj,
            args_list[0],
            function_node,
            account_id,
            {"case_part": "0_cond"},
        )
        v_types_true = handle_expression(
            metric_obj,
            args_list[1],
            function_node,
            account_id,
            {"case_part": "1_res"},
        )
        if len(function_dict["args"]) == 3:
            v_types_false = handle_expression(
                metric_obj,
                args_list[2],
                function_node,
                account_id,
                {"case_part": "2_else"},
            )
            common_data_types = filter_common_data_types(
                [v_types_true, v_types_false],
                "if",
                function_node,
                metric_obj,
            )
            return_types = common_data_types
        else:
            return_types = v_types_true
    else:
        for a_idx, arg in enumerate(args_list):
            arg_type = next(iter(arg))
            if arg_type == "Unnamed":
                if (function_name in ["datediff", "date_part", "dateadd"]) and next(
                    iter(arg[arg_type]["Expr"])
                ) == "Identifier":
                    arg_dict = arg[arg_type]
                    if arg[arg_type]["Expr"]["Identifier"]["value"].lower() in [
                        "day",
                        "dow",
                        "datetime",
                        "varchar",
                    ]:
                        arg_dict = {
                            Parser.VALUE: arg[arg_type]["Expr"][Parser.IDENTIFIER]
                        }
                    v_types = handle_expression(
                        metric_obj,
                        arg_dict,
                        function_node,
                        account_id,
                        {"child_idx": a_idx},
                    )
                    return_types = check_func_types_compatible(
                        function_name, function_node, v_types, metric_obj
                    )
                else:
                    v_types = handle_expression(
                        metric_obj,
                        arg[arg_type],
                        function_node,
                        account_id,
                        {"child_idx": a_idx},
                    )
                    return_types = check_func_types_compatible(
                        function_name, function_node, v_types, metric_obj
                    )
            else:
                raise ValueError(
                    f'Unknown function arg type: {arg_type} in "create_function_node". function_dict:\n {str(function_dict)}'
                )

        if function_dict["args"]["List"]["duplicate_treatment"] is not None:
            function_node.props.update(
                {"distinct": function_dict["args"]["List"]["duplicate_treatment"]}
            )

    return return_types


def create_binary_op_node(
    metric_obj: Metric,
    binaryop_dict: dict,
    parent: Node | None,
    account_id: str,
    optional_edge_params=None,
):
    binaryop_node = Node(
        name=binaryop_dict["op"],
        label=Labels.OPERATOR,
        props={"name": binaryop_dict["op"]},
    )
    symbol, _ = get_symbol(binaryop_dict["op"])

    metric_obj.add_edge((parent, binaryop_node, optional_edge_params), "formula")
    left_dict = binaryop_dict["left"]

    v_types_left = handle_expression(
        metric_obj, left_dict, binaryop_node, account_id, {"child_idx": 0}
    )

    right_dict = binaryop_dict["right"]
    v_types_right = handle_expression(
        metric_obj, right_dict, binaryop_node, account_id, {"child_idx": 1}
    )

    if symbol in ["and", "or"]:
        return ["boolean"]
    common_data_types = filter_common_data_types(
        [v_types_left, v_types_right],
        symbol,
        binaryop_node,
        metric_obj,
    )
    return common_data_types


def create_unary_op_node(
    metric_obj: Metric,
    unary_dict: dict,
    parent: Node | None,
    account_id: str,
    optional_edge_params=None,
):
    unary_node = Node(
        name=unary_dict["op"], label=Labels.OPERATOR, props={"name": unary_dict["op"]}
    )
    metric_obj.add_edge((parent, unary_node, optional_edge_params), "formula")
    child_dict = unary_dict["expr"]

    v_types = handle_expression(metric_obj, child_dict, unary_node, account_id)
    return v_types


def create_tuple_node(
    metric_obj: Metric,
    value_dict: dict,
    parent: Node | None,
    account_id: str,
    optional_edge_params=None,
):
    if (
        parent.props["label"].lower() == "function"
    ):  # parent is nested function, put alias on subselect and put subselect to main sql FROM
        parent.props.update({"nested_aggr_for_groupby": True})
    aggr = value_dict[0]
    cond = value_dict[1]
    item_node = Node(
        name="metric_item", label=Labels.METRIC_ITEM, props={"name": "metric_item"}
    )
    agg_node = Node(name="agg", label=Labels.METRIC_AGG, props={"name": "agg"})
    cond_node = Node(
        name=cond["Function"]["name"][0]["value"],
        label=Labels.METRIC_COND,
        props={"name": cond["Function"]["name"][0]["value"]},
    )  # name could be Over or Filter

    metric_obj.add_edge((parent, item_node, optional_edge_params), "formula")
    metric_obj.add_edge((item_node, agg_node, {"child_idx": 0}), "formula")

    return_types = handle_expression(
        metric_obj, aggr, agg_node, account_id, optional_edge_params
    )

    if len(cond) > 0:
        metric_obj.add_edge((item_node, cond_node, {"child_idx": 1}), "formula")
        _ = handle_expression(
            metric_obj, cond, cond_node, account_id, optional_edge_params
        )

    return return_types


def create_value_node(
    metric_obj, value_dict, parent, account_id, optional_edge_params=None
):
    value_type = next(iter(value_dict))
    if value_type == "Number":
        val = value_dict[value_type][0]

    elif value_type == "SingleQuotedString":
        val = value_dict[value_type]
        try:
            pendulum.parse(val, strict=False)
            if not val.isdigit():
                value_type = "datetime"
        except Exception:
            pass

    elif value_type == "Boolean":
        val = value_dict[value_type]

    elif value_type == "Interval":
        val = f"Interval {value_dict[value_type]['value']} {value_dict[value_type]['leading_field']}"

    elif value_type in ["data_type", "value"]:
        val = value_dict["value"]
        if val in SQLFunctions.editable_time_functions:
            val = SQLFunctions.editable_time_functions[val]
            value_type = "datetime"
        elif val.lower() in SQLFunctions.dates_parameters:
            value_type = val.lower()
        elif val in SQLFunctions.editable_numeric_functions:
            val = SQLFunctions.editable_numeric_functions[val]
            value_type = "int"

    elif value_dict == "Null":
        val = "Null"

    else:
        raise ValueError(f"Unknown value type: {value_type} ")

    if not pd.isna(val) and not pd.isnull(val):
        value_node = Node(
            name="Value",
            label=Labels.CONSTANT,
            props={"name": val, "data_type": value_type},
        )
        metric_obj.add_edge((parent, value_node, optional_edge_params), "formula")

        if value_type == "SingleQuotedString":
            val = "'" + str(val) + "'"

        value_node.props.update({"data_type": value_type})

    return [value_type]


def extract_attr_name_id_from_expr_dict(expr_dict: dict, metric: Metric):
    element = expr_dict["Value"]["SingleQuotedString"]
    term_name = element.split(".")[0]
    attr_name = element.split(".")[1]
    attr_id = metric.get_attr_id_from_term_attr_name(term_name, attr_name)
    return attr_id, attr_name


def connect_to_attr_metric(
    metric_obj: Metric,
    value_dict: dict,
    parent,
    account_id: str,
    optional_edge_params=None,
    from_bi_fields: list[str] = None,
):
    element = value_dict["Value"]["SingleQuotedString"]
    if "." in element:  # bt.attr
        attr_id, attr_name = extract_attr_name_id_from_expr_dict(value_dict, metric_obj)
        data_types = metric_obj.get_attr_data_types(attr_id)
        value_types = data_types
        value_node = Node(
            name=attr_name,
            label=Labels.ATTR,
            existing_id=attr_id,
            props={"name": attr_name, "clean_name": attr_name},
        )
        metric_obj.add_edge((parent, value_node, optional_edge_params), "formula")

    else:  # metric
        value_types = ["int"]
        metric_id = get_child_metric_id(element, account_id, from_bi_fields[0])
        value_node = Node(
            name=element,
            label=Labels.METRIC,
            existing_id=metric_id,
            props={"name": element},
        )
        metric_obj.add_edge((parent, value_node, optional_edge_params), "formula")
    return value_types


def get_next_expression(expression_dict: dict):
    expr_dict = expression_dict
    expr_type = next(iter(expr_dict))
    while expr_type.lower() in [
        "nested",
        "expr",
        "unnamedexpr",
        "unnamed",
    ]:  # ignore being nested
        expr_dict = expr_dict[expr_type]
        expr_type = next(iter(expr_dict))
    return expr_dict, expr_type


def handle_expression(
    metric_obj, expr_dict, parent, account_id, optional_edge_params=None
):
    expr_dict, expr_type = get_next_expression(expr_dict)
    if len(expr_dict) > 1:
        raise Exception(
            f'Unexpected structure of "handle_expression". expr_dict\n: {expr_dict}'
        )
    if expr_type == Parser.BINARY_OP:
        v_types = create_binary_op_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type == Parser.UNARY_OP:
        v_types = create_unary_op_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type == SQL.BETWEEN:
        v_types = create_between_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type == Parser.INTERVAL:
        v_types = create_interval_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type == SQL.CASE:
        v_types = create_case_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type == Parser.LIKE:
        v_types = create_like_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type == Parser.ISNULL:
        v_types = create_isnull_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type == Parser.ISNOTNULL:
        v_types = create_isnotnull_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type in [Parser.CAST, Parser.TRIM, Parser.CONVERT]:
        if expr_type in [Parser.CAST, Parser.CONVERT]:
            if isinstance(expr_dict[expr_type]["data_type"], str):
                casted_dtype = expr_dict[expr_type]["data_type"]
            else:
                for k, v in expr_dict[expr_type]["data_type"].items():
                    if isinstance(v, int):
                        casted_dtype = f"{k}({v})"
                    elif v is None or None in v:
                        casted_dtype = f"{k}"
                    else:
                        casted_dtype = f"{k}{v}"

        expr_dict = expr_dict[expr_type]["expr"]
        if (
            next(iter(expr_dict)) == Parser.IDENTIFIER
            and expr_dict[next(iter(expr_dict))]["value"] == "SYSDATE"
        ):
            expr_dict = {Parser.VALUE: expr_dict[Parser.IDENTIFIER]}
        _ = handle_expression(
            metric_obj, expr_dict, parent, account_id, optional_edge_params
        )  # only casted type matters
        v_types = [casted_dtype]

    elif expr_type == Parser.FUNCTION:
        v_types = create_function_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type in [Parser.VALUE, Parser.TYPED_STRING, Parser.IDENTIFIER]:
        v_types = create_value_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    elif expr_type == "Array":  # [bt.attr] or [metric] are parsed as arrays
        v_types = connect_to_attr_metric(
            metric_obj,
            expr_dict[expr_type]["elem"][0],
            parent,
            account_id,
            optional_edge_params,
        )
    elif expr_type == "ArrayIndex":
        v_types = connect_to_attr_metric(
            metric_obj,
            expr_dict[expr_type]["indexes"][0],
            parent,
            account_id,
            optional_edge_params,
        )
    elif expr_type.lower() == Parser.SUBQUERY.lower():
        v_types = handle_expression(
            metric_obj,
            expr_dict["Subquery"]["body"]["Select"]["projection"][0],
            parent,
            account_id,
            optional_edge_params,
        )
    elif expr_type == Parser.TUPLE:  # (..., Filter()) is a tuple
        v_types = create_tuple_node(
            metric_obj, expr_dict[expr_type], parent, account_id, optional_edge_params
        )
    else:
        raise ValueError(
            f'Unknown expression type: {expr_type} in "handle_expression". expr_dict:\n {str(expr_dict)}'
        )
    return v_types


def filter_common_data_types(
    data_types_lists: list[list[DataTypes]], op_name: str, op_node: Node, metric: Metric
):
    data_types_families = [get_types_families(types) for types in data_types_lists]
    # we are looking for the datatypes common between ALL the types lists (data_types_lists)
    # so for [string], [string,number], [string,date] -> we will expect to receive string
    common_data_types = reduce(
        lambda common_set, current_types: list(
            set(common_set).intersection(set(current_types))
        ),
        data_types_families,
    )
    if len(common_data_types) == 0:
        families = " and ".join([",".join(types) for types in data_types_families])
        raise TypesError(
            f"""Could not perform operation "{op_name}" due to a type mismatch. no match between {families} """
        )

    metric.filter_incompatible_combinations(op_node, common_data_types)
    return common_data_types


def check_func_types_compatible(
    func_name: str, func_node: Node, arg_data_types: list[str], metric_obj: Metric
) -> list[str]:
    arg_types_to_lower = [arg.lower() for arg in arg_data_types if arg is not None]
    func_valid_args_types = SQLFunctions.all_funcs[func_name.lower()][0]
    compatible_types = list(
        set(arg_types_to_lower).intersection(set(func_valid_args_types))
    )
    if len(compatible_types) == 0:
        # function_types_familes = get_types_families(func_valid_args_types)
        # arg_types_families = get_types_families(arg_types_to_lower)
        # raise TypesError(
        #     f'Incompatibile types: the function "{func_name}" accepts only {",".join(function_types_familes)}. but got {",".join(arg_types_families)}.'
        # )
        pass
    metric_obj.filter_incompatible_combinations(func_node, compatible_types)
    return compatible_types


def check_validity(formula: str):
    def balanced_parentheses(myStr):
        open_list = ["[", "("]
        close_list = ["]", ")"]
        stack = []
        for index, i in enumerate(myStr):
            if i in open_list:
                stack.append({i: index})
            elif i in close_list:
                pos = close_list.index(i)
                if (len(stack) > 0) and (
                    open_list[pos] == list(stack[len(stack) - 1].keys())[0]
                ):
                    stack.pop()
                else:
                    raise UnbalancedError(
                        f'Unbalanced paranthesis. {myStr[: index + 1]} <- "{i}" is closed but not opened.'
                    )
        if len(stack) != 0:
            raise UnbalancedError(
                f'Unbalanced paranthesis. {myStr[: (list(stack[-1].values())[0]) + 1]} <- "{list(stack[-1].keys())[0]}" is opened but not closed.'
            )

    balanced_parentheses(formula)


def parse_single_metric(
    account_id,
    metric_obj: Metric,
    dialect="postgres",
    from_bi_fields: list[str] = None,
    population=False,
):
    try:
        check_validity(metric_obj.formula)
    except UnbalancedError as e:
        logger.error(f"metric [{metric_obj.name}] is not well defined.\n {e}")
        raise e

    formula = metric_obj.formula.replace("[", "['").replace("]", "']")
    if not from_bi_fields:
        # Regex to extract GROUP BY(...) clause
        group_by_match = re.search(r"GROUP BY\s*\([^)]+\)", formula, re.IGNORECASE)

        if group_by_match:
            group_by_clause = group_by_match.group()
            formula_without_group_by = formula.replace(group_by_clause, "").strip()
            formula_without_group_by = re.sub(r"\s+", " ", formula_without_group_by)

            query = f"SELECT {formula_without_group_by} FROM data {group_by_clause}"
        else:
            query = f"SELECT {formula.strip()} FROM data"
        try:
            parsed_query = parse_sql(sql=query, dialect=dialect)
        except ValueError as e:
            logger.error(
                f"metric [{metric_obj.name}] {e.args[0].replace('Query', '').replace('sql', '')}"
            )
            raise e

        try:
            snippets_df = get_metric_snippets_combinations(
                account_id,
                metric_obj.formula,
                metric_obj.id,
                metric_obj.name,
                population,
            )
            metric_obj.set_metric_snippets(snippets_df)
        except SnippetError as e:
            raise e

        try:
            handle_expression(
                metric_obj,
                parsed_query[0]["Query"]["body"]["Select"]["projection"][0][
                    "UnnamedExpr"
                ],
                metric_obj.metric_node,
                account_id,
            )
        except (SnippetError, TypesError, ValueError) as e:
            logger.error(f"Metric [{metric_obj.name}] is not well defined.\n {e}")
            raise e
    else:
        sub_metrics = get_child_metrics(account_id, from_bi_fields[0])
        for sub_metric in sub_metrics:
            sub_metric_node = Node(
                sub_metric["name"],
                label=Labels.METRIC,
                existing_id=sub_metric["id"],
                props=sub_metric,
                match_props=sub_metric,
            )
            metric_obj.add_edge((metric_obj.metric_node, sub_metric_node), "formula")

    return metric_obj


##TODO: what is this for?
def find_common_dimensions_and_filters(nodes_found, account_id, metric_obj):
    nodes_for_dimensions = nodes_found["subset"][0]
    add_common_dimensions_to_metric(
        str(metric_obj.id), nodes_for_dimensions, account_id
    )
    add_common_filters_to_metric(str(metric_obj.id), nodes_for_dimensions, account_id)
