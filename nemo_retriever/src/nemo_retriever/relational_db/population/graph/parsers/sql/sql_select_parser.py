import logging
import uuid
import networkx as nx

# import traceback
from typing import Callable, Optional, Union, Any
from functools import reduce

from shared.graph.model.query import (
    Query,
    MissingDataError,
    NoFKError,
    handle_node_update,
    is_join,
    get_should_update,
)
from shared.graph.model.reserved_words import (
    Props,
    Labels,
    SQL,
    Parser,
    ArgsForSQLFunctions,
    SQLFunctions,
    JoinNodes,
    SQLFunctionsWithConsantArg,
)
from shared.graph.model.node import Node
from shared.graph.parsers.sql.op_name_to_symbol import get_symbol
from shared.graph.parsers.sql.utils import get_key_recursive
import pendulum
from shared.graph.services.queries_comparison.compare_queries import (
    remove_aliases_from_graph,
)

logger = logging.getLogger("sql_select_parser.py")
keep_string_values = False
data_type = []


class ValuesBodyError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def flatten_nested_list(nested_list):
    new_list = []
    for item in nested_list:
        if item:
            if isinstance(item, str) and item not in new_list:
                new_list.append(item)
            elif isinstance(item, list):
                new_list.extend(flatten_nested_list(item))
    return new_list


def prepare_data_types(data_types: Union[list[str], list[str], str, None]) -> list[str]:
    if isinstance(data_types, str):
        return [data_types]
    if isinstance(data_types, list):
        return flatten_nested_list(data_types)
    return []


def create_edge_params(optional_edge_params, sql_id):
    edge_params = {Props.SQL_ID: str(sql_id)}
    if optional_edge_params is not None:
        edge_params.update(optional_edge_params)
    return edge_params


def create_top_node(
    query_obj: Query,
    top_dict: Optional[dict[str, Any]],
    parent: Node,
    sql_id: str,
    is_full_parse: bool = False,
) -> tuple[str, str]:
    if top_dict is None:
        return "", ""
    label = Labels.COMMAND
    top_node = query_obj.create_current_node(
        name="Top",
        label=label,
        props={"name": "Top", Props.SQL_ID: str(sql_id)},
        section_type=SQL.TOP,
        edge_params=create_edge_params(None, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    if "Constant" in top_dict["quantity"]:
        quantity_value = top_dict["quantity"]["Constant"]
        _, value_sql_str, sql_part, data_type = create_value_node(
            query_obj=query_obj,
            section_type=SQL.TOP,
            value_dict=quantity_value,
            parent=top_node,
            sql_id=sql_id,
            is_full_parse=is_full_parse,
        )
    else:
        quantity_value = top_dict["quantity"]
        _, value_sql_str, sql_part, data_type = handle_expression(
            query_obj=query_obj,
            section_type=SQL.TOP,
            expr_dict=quantity_value,
            parent=top_node,
            sql_id=sql_id,
            is_full_parse=is_full_parse,
        )
    pointer_str = "top " + value_sql_str
    handle_node_update(
        label=label,
        node=top_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": pointer_str, "sql_part": pointer_str},
    )
    return pointer_str, pointer_str


def create_distinct_node(
    query_obj: Query,
    section_type: str,
    parent: Node,
    sql_id: str,
    is_full_parse: bool = False,
) -> None:
    _ = query_obj.create_current_node(
        name="Distinct",
        label=Labels.COMMAND,
        props={"name": "Distinct", Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params={Props.SQL_ID: str(sql_id)},
        parent=parent,
        is_full_parse=is_full_parse,
    )


def create_over_node(
    query_obj: Query,
    section_type: str,
    parent: Node,
    sql_id: str,
    over_what: dict[str, Any],
    is_full_parse: bool = False,
) -> tuple[str, str]:
    label = Labels.COMMAND
    over_node = query_obj.create_current_node(
        name="Over",
        label=label,
        props={"name": "Over", Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params={Props.SQL_ID: str(sql_id)},
        parent=parent,
        is_full_parse=is_full_parse,
    )

    if "WindowSpec" in over_what:
        over_what = over_what["WindowSpec"]

    over_sql_str = ""
    over_sql_parts_str = ""

    if len(over_what["partition_by"]) > 0:
        over_what_str = "partition_by"
    elif len(over_what["order_by"]) > 0:
        over_what_str = "order_by"
    else:
        return "over ()", "over ()"
    for i, over_item in zip(
        range(len(over_what[over_what_str])), over_what[over_what_str]
    ):
        if next(iter(over_item)) == "Tuple":
            first_item = over_item["Tuple"][0]
            _, f_expr_sql_str, f_sql_part, data_type = handle_expression(
                query_obj,
                SQL.OVER,
                first_item,
                over_node,
                sql_id,
                is_full_parse=is_full_parse,
            )

            second_item = over_item["Tuple"][1]
            _, s_expr_sql_str, s_sql_part, data_type = handle_expression(
                query_obj,
                SQL.OVER,
                second_item,
                over_node,
                sql_id,
                is_full_parse=is_full_parse,
            )

            over_sql_str = f"{over_sql_str}({f_expr_sql_str}, {s_expr_sql_str}), "
            over_sql_parts_str = f"{over_sql_parts_str}({f_sql_part}, {s_sql_part}), "
        else:
            _, expr_sql_str, sql_part, data_type = handle_expression(
                query_obj,
                SQL.OVER,
                over_item,
                over_node,
                sql_id,
                is_full_parse=is_full_parse,
            )
            over_sql_str = f"{over_sql_str}{expr_sql_str}, "
            over_sql_parts_str = f"{over_sql_parts_str}{sql_part}, "
    over_sql_str = over_sql_str[:-2]  # .replace("\'", "")
    over_sql_parts_str = over_sql_parts_str[:-2]

    pointer_str = f"over ({over_what_str.replace('_', ' ')} {over_sql_str})"  # order_by to order by
    pointer_sql_parts_str = (
        f"over ({over_what_str.replace('_', ' ')} {over_sql_parts_str})"
    )

    handle_node_update(
        label=label,
        node=over_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": pointer_str, "sql_part": pointer_sql_parts_str},
    )

    return pointer_str, pointer_sql_parts_str


def create_case_node(
    query_obj: Query,
    section_type: str,
    case_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    label = Labels.OPERATOR
    case_node = query_obj.create_current_node(
        name="Case",
        label=label,
        props={"name": "Case", Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    conditions_list = case_dict["conditions"]
    results_list = case_dict["results"]
    else_result = case_dict["else_result"]

    case_sql_str = "case"
    case_sql_parts_str = "case"
    data_types = []
    if "Value" in conditions_list[0].keys():
        column_node, column_str, sql_part, value_data_type = handle_expression(
            query_obj,
            section_type,
            case_dict["operand"],
            case_node,
            sql_id,
            is_full_parse=is_full_parse,
        )
        case_sql_str = f"{case_sql_str} {column_str}"
        case_sql_parts_str = f"{case_sql_parts_str} {sql_part}"
        data_types.extend(value_data_type)
    for i in range(len(conditions_list)):
        condition_node, condition_sql_str, con_sql_part, _ = handle_expression(
            query_obj,
            section_type,
            conditions_list[i],
            case_node,
            sql_id,
            is_full_parse=is_full_parse,
        )
        result_node, result_sql_str, res_sql_part, result_data_type = handle_expression(
            query_obj,
            section_type,
            results_list[i],
            case_node,
            sql_id,
            is_full_parse=is_full_parse,
        )
        case_sql_str = f"{case_sql_str} when {condition_sql_str} then {result_sql_str}"
        case_sql_parts_str = (
            f"{case_sql_parts_str} when {con_sql_part} then {res_sql_part}"
        )
        data_types.extend(result_data_type)
    if else_result is not None:
        _, else_sql_str, else_sql_part, data_type = handle_expression(
            query_obj,
            section_type,
            else_result,
            case_node,
            sql_id,
            is_full_parse=is_full_parse,
        )
        case_sql_str = f"{case_sql_str} else {else_sql_str}"
        case_sql_parts_str = f"{case_sql_parts_str} else {else_sql_part}"
        data_types.extend(result_data_type)
    pointer_str = f"{case_sql_str} end"
    pointer_sql_part_str = f"{case_sql_parts_str} end"
    data_type = prepare_data_types(data_types)
    handle_node_update(
        label=label,
        node=case_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": pointer_str,
            "sql_part": pointer_sql_part_str,
            "data_type": data_type,
        },
    )
    return case_node, pointer_str, pointer_sql_part_str, data_type


def create_inlist_node(
    query_obj: Query,
    section_type: str,
    inlist_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    negated = inlist_dict["negated"]
    if negated:
        operator = "Not In"
    else:
        operator = "In"
    label = Labels.OPERATOR
    inlist_node = query_obj.create_current_node(
        name=operator,
        label=label,
        props={"name": operator, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    _, expr_sql_str, exp_sql_part, data_type = handle_expression(
        query_obj,
        section_type,
        inlist_dict["expr"],
        inlist_node,
        sql_id,
        is_full_parse=is_full_parse,
    )
    the_list = inlist_dict["list"]
    list_exprs = []
    list_sql_parts_exprs = []
    for expr in the_list:
        _, i_expr_str, sql_part, data_type = handle_expression(
            query_obj,
            section_type,
            expr,
            inlist_node,
            sql_id,
            is_full_parse=is_full_parse,
        )
        list_exprs.append(i_expr_str)
        list_sql_parts_exprs.append(sql_part)
    helper_expr_str = f"({', '.join(list_exprs)})"
    helper_sql_parts = f"({', '.join(list_sql_parts_exprs)})"
    final_expr_str = f"{expr_sql_str} {operator} {helper_expr_str}"
    final_sql_parts = f"{exp_sql_part} {operator} {helper_sql_parts}"
    handle_node_update(
        label=label,
        node=inlist_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": final_expr_str, "sql_part": final_sql_parts},
    )
    data_type = ["boolean"]
    return inlist_node, final_expr_str, final_sql_parts, data_type


def create_between_node(
    query_obj: Query,
    section_type: str,
    between_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    negated = between_dict["negated"]
    if negated:
        operator = "NotBetween"
    else:
        operator = "Between"
    label = Labels.OPERATOR
    between_node = query_obj.create_current_node(
        name=operator,
        label=label,
        props={"name": operator, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    _, low_sql_str, low_sql_part, data_type = handle_expression(
        query_obj=query_obj,
        section_type=section_type,
        expr_dict=between_dict["low"],
        parent=between_node,
        sql_id=sql_id,
        optional_edge_params={"child_idx": 0},
        is_full_parse=is_full_parse,
    )
    _, expr_sql_str, expr_sql_part, data_type = handle_expression(
        query_obj=query_obj,
        section_type=section_type,
        expr_dict=between_dict["expr"],
        parent=between_node,
        sql_id=sql_id,
        optional_edge_params={"child_idx": 1},
        is_full_parse=is_full_parse,
    )
    _, high_sql_str, high_sql_part, data_type = handle_expression(
        query_obj=query_obj,
        section_type=section_type,
        expr_dict=between_dict["high"],
        parent=between_node,
        sql_id=sql_id,
        optional_edge_params={"child_idx": 2},
        is_full_parse=is_full_parse,
    )

    expr_sql_str = f"{expr_sql_str} between {low_sql_str} and {high_sql_str}"
    sql_parts = f"{expr_sql_part} between {low_sql_part} and {high_sql_part}"
    handle_node_update(
        label=label,
        node=between_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": expr_sql_str, "sql_part": sql_parts},
    )
    data_type = ["boolean"]
    return between_node, expr_sql_str, sql_parts, data_type


def create_function_node(
    query_obj: Query,
    section_type: str,
    function_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    """
    "name"
    "args"
    "over"
    "distinct"
    """
    function_name = function_dict["name"][0]["value"].lower()
    label = Labels.FUNCTION
    function_node = query_obj.create_current_node(
        name=function_name,
        label=label,
        props={"name": function_name, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )
    args_sql_str = ""
    args_sql_parts_str = ""
    over_sql_str = ""
    over_sql_parts_str = ""
    arg_data_type = []
    if isinstance(function_dict["args"], list):
        args_list = function_dict["args"]
    elif function_dict["args"] == "None":
        args_list = []
    else:
        args_list = function_dict["args"]["List"]["args"]
        if function_dict["args"]["List"]["duplicate_treatment"] == "Distinct":
            create_distinct_node(
                query_obj, section_type, function_node, sql_id, is_full_parse
            )
            args_sql_str = "distinct " + args_sql_str
            args_sql_parts_str = "distinct" + args_sql_parts_str

    for i, arg in enumerate(args_list):
        arg_type = next(iter(arg))
        if arg_type == "Unnamed":
            ## what about qualified wildcards?
            if arg[arg_type] == Parser.WILDCARD:
                connect_wildcard_to_query_tables(
                    query_obj,
                    parent,
                    sql_id,
                    section_type,
                    agg_function=True,
                    is_full_parse=is_full_parse,
                )
                args_sql_str = f"{args_sql_str}*, "
                arg_data_type = list(
                    set(
                        reduce(
                            lambda data_types, current: data_types
                            + prepare_data_types(
                                current.get_properties().get("data_type")
                            ),
                            query_obj.get_projection_nodes(),
                            [],
                        )
                    )
                )
            elif arg[arg_type] == Parser.Q_WILDCARD:
                table_alias = arg[arg_type][0]["value"]
                table_node = query_obj.get_table_node_from_schema(table_alias)
                connect_wildcard_to_query_tables(
                    query_obj,
                    parent,
                    sql_id,
                    section_type,
                    agg_function=True,
                    is_full_parse=is_full_parse,
                    table_node=table_node,
                )
                args_sql_str = f"{args_sql_str}*, "
                arg_data_type = list(
                    set(
                        reduce(
                            lambda data_types, current: data_types
                            + prepare_data_types(
                                current.get_properties().get("data_type")
                            ),
                            query_obj.get_projection_nodes(),
                            [],
                        )
                    )
                )
            elif (
                function_name in SQLFunctionsWithConsantArg.functions
                and next(iter(arg[arg_type]["Expr"])) == "Identifier"
            ):
                arg_dict = arg[arg_type]
                if (
                    arg[arg_type]["Expr"]["Identifier"]["value"].lower()
                    in ArgsForSQLFunctions.args
                ):
                    arg_dict = {Parser.VALUE: arg[arg_type]["Expr"][Parser.IDENTIFIER]}
                (
                    _,
                    expr_sql_str,
                    sql_part,
                    arg_data_type,
                ) = handle_expression(
                    query_obj=query_obj,
                    section_type=section_type,
                    expr_dict=arg_dict,
                    parent=function_node,
                    sql_id=sql_id,
                    optional_edge_params={"child_idx": i},
                    is_full_parse=is_full_parse,
                )
                args_sql_str = f"{args_sql_str}{expr_sql_str}, "
                args_sql_parts_str = f"{args_sql_parts_str}{sql_part}, "
            elif (function_name == "date_format") and next(
                iter(arg[arg_type]["Expr"])
            ) == "AtTimeZone":
                arg_dict = arg[arg_type]["Expr"]["AtTimeZone"]["timestamp"]
                (
                    _,
                    expr_sql_str,
                    sql_part,
                    arg_data_type,
                ) = handle_expression(
                    query_obj=query_obj,
                    section_type=section_type,
                    expr_dict=arg_dict,
                    parent=function_node,
                    sql_id=sql_id,
                    optional_edge_params={"child_idx": i},
                    is_full_parse=is_full_parse,
                )
                timezone = arg[arg_type]["Expr"]["AtTimeZone"]["time_zone"]
                args_sql_str = (
                    f"{args_sql_str}{expr_sql_str} AT TIME ZONE '{timezone}', "
                )
                args_sql_parts_str = (
                    f"{args_sql_parts_str}{sql_part} AT TIME ZONE '{timezone}', "
                )
            else:
                (
                    _,
                    expr_sql_str,
                    sql_part,
                    arg_data_type,
                ) = handle_expression(
                    query_obj=query_obj,
                    section_type=section_type,
                    expr_dict=arg[arg_type],
                    parent=function_node,
                    sql_id=sql_id,
                    optional_edge_params={"child_idx": i},
                    is_full_parse=is_full_parse,
                )
                if (
                    expr_sql_str.strip().find("(") == 0
                    and expr_sql_str.strip().rfind(")") == len(expr_sql_str.strip()) - 1
                ):
                    expr_sql_str = expr_sql_str[1:-1]
                    sql_part = sql_part[1:-1]
                args_sql_str = f"{args_sql_str}{expr_sql_str}, "
                args_sql_parts_str = f"{args_sql_parts_str}{sql_part}, "
        else:
            raise Exception(
                f'Unknown function arg type: {arg_type} in "create_function_node". function_dict:\n {str(function_dict)}'
            )
    args_sql_str = args_sql_str[:-2]
    args_sql_parts_str = args_sql_parts_str[:-2]

    if (
        SQL.DISTINCT in function_dict
        and function_dict[SQL.DISTINCT]
        and len(args_sql_str) > 0
    ):  # cannot be on empty arguments, add distinct
        create_distinct_node(
            query_obj, section_type, function_node, sql_id, is_full_parse
        )
        args_sql_str = "distinct " + args_sql_str
        args_sql_parts_str = "distinct" + args_sql_parts_str

    if (
        SQL.OVER in function_dict and function_dict[SQL.OVER]
    ):  # can be on empty arguments
        over_sql_str, over_sql_parts_str = create_over_node(
            query_obj,
            section_type,
            function_node,
            sql_id,
            function_dict[SQL.OVER],
            is_full_parse,
        )
        # args_sql_str = f'{args_sql_str}'

    if len(args_sql_str) == 0:
        func_sql_str = f"{function_name}()"
        func_sql_parts_str = f"{function_name}()"
    else:
        if (
            args_sql_str.strip().find("(") == 0
            and args_sql_str.strip().rfind(")") == len(args_sql_str.strip()) - 1
        ):
            func_sql_str = f"{function_name}{args_sql_str}"
            func_sql_parts_str = f"{function_name}{args_sql_parts_str}"

        else:
            func_sql_str = f"{function_name}({args_sql_str})"
            func_sql_parts_str = f"{function_name}({args_sql_parts_str})"

    if len(over_sql_str) > 0:
        func_sql_str = f"{func_sql_str} {over_sql_str}"
        func_sql_parts_str = f"{func_sql_parts_str} {over_sql_parts_str}"

    data_type = change_to_output_dtype(function_name, arg_data_type)

    handle_node_update(
        label=label,
        node=function_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": func_sql_str,
            "sql_part": func_sql_parts_str,
            "data_type": data_type,
        },
    )
    return function_node, func_sql_str, func_sql_parts_str, data_type


def create_arrayagg_node(
    query_obj: Query,
    section_type: str,
    arrayagg_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
) -> tuple[Node, str, str, list[str]]:
    expr_node, expr_str, expr_sql_str, data_type = handle_expression(
        query_obj,
        section_type,
        arrayagg_dict["expr"],
        parent,
        sql_id,
        optional_edge_params,
    )
    # distinct_str = 'distinct' if arrayagg_dict['distinct'] else ''
    # expr_sql_str = f"array_agg({distinct_str} {expr_str})"
    # expr_sql_part = f"array_agg({distinct_str} {expr_sql_str})"
    return expr_node, expr_str, expr_sql_str, data_type


def create_array_node(
    query_obj: Query,
    section_type: str,
    array_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    edge_params = create_edge_params(optional_edge_params, sql_id)
    label = Labels.OPERATOR
    array_node = query_obj.create_current_node(
        name=Parser.ARRAY,
        label=label,
        props={"name": Parser.ARRAY, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=edge_params,
        parent=parent,
        is_full_parse=is_full_parse,
    )

    expr_str_list, expr_sql_str_list, data_type_list = [], [], []
    for i, elem in enumerate(array_dict["elem"]):
        expr_node, expr_str, expr_sql_str, data_type = handle_expression(
            query_obj,
            section_type,
            elem,
            array_node,
            sql_id,
            {"child_idx": 0},
            is_full_parse,
        )
        expr_str_list.append(expr_str)
        expr_sql_str_list.append(expr_sql_str)
        data_type_list.extend(data_type)
    expr_sql_str = f"[{', '.join(expr_str_list)}]"
    data_type_list = prepare_data_types(data_type_list)
    handle_node_update(
        label=label,
        node=array_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": expr_sql_str,
            "sql_part": expr_sql_str,
            "data_type": data_type_list,
        },
    )
    return array_node, expr_str, expr_sql_str, data_type_list


def create_arrayindex_node(
    query_obj: Query,
    section_type: str,
    arrayindex_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    if "Nested" in arrayindex_dict["obj"]:
        function_dict = arrayindex_dict["obj"]["Nested"]
    else:
        function_dict = arrayindex_dict["obj"]
    func_node, func_str, func_sql_part, data_type = handle_expression(
        query_obj,
        section_type,
        function_dict,
        parent,
        sql_id,
        {"child_idx": 0},
        is_full_parse,
    )
    index = arrayindex_dict["indexes"][0]["Value"]["Number"][0]
    expr_sql_str = f"{func_str}[{index}]"
    expr_sql_part = f"{func_sql_part}[{index}]"
    return func_node, expr_sql_str, expr_sql_part, data_type


def create_interval_node(
    query_obj: Query,
    section_type: str,
    interval_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    edge_params = create_edge_params(optional_edge_params, sql_id)
    label = Labels.OPERATOR
    interval_node = query_obj.create_current_node(
        name=Parser.INTERVAL,
        label=label,
        props={"name": Parser.INTERVAL, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=edge_params,
        parent=parent,
        is_full_parse=is_full_parse,
    )

    expr_node, expr_str, exp_sql_part, _ = handle_expression(
        query_obj,
        section_type,
        interval_dict["value"],
        interval_node,
        sql_id,
        {"child_idx": 0},
        is_full_parse,
    )

    _ = query_obj.create_current_node(
        name=interval_dict["leading_field"],
        label=Labels.CONSTANT,
        props={"name": interval_dict["leading_field"], Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(None, sql_id),
        parent=interval_node,
        is_full_parse=is_full_parse,
    )

    if isinstance(expr_str, str):
        expr_sql_str = f"interval {expr_str} {interval_dict['leading_field']}"
        exp_sql_part = f"interval {exp_sql_part} {interval_dict['leading_field']}"
    else:
        expr_sql_str = f"interval '{expr_str}' {interval_dict['leading_field']}"
        exp_sql_part = f"interval '{exp_sql_part}' {interval_dict['leading_field']}"
    data_type = ["datetime"]
    handle_node_update(
        label=label,
        node=interval_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": expr_sql_str,
            "data_type": data_type,
            "sql_part": exp_sql_part,
        },
    )
    return interval_node, expr_sql_str, exp_sql_part, data_type


def create_like_node(
    query_obj: Query,
    section_type: str,
    like_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    negated = like_dict["negated"]
    if negated:
        operator = "Not Like"
    else:
        operator = "Like"
    label = Labels.OPERATOR
    like_node = query_obj.create_current_node(
        name=operator,
        label=label,
        props={"name": operator, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    expr_node, expr_str, exp_sql_part, data_type = handle_expression(
        query_obj, section_type, like_dict["expr"], like_node, sql_id, {"child_idx": 0}
    )
    _, pattern_str, pattern_sql_part, data_type = handle_expression(
        query_obj,
        section_type,
        like_dict["pattern"],
        like_node,
        sql_id,
        {"child_idx": 1},
        is_full_parse,
    )

    expr_sql_str = f"{expr_str} {operator} {pattern_str}"
    exp_sql_part = f"{exp_sql_part} {operator} {pattern_sql_part}"
    data_type = ["boolean"]

    handle_node_update(
        label=label,
        node=like_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": expr_sql_str,
            "data_type": data_type,
            "sql_part": exp_sql_part,
        },
    )
    return like_node, expr_sql_str, exp_sql_part, data_type


def create_unary_op_node(
    query_obj: Query,
    section_type: str,
    unaryop_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    label = Labels.OPERATOR
    unaryop_node = query_obj.create_current_node(
        name=unaryop_dict["op"],
        label=label,
        props={"name": unaryop_dict["op"], Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    child_dict = unaryop_dict["expr"]
    child_node, expr_sql_str, sql_part, data_type = handle_expression(
        query_obj, section_type, child_dict, unaryop_node, sql_id, None, is_full_parse
    )
    handle_node_update(
        label=label,
        node=unaryop_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": expr_sql_str,
            "sql_part": sql_part,
            "data_type": data_type,
        },
    )
    return unaryop_node, expr_sql_str, sql_part, data_type


def create_any_op_node(
    query_obj: Query,
    section_type: str,
    anyop_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    label = Labels.OPERATOR
    anyop_node = query_obj.create_current_node(
        name=Parser.ANY_OP,
        label=label,
        props={"name": Parser.ANY_OP, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )
    if "compare_op" in anyop_dict:
        binaryop_dict = {
            Parser.BINARY_OP: {
                "left": anyop_dict["left"],
                "op": anyop_dict["compare_op"],
                "right": anyop_dict["right"],
            }
        }
        _, expr_sql_str, sql_part, data_type = handle_expression(
            query_obj,
            section_type,
            binaryop_dict,
            anyop_node,
            sql_id,
            None,
            is_full_parse,
        )
    else:
        _, expr_sql_str, sql_part, data_type = handle_expression(
            query_obj, section_type, anyop_dict, anyop_node, sql_id, None, is_full_parse
        )
    data_type = ["boolean"]
    handle_node_update(
        label=label,
        node=anyop_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": expr_sql_str,
            "sql_part": sql_part,
            "data_type": data_type,
        },
    )
    return anyop_node, expr_sql_str, sql_part, data_type


def create_binary_op_node(
    query_obj: Query,
    section_type: str,
    binaryop_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    label = Labels.OPERATOR
    binaryop_node = query_obj.create_current_node(
        name=binaryop_dict["op"],
        label=label,
        props={"name": binaryop_dict["op"], Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )
    child_idx_left = None
    child_idx_right = None
    if binaryop_dict["op"] not in ["Eq", "NotEq", "And", "Or"]:
        child_idx_left = {"child_idx": 0}
        child_idx_right = {"child_idx": 1}

    left_dict = binaryop_dict["left"]
    left_node, left_expr_str, left_sql_part, left_data_type = handle_expression(
        query_obj=query_obj,
        section_type=section_type,
        expr_dict=left_dict,
        parent=binaryop_node,
        sql_id=sql_id,
        optional_edge_params=child_idx_left,
        is_full_parse=is_full_parse,
    )
    right_dict = binaryop_dict["right"]
    right_node, right_expr_str, right_sql_part, right_data_type = handle_expression(
        query_obj=query_obj,
        section_type=section_type,
        expr_dict=right_dict,
        parent=binaryop_node,
        sql_id=sql_id,
        optional_edge_params=child_idx_right,
        is_full_parse=is_full_parse,
    )

    # if the binary op is "Eq" and its origin is the join operation then connect the tables with a join edge
    if binaryop_dict["op"] == "Eq" and parent.get_name() in [
        "Inner",
        "LeftOuter",
        "RightOuter",
        "Outer",
        "Joins",
    ]:
        if left_node.get_label() in [
            Labels.COLUMN,
            Labels.TEMP_COLUMN,
        ] and right_node.get_label() in [Labels.COLUMN, Labels.TEMP_COLUMN]:
            left_table_name = left_node.get_match_props()["table_name"]
            if len(left_table_name) == 0 or left_table_name == '""':
                left_node = query_obj.find_column_node(left_node.get_name())[0]
                left_table_name = left_node.get_match_props()["table_name"]

            right_table_name = right_node.get_match_props()["table_name"]
            if len(right_table_name) == 0 or right_table_name == '""':
                right_node = query_obj.find_column_node(right_node.get_name())[0]
                right_table_name = right_node.get_match_props()["table_name"]
            ## We sort the tables to ensure consistent join operators
            if left_table_name > right_table_name:
                left_node, right_node = right_node, left_node
                left_expr_str, right_expr_str = right_expr_str, left_expr_str
                left_sql_part, right_sql_part = right_sql_part, left_sql_part
                left_data_type, right_data_type = right_data_type, left_data_type

    symbol, data_type = get_symbol(binaryop_dict["op"])
    if symbol in SQL.SET_OP_OPERATIONS:
        data_type = prepare_data_types(left_data_type + right_data_type)
    else:
        data_type = prepare_data_types(data_type)
    expr_sql_str = f"{left_expr_str} {symbol} {right_expr_str}"
    sql_part = f"{left_sql_part} {symbol} {right_sql_part}"
    if symbol in ["+", "-", "*", "/"]:
        expr_sql_str = "( " + expr_sql_str + " )"
    handle_node_update(
        label=label,
        node=binaryop_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": expr_sql_str,
            "sql_part": sql_part,
            "data_type": data_type,
        },
    )
    return binaryop_node, expr_sql_str, sql_part, data_type


def create_column_node(
    query_obj: Query,
    section_type: str,
    column_dict: Union[str, dict[str, Any]],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    if column_dict == Parser.WILDCARD:
        connect_wildcard_to_query_tables(
            query_obj, parent, sql_id, section_type, is_full_parse=is_full_parse
        )
        identifier_sql_str = "*"
    else:
        identifier_key = next(iter(column_dict))
        identifier_val = column_dict[identifier_key]
        schema_name = ""
        if identifier_key == "CompoundIdentifier":
            if (
                len(identifier_val) == 2
            ):  # CompoundIdentifier does not include schema_name
                tbl_idx = 0
                col_idx = 1
            else:  # CompoundIdentifier includes: schema_name, table_name, column_name
                schema_name = identifier_val[0]["value"]
                tbl_idx = 1
                col_idx = 2
            table_name = identifier_val[tbl_idx]["value"]
            if table_name.lower() in query_obj.get_tables_aliases():
                table_node = query_obj.get_tables_aliases()[table_name.lower()]
            else:
                table_node = query_obj.get_table_node_from_schema(
                    table_name, schema_name
                )
            col_name = identifier_val[col_idx]["value"]
            col_node = query_obj.get_column_node(col_name, table_node, section_type)
        elif identifier_key == "Identifier":
            col_name = identifier_val["value"]
            # CURRENT_USER is a function in snowflake
            if col_name.lower() in [
                "CURRENT_USER".lower(),
                "SYSDATE".lower(),
                "ROWNUM".lower(),
            ]:
                return create_value_node(
                    query_obj=query_obj,
                    section_type=section_type,
                    value_dict=identifier_val,
                    parent=parent,
                    sql_id=sql_id,
                    optional_edge_params=optional_edge_params,
                    is_full_parse=is_full_parse,
                )
            else:
                col_node = query_obj.find_column_node(col_name)
        else:
            raise Exception(
                f'Unknown identifier type: {identifier_key} in "create_column_node". column_dict:\n {str(column_dict)}'
            )
        if isinstance(col_node, Node):
            current_node = col_node
            data_type = prepare_data_types(col_node.get_properties().get("data_type"))
            label_of_current_node = current_node.label
            edge_params = create_edge_params(optional_edge_params, sql_id)
            query_obj.add_edge((parent, current_node, edge_params), section_type)
            query_obj.increment_nodes_counter()
        elif len(col_node) == 1:
            current_node = col_node[0]
            data_type = prepare_data_types(
                current_node.get_properties().get("data_type")
            )
            label_of_current_node = current_node.label
            edge_params = create_edge_params(optional_edge_params, sql_id)
            query_obj.add_edge((parent, current_node, edge_params), section_type)
            query_obj.increment_nodes_counter()
        else:
            label_of_current_node = Labels.SET_OP_COLUMN
            data_type = prepare_data_types(
                col_node[0].get_properties().get("data_type")
            )
            current_node = query_obj.create_current_node(
                name=col_name,
                label=label_of_current_node,
                props={"name": col_name, Props.SQL_ID: str(sql_id)},
                section_type=section_type,
                edge_params=create_edge_params(optional_edge_params, sql_id),
                parent=parent,
                is_full_parse=is_full_parse,
            )
            handle_node_update(
                label=label_of_current_node,
                node=current_node,
                is_full_parse=is_full_parse,
                payload={"data_type": data_type},
            )
            for c in col_node:
                edge_params = create_edge_params(None, sql_id)
                query_obj.add_edge((current_node, c, edge_params), section_type)

        sql_part = ""
        if label_of_current_node in [Labels.COLUMN, Labels.TEMP_COLUMN]:
            sql_part = col_name
            if identifier_key == "CompoundIdentifier":
                if len(schema_name) > 0:
                    sql_part = f"{schema_name}.{table_name}.{col_name}"
                else:
                    sql_part = f"{table_name}.{col_name}"
            match_props = current_node.get_match_props()
            identifier_sql_str = f"{match_props['schema_name']}.{match_props['table_name']}.{match_props['name']}"
        elif label_of_current_node == Labels.SET_OP_COLUMN:
            identifier_sql_str = current_node.get_name()
            sql_part = current_node.get_name()
        elif label_of_current_node == Labels.ALIAS:
            # The aliased_expr holds the true computed expression that is projected from "downstream" the SQL.
            # We keep the possibility to return the computed expression instead of the name of the alias:
            alias_props = current_node.props
            identifier_sql_str = alias_props["aliased_expr"]
            sql_part = alias_props["sql_part"]
            # identifier_sql_str = col_node.get_name()
        elif label_of_current_node == Labels.FUNCTION:
            identifier_sql_str = (
                current_node.get_properties().get("expr_str").lower().replace(" ", "")
            )
            sql_part = (
                current_node.get_properties().get("sql_part").lower().replace(" ", "")
            )  # ??
        else:
            raise Exception(
                f'Unknown identifier label: {current_node.get_label()} in "create_column_node". column_dict:\n {str(column_dict)}'
            )
    return current_node, identifier_sql_str, sql_part, data_type


def create_extract_node(
    query_obj: Query,
    section_type: str,
    extract_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    extracted_field = extract_dict["field"]
    expr_dict = extract_dict["expr"]
    label = Labels.FUNCTION
    extract_node = query_obj.create_current_node(
        name="Extract",
        label=label,
        props={"name": "Extract", Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    _ = query_obj.create_current_node(
        name="Value",
        label=Labels.CONSTANT,
        props={"name": f"Value={extracted_field}", Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=extract_node,
        is_full_parse=is_full_parse,
    )

    _, expr_sql_str, sql_part, _ = handle_expression(
        query_obj, section_type, expr_dict, extract_node, sql_id, None, is_full_parse
    )

    handle_node_update(
        label=label,
        node=extract_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": expr_sql_str, "sql_part": sql_part},
    )
    data_type = ["int"]
    return extract_node, expr_sql_str, sql_part, data_type


def create_listagg_node(
    query_obj: Query,
    section_type: str,
    listagg_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    expr_dict = listagg_dict["expr"]
    label = Labels.FUNCTION
    listagg_node = query_obj.create_current_node(
        name=Parser.LISTAGG,
        label=label,
        props={"name": Parser.LISTAGG, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    _, expr_str, sql_part, _ = handle_expression(
        query_obj, section_type, expr_dict, listagg_node, sql_id, None, is_full_parse
    )
    pointer_str = f"listagg({expr_str})"
    pointer_sql_parts = f"listagg({sql_part})"

    handle_node_update(
        label=label,
        node=listagg_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": pointer_str, "sql_part": pointer_sql_parts},
    )
    data_type = ["string"]
    return listagg_node, pointer_str, pointer_sql_parts, data_type


def create_exists_node(
    query_obj: Query,
    section_type: str,
    exists_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    label = Labels.OPERATOR
    exists_node = query_obj.create_current_node(
        name=Parser.EXISTS,
        label=label,
        props={"name": Parser.EXISTS, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    sql_id_increased, sql_node, subquery_obj = create_sub_sql_node(
        query_obj=query_obj,
        section_type=section_type,
        optional_edge_params=optional_edge_params,
        parent=exists_node,
        sql_id=sql_id,
        is_sub_select=True,
        is_full_parse=is_full_parse,
    )
    previous_with_subselects = query_obj.get_subselects_by_section(SQL.WITH)
    for previous_with in previous_with_subselects:
        subquery_obj.add_subselect(previous_with, previous_with.get_id(), SQL.WITH)
        subquery_obj.add_alias_to_table(previous_with.get_name(), previous_with)
    query_dict = exists_dict
    if "subquery" in exists_dict:
        query_dict = exists_dict["subquery"]
    expr_sql_str, sql_part_str, _ = create_query(
        subquery_obj,
        query_dict,
        parent=sql_node,
        sql_id=sql_id_increased,
        is_full_parse=is_full_parse,
    )
    query_obj.add_edges(subquery_obj.get_edges(), Parser.SUBSELECT)

    pointer_str = f"{'' if not exists_dict['negated'] else 'Not '}Exists {expr_sql_str}"
    pointer_str_parts = (
        f"{'' if not exists_dict['negated'] else 'Not '}Exists {sql_part_str}"
    )
    data_type = ["boolean"]
    handle_node_update(
        label=label,
        node=exists_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": pointer_str,
            "sql_part": pointer_str_parts,
            "data_type": data_type,
        },
    )
    return exists_node, pointer_str, pointer_str_parts, data_type


def create_value_node(
    query_obj: Query,
    section_type: str,
    value_dict: Union[int, dict[str, Any]],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    if isinstance(value_dict, int):
        val = value_dict
        val_str = f"Value={val}"
        value_type = "int"
    else:
        value_type = next(iter(value_dict))
        if value_type == "Number":
            val = value_dict[value_type][0]
            # if parent.name.lower() == "minus":
            #     val = "-" + val
        elif value_type in ["SingleQuotedString", "NationalStringLiteral"]:
            val = f"'{value_dict[value_type]}'"  # alternative is f"'{}'"
        elif value_type in ["data_type", "value"]:
            val = value_dict["value"]
        elif value_type == "Boolean":
            val = value_dict[value_type]
        elif value_type == "Interval":
            val = f"Interval '{value_dict[value_type]['value']}' {value_dict[value_type]['leading_field']}"
        elif value_type == "Placeholder":
            val = value_dict[value_type]
        elif value_dict == "Null":
            val = "Null"
        else:
            raise Exception(
                f'Unknown value type: {value_type} in "create_value_node". value_dict:\n {str(value_dict)}'
            )
        if (
            value_dict == "Null"
            or value_type == "Boolean"
            or val.isnumeric()
            or val.lower() == "SYSDATE".lower()
            or val.lower() == "DAY".lower()
        ):
            val_str = f"Value={val}"
        else:
            try:
                # if it is a datetime value then keep the value
                pendulum.parse(val, strict=False)
                val_str = f"Value={val}"
            except (ValueError, TypeError, IndexError, OverflowError):
                val_str = f"Value={val}" if keep_string_values else "Value"
    label = Labels.CONSTANT
    value_node = query_obj.create_current_node(
        name=val_str,
        label=label,
        props={"name": val_str, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    pointer_str = f"{val}"
    data_type = [value_type]
    handle_node_update(
        label=label,
        node=value_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": pointer_str,
            "sql_part": pointer_str,
            "data_type": data_type,
        },
    )
    return value_node, pointer_str, pointer_str, data_type


def create_jsonaccess_node(
    query_obj: Query,
    section_type: str,
    jsonaccess_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    # Todo: Make this prettier
    key = list(jsonaccess_dict["path"].values())[0][0]
    if "Dot" in key:
        key = key["Dot"]["key"]
    column_node, expr_str, sql_part, data_type = handle_expression(
        query_obj,
        section_type,
        jsonaccess_dict["value"],
        parent,
        sql_id,
        optional_edge_params,
        is_full_parse,
    )
    exp_to_return = f"{column_node.get_name()}:{key}"
    return column_node, exp_to_return, exp_to_return, data_type


def handle_cast(
    query_obj: Query,
    section_type: str,
    expr_type: str,
    expr_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
):
    casted_data_types = []
    if expr_type == Parser.ATTIMEZONE:
        expr_dict = expr_dict[expr_type]["timestamp"][expr_type]["timestamp"]
        expr_type = Parser.CAST
    if expr_type in [Parser.CAST, Parser.CONVERT]:
        if isinstance(expr_dict[expr_type]["data_type"], str):
            casted_data_types = [expr_dict[expr_type]["data_type"]]
        elif isinstance(expr_dict[expr_type]["data_type"], dict):
            casted_data_types = list(expr_dict[expr_type]["data_type"].keys())
    expr_dict = expr_dict[expr_type]["expr"]
    if (
        next(iter(expr_dict)) == Parser.IDENTIFIER
        and expr_dict[next(iter(expr_dict))]["value"] == "SYSDATE"
    ):
        expr_dict = {Parser.VALUE: expr_dict[Parser.IDENTIFIER]}
    subgraph_root, expr_sql_str, sql_part, expr_data_type = handle_expression(
        query_obj,
        section_type,
        expr_dict,
        parent,
        sql_id,
        optional_edge_params,
        is_full_parse,
    )
    if expr_type in [Parser.CAST, Parser.CONVERT]:
        expr_sql_str = f"{expr_type}({expr_sql_str} as {casted_data_types[0]})"
        sql_part = f"{expr_type}({sql_part} as {casted_data_types[0]})"
        data_type = casted_data_types
    else:
        expr_sql_str = f"{expr_type}({expr_sql_str})"
        sql_part = f"{expr_type}({sql_part})"
        data_type = expr_data_type

    return subgraph_root, expr_sql_str, sql_part, data_type


def handle_expression(
    query_obj: Query,
    section_type: str,
    expr_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    expr_type = next(iter(expr_dict))
    sql_part = ""
    while expr_type.lower() in [
        "Nested".lower(),
        "Expr".lower(),
        "Substring".lower(),
    ]:  # ignore being nested and ignore substring
        expr_dict = expr_dict[expr_type]
        expr_type = next(iter(expr_dict))

    if len(expr_dict) > 1:
        raise Exception(
            f'Unexpected structure of "handle_expression". expr_dict\n: {expr_dict}'
        )
    expr_sql_str = ""

    if expr_type == Parser.ARRAY:
        subgraph_root, expr_sql_str, sql_part, data_type = create_array_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.ARRAYAGG:
        subgraph_root, expr_sql_str, sql_part, data_type = create_arrayagg_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.ARRAYINDEX:
        subgraph_root, expr_sql_str, sql_part, data_type = create_arrayindex_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.INTERVAL:
        subgraph_root, expr_sql_str, sql_part, data_type = create_interval_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type in [Parser.LIKE, Parser.ILIKE]:
        subgraph_root, expr_sql_str, sql_part, data_type = create_like_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.JSON_ACCESS:
        subgraph_root, expr_sql_str, sql_part, data_type = create_jsonaccess_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.EXISTS:
        subgraph_root, expr_sql_str, sql_part, data_type = create_exists_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.LISTAGG:
        subgraph_root, expr_sql_str, sql_part, data_type = create_listagg_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.EXTRACT:
        subgraph_root, expr_sql_str, sql_part, data_type = create_extract_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )

    elif expr_type in [
        Parser.CAST,
        Parser.CONVERT,
        Parser.TRIM,
        Parser.FLOOR,
        Parser.CEIL,
        Parser.ATTIMEZONE,
    ]:
        subgraph_root, expr_sql_str, sql_part, data_type = handle_cast(
            query_obj,
            section_type,
            expr_type,
            expr_dict,
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.SET_OPERATION:
        sql_id_increased, sql_node, subquery_obj = create_sub_sql_node(
            query_obj,
            section_type,
            optional_edge_params,
            parent,
            sql_id,
            is_sub_select=True,
            is_full_parse=is_full_parse,
        )
        previous_with_subselects = query_obj.get_subselects_by_section(SQL.WITH)
        for previous_with in previous_with_subselects:
            subquery_obj.add_subselect(previous_with, previous_with.get_id(), SQL.WITH)
            subquery_obj.add_alias_to_table(previous_with.get_name(), previous_with)
        expr_node, expr_sql_str, sql_part, data_type = create_set_op_node(
            subquery_obj,
            section_type,
            expr_dict[expr_type],
            sql_node,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
        query_obj.add_edges(subquery_obj.get_edges(), Parser.SUBSELECT)
        subgraph_root = sql_node
        edge_params = create_edge_params(optional_edge_params, sql_id)
        query_obj.add_edge((sql_node, expr_node, edge_params), section_type)
    elif expr_type == Parser.UNARY_OP:
        subgraph_root, expr_sql_str, sql_part, data_type = create_unary_op_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.ANY_OP:
        subgraph_root, expr_sql_str, sql_part, data_type = create_any_op_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.BINARY_OP:
        subgraph_root, expr_sql_str, sql_part, data_type = create_binary_op_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type in [
        Parser.ISNULL,
        Parser.ISNOTNULL,
        Parser.ISTRUE,
        Parser.ISNOTTRUE,
        Parser.ISFALSE,
        Parser.ISNOTFALSE,
    ]:
        subgraph_root, expr_sql_str, sql_part = create_condition_node(
            query_obj=query_obj,
            section_type=section_type,
            expr_dict=expr_dict[expr_type],
            expr_type=expr_type,
            parent=parent,
            sql_id=sql_id,
            optional_edge_params=optional_edge_params,
            is_full_parse=is_full_parse,
        )
        data_type = ["boolean"]
    elif expr_type == SQL.BETWEEN:
        subgraph_root, expr_sql_str, sql_part, data_type = create_between_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.INLIST:
        subgraph_root, expr_sql_str, sql_part, data_type = create_inlist_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == SQL.CASE:
        subgraph_root, expr_sql_str, sql_part, data_type = create_case_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.IDENTIFIER or expr_type == Parser.COMPOUND_IDENTIFIER:
        subgraph_root, expr_sql_str, sql_part, data_type = create_column_node(
            query_obj,
            section_type,
            expr_dict,
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.FUNCTION:
        subgraph_root, expr_sql_str, sql_part, data_type = create_function_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.VALUE or expr_type == Parser.TYPED_STRING:
        subgraph_root, expr_sql_str, sql_part, data_type = create_value_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif expr_type == Parser.IN_SUBQUERY:
        subgraph_root, expr_sql_str, sql_part, data_type = create_insubquery_node(
            query_obj,
            section_type,
            expr_dict[expr_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse,
        )
    elif (
        expr_type.lower() == Parser.SUBQUERY.lower()
        or expr_type.lower() == Parser.QUERY.lower()
    ):
        sql_id_increased, sql_node, subquery_obj = create_sub_sql_node(
            query_obj,
            section_type,
            optional_edge_params,
            parent,
            sql_id,
            is_sub_select=True,
        )
        previous_with_subselects = query_obj.get_subselects_by_section(SQL.WITH)
        for previous_with in previous_with_subselects:
            subquery_obj.add_subselect(previous_with, previous_with.get_id(), SQL.WITH)
            subquery_obj.add_alias_to_table(previous_with.get_name(), previous_with)
        expr_sql_str, sql_part, data_type = create_query(
            subquery_obj, expr_dict[expr_type], parent=sql_node, sql_id=sql_id_increased
        )
        query_obj.add_edges(subquery_obj.get_edges(), Parser.SUBSELECT)
        subgraph_root = sql_node
    elif expr_type.lower() == SQL.SELECT.lower():
        sql_id_increased, sql_node, subquery_obj = create_sub_sql_node(
            query_obj,
            section_type,
            optional_edge_params,
            parent,
            sql_id,
            is_sub_select=True,
            is_full_parse=is_full_parse,
        )
        previous_with_subselects = query_obj.get_subselects_by_section(SQL.WITH)
        for previous_with in previous_with_subselects:
            subquery_obj.add_subselect(previous_with, previous_with.get_id(), SQL.WITH)
            subquery_obj.add_alias_to_table(previous_with.get_name(), previous_with)
        _, expr_sql_str, sql_part, data_type = create_select_node(
            subquery_obj, expr_dict[expr_type], parent=sql_node, sql_id=sql_id_increased
        )
        query_obj.add_edges(subquery_obj.get_edges(), Parser.SUBSELECT)
        subgraph_root = sql_node
    else:
        raise Exception(
            f'Unknown expression type: {expr_type} in "handle_expression". expr_dict:\n {str(expr_dict)}'
        )
    return subgraph_root, expr_sql_str, sql_part, data_type


def create_select_node(
    query_obj: Query,
    select_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    fks: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    label = Labels.COMMAND
    select_node = query_obj.create_current_node(
        name="Select",
        label=label,
        props={"name": "Select", Props.SQL_ID: str(sql_id)},
        section_type=SQL.SELECT,
        edge_params={Props.SQL_ID: str(sql_id)},
        parent=parent,
        is_full_parse=is_full_parse,
    )

    from_sql_str, from_sql_parts_str = create_from_node(
        query_obj, select_dict[SQL.FROM], select_node, sql_id, fks, is_full_parse
    )
    projected_data_types, proj_sql_str, proj_sql_parts = create_projection_nodes(
        query_obj, select_dict[Parser.PROJECTION], select_node, sql_id, is_full_parse
    )
    where_sql_str, where_sql_parts = create_where_node(
        query_obj, select_dict[Parser.SELECTION], select_node, sql_id, is_full_parse
    )
    group_by_sql_str, group_by_sql_parts_str = create_group_by_node(
        query_obj,
        select_dict[SQL.GROUP_BY],
        select_dict[SQL.HAVING],
        select_node,
        sql_id,
        is_full_parse,
    )
    distinct_str = ""
    if select_dict[SQL.DISTINCT]:
        create_distinct_node(
            query_obj, SQL.DISTINCT, select_node, sql_id, is_full_parse
        )
        distinct_str = "distinct"
    top_sql_str, top_sql_parts_str = create_top_node(
        query_obj, select_dict[SQL.TOP], select_node, sql_id, is_full_parse
    )

    query_obj.add_sql_strs_to_node(proj_sql_parts, from_sql_parts_str, where_sql_parts)
    pointer_str = " ".join(
        f"select {top_sql_str} {distinct_str} {proj_sql_str} from {from_sql_str} {where_sql_str} {group_by_sql_str}".split()
    )
    poniter_sql_parts_str = " ".join(
        f"select {top_sql_parts_str} {distinct_str} {proj_sql_parts} from {from_sql_parts_str} {where_sql_parts} {group_by_sql_parts_str}".split()
    )
    data_type = {}
    if projected_data_types:
        data_type = {"data_type": projected_data_types}
    handle_node_update(
        label=label,
        node=select_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": pointer_str,
            "sql_part": poniter_sql_parts_str,
            **data_type,
        },
    )
    return select_node, pointer_str, poniter_sql_parts_str, projected_data_types


def create_condition_node(
    query_obj: Query,
    section_type: str,
    expr_dict: dict[str, Any],
    expr_type: str,
    parent: Node,
    sql_id: str,
    is_full_parse: bool = False,
    optional_edge_params: Optional[dict[str, Any]] = None,
) -> tuple[Node, str, str, list[str]]:
    label = Labels.OPERATOR
    condition_node = query_obj.create_current_node(
        name=expr_type,
        label=label,
        props={"name": expr_type, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )
    _, expr_sql_str, sql_part, data_type = handle_expression(
        query_obj,
        section_type,
        expr_dict,
        condition_node,
        sql_id,
        optional_edge_params,
        is_full_parse,
    )
    condition_to_str_map = {
        Parser.ISNULL: "is null",
        Parser.ISNOTNULL: "is not null",
        Parser.ISTRUE: "is true",
        Parser.ISNOTTRUE: "is not true",
        Parser.ISFALSE: "is false",
        Parser.ISNOTFALSE: "is not false",
    }
    expr_sql_str = f"{expr_sql_str} {condition_to_str_map[expr_type]}"
    sql_part = f"{sql_part} {condition_to_str_map[expr_type]}"

    handle_node_update(
        label=label,
        node=condition_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": expr_sql_str, "sql_part": sql_part},
    )
    return condition_node, expr_sql_str, sql_part


def create_group_by_node(
    query_obj: Query,
    group_by_list: list[dict[str, Any]],
    having_dict: Optional[dict[str, Any]],
    parent: Node,
    sql_id: str,
    is_full_parse: bool = False,
) -> tuple[str, str]:
    if len(group_by_list) == 0:
        return "", ""

    if "Expressions" in group_by_list:
        group_by_list = group_by_list["Expressions"]
        if len(group_by_list) == 0:
            return "", ""
        elif (
            len(group_by_list) == 2
            and len(group_by_list[0]) == 0
            and len(group_by_list[1]) == 0
        ):
            return "", ""
        elif (
            len(group_by_list) == 2
            and len(group_by_list[0]) > 0
            and len(group_by_list[1]) == 0
        ):
            group_by_list = group_by_list[0]

    if group_by_list == "All":
        return "group by all", "group by all"
    if isinstance(group_by_list, dict) and "All" in group_by_list:
        return "group by all", "group by all"
    label = Labels.COMMAND
    group_by_node = query_obj.create_current_node(
        name="Group_by",
        label=label,
        props={"name": "Group_by", Props.SQL_ID: str(sql_id)},
        section_type=SQL.GROUP_BY,
        edge_params={Props.SQL_ID: str(sql_id)},
        parent=parent,
        is_full_parse=is_full_parse,
    )

    group_by_sql_str = ""
    group_by_sql_parts = ""
    for i, group_by_item in zip(range(len(group_by_list)), group_by_list):
        _, expr_sql_str, sql_part, data_type = handle_expression(
            query_obj,
            SQL.GROUP_BY,
            group_by_item,
            group_by_node,
            sql_id,
            None,
            is_full_parse,
        )
        group_by_sql_str = f"{group_by_sql_str}{expr_sql_str}, "
        group_by_sql_parts = f"{group_by_sql_parts}{sql_part}"
    group_by_sql_str = group_by_sql_str[:-2].replace("'", "")
    group_by_sql_parts = group_by_sql_parts[:-2].replace("'", "")

    having_sql_str, having_sql_parts_str = create_having_node(
        query_obj, SQL.GROUP_BY, having_dict, group_by_node, sql_id, is_full_parse
    )
    pointer_str = f"group by {group_by_sql_str}"
    pointer_str_sql_parts = f"group by {group_by_sql_parts}"
    if (
        having_sql_str is not None and having_sql_str.strip() != ""
    ):  # having parts is therefore also not null
        pointer_str += f"having {having_sql_str}"
        pointer_str_sql_parts += f"having {having_sql_parts_str}"

    handle_node_update(
        label=label,
        node=group_by_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": pointer_str, "sql_part": pointer_str_sql_parts},
    )
    return pointer_str, pointer_str_sql_parts


def create_having_node(
    query_obj: Query,
    section_type: str,
    having_dict: Optional[dict[str, Any]],
    parent: Node,
    sql_id: str,
    is_full_parse: bool = False,
) -> tuple[str, str]:
    if having_dict is None:
        return "", ""
    expr_type = next(iter(having_dict))
    while expr_type in ["Nested", "Expr"]:  # ignore being nested
        having_dict = having_dict[expr_type]
        expr_type = next(iter(having_dict))
    label = Labels.COMMAND
    having_node = query_obj.create_current_node(
        name="Having",
        label=label,
        props={"name": "Having", Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params={Props.SQL_ID: str(sql_id)},
        parent=parent,
        is_full_parse=is_full_parse,
    )

    having_sql_str = ""
    for key, value in having_dict.items():
        if key == "BinaryOp":
            _, having_sql_str, having_sql_parts_str, _ = create_binary_op_node(
                query_obj=query_obj,
                section_type=section_type,
                binaryop_dict=value,
                parent=having_node,
                sql_id=sql_id,
                is_full_parse=is_full_parse,
            )
        else:
            _, having_sql_str, having_sql_parts_str, _ = handle_expression(
                query_obj=query_obj,
                section_type=section_type,
                expr_dict=having_dict,
                parent=having_node,
                sql_id=sql_id,
                is_full_parse=is_full_parse,
            )
            # raise Exception(f"Unknown key type: {key} in \"create_having_node\". having_dict:\n {str(having_dict)}")
    handle_node_update(
        label=label,
        node=having_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": having_sql_str, "sql_part": having_sql_parts_str},
    )
    return having_sql_str, having_sql_parts_str


def create_where_node(
    query_obj: Query,
    where_dict: Optional[dict[str, Any]],
    parent: Node,
    sql_id: str,
    is_full_parse: bool = False,
) -> tuple[str, str]:
    if where_dict is None:
        return "", ""
    label = Labels.COMMAND
    where_node = query_obj.create_current_node(
        name="Where",
        label=label,
        props={"name": "Where", Props.SQL_ID: str(sql_id)},
        section_type=SQL.WHERE,
        edge_params={Props.SQL_ID: str(sql_id)},
        parent=parent,
        is_full_parse=is_full_parse,
    )

    if len(where_dict) > 1:
        raise Exception(
            f'Unexpected structure of the where clause in "create_where_node". where_dict\n: {where_dict}'
        )

    _, expr_sql_str, sql_part, data_type = handle_expression(
        query_obj, SQL.WHERE, where_dict, where_node, sql_id, None, is_full_parse
    )
    pointer_str = "where " + expr_sql_str
    pointer_sql_parts_str = "where " + sql_part
    handle_node_update(
        label=label,
        node=where_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": pointer_str, "sql_part": pointer_sql_parts_str},
    )
    return pointer_str, pointer_sql_parts_str


def create_insubquery_node(
    query_obj: Query,
    section_type: str,
    insubquery_dict: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    negated = insubquery_dict[Parser.NEGATED]
    if negated:
        in_node_name = "Not In"
    else:
        in_node_name = "In"
    label = Labels.OPERATOR
    in_node = query_obj.create_current_node(
        name=in_node_name,
        label=label,
        props={"name": in_node_name, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )
    for iKey, iVal in insubquery_dict.items():
        if iKey == "expr":
            if next(iter(iVal)) == Parser.TUPLE:
                iVal = iVal[Parser.TUPLE]
                for item in iVal:
                    _, col_sql_str, col_sql_part, data_type = handle_expression(
                        query_obj=query_obj,
                        section_type=section_type,
                        expr_dict=item,
                        parent=in_node,
                        sql_id=sql_id,
                        is_full_parse=is_full_parse,
                    )
            else:
                _, col_sql_str, col_sql_part, data_type = handle_expression(
                    query_obj=query_obj,
                    section_type=section_type,
                    expr_dict=iVal,
                    parent=in_node,
                    sql_id=sql_id,
                    is_full_parse=is_full_parse,
                )
        elif iKey == Parser.SUBQUERY:
            sql_id_increased, sql_node, subquery_obj = create_sub_sql_node(
                query_obj=query_obj,
                section_type=section_type,
                parent=in_node,
                sql_id=sql_id,
                is_sub_select=True,
                is_full_parse=is_full_parse,
                optional_edge_params=optional_edge_params,
            )
            previous_with_subselects = query_obj.get_subselects_by_section(SQL.WITH)
            for previous_with in previous_with_subselects:
                subquery_obj.add_subselect(
                    previous_with, previous_with.get_id(), SQL.WITH
                )
                subquery_obj.add_alias_to_table(previous_with.get_name(), previous_with)
            subquery_sql_str, subquery_sql_part, _ = create_query(
                query_obj=subquery_obj,
                parent=sql_node,
                query_dict=iVal,
                sql_id=sql_id_increased,
                is_full_parse=is_full_parse,
            )
            query_obj.add_edges(subquery_obj.get_edges(), section_type)
        elif iKey == Parser.NEGATED:
            """Already handled before creating the "In" or "Not In" node."""
            pass
        else:
            raise Exception(
                f'Unknown key in InSubquery: {iKey} in "create_insubquery_node". insubquery_dict:\n {str(insubquery_dict)}'
            )

    pointer_str = f"{col_sql_str} {in_node_name} ({subquery_sql_str})"
    pointer_sql_parts_str = f"{col_sql_part} {in_node_name} ({subquery_sql_part})"
    handle_node_update(
        label=label,
        node=in_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": pointer_str, "sql_part": pointer_sql_parts_str},
    )
    data_type = ["boolean"]
    return in_node, pointer_str, pointer_sql_parts_str, data_type


def create_projection_nodes(
    query_obj: Query,
    projection_list: list[dict[str, Any]],
    parent: Node,
    sql_id: str,
    is_full_parse: bool = False,
) -> tuple[Optional[str], str, str]:
    data_types = []
    proj_sql_str = []
    proj_sql_parts = []
    for proj_item in projection_list:
        if proj_item == Parser.WILDCARD or (
            isinstance(proj_item, dict) and Parser.WILDCARD in proj_item
        ):  # need to understand the new properties
            label = Labels.CONSTANT
            connect_wildcard_to_query_tables(
                query_obj=query_obj,
                parent=parent,
                sql_id=sql_id,
                section_type=SQL.SELECT,
                agg_function=False,
                is_full_parse=is_full_parse,
            )
            proj_sql_str.append("*")
        else:
            for proj_key, proj_val in proj_item.items():
                if proj_key == Parser.Q_WILDCARD:
                    label = Labels.CONSTANT
                    table_name = (
                        proj_val[0][0]["value"]
                        if isinstance(proj_val[0], list)
                        else proj_val[0]["value"]
                    )  # 0.2.28 has one more inner list
                    if table_name.lower() in query_obj.get_tables_aliases().keys():
                        table_node = query_obj.get_tables_aliases()[table_name.lower()]
                    else:
                        table_node = query_obj.get_table_node_from_schema(
                            table_name, ""
                        )
                    connect_wildcard_to_query_tables(
                        query_obj=query_obj,
                        parent=parent,
                        sql_id=sql_id,
                        section_type=SQL.SELECT,
                        agg_function=False,
                        is_full_parse=is_full_parse,
                        table_node=table_node,
                    )
                    proj_sql_str.append(f"{table_name}.*")

                elif proj_key == "ExprWithAlias":
                    alias_name = proj_val["alias"]["value"]
                    label = Labels.ALIAS
                    alias_node = add_alias(
                        query_obj=query_obj,
                        section_type=SQL.SELECT,
                        parent=parent,
                        sql_id=sql_id,
                        alias_name=alias_name,
                        optional_edge_params=None,
                        is_full_parse=is_full_parse,
                    )
                    proj_val = proj_val["expr"]
                    (
                        _,
                        expr_sql_str,
                        sql_part,
                        data_type,
                    ) = handle_expression(
                        query_obj=query_obj,
                        section_type=SQL.SELECT,
                        expr_dict=proj_val,
                        parent=alias_node,
                        sql_id=sql_id,
                        is_full_parse=is_full_parse,
                    )
                    handle_node_update(
                        label=label,
                        node=alias_node,
                        is_full_parse=is_full_parse,
                        payload={
                            "aliased_expr": f"{expr_sql_str}",
                            "expr_str": f"{expr_sql_str} as {alias_name}",
                            "sql_part": f"{sql_part} as {alias_name}",
                        },
                    )
                    query_obj.add_expression_alias(alias_name, alias_node)
                    query_obj.add_projection_node(alias_node)
                    handle_node_update(
                        label=label,
                        node=alias_node,
                        is_full_parse=is_full_parse,
                        payload={"data_type": prepare_data_types(data_type)},
                    )
                    data_types.extend(data_type)
                    proj_sql_str.append(f"{expr_sql_str} as {alias_name}")
                    proj_sql_parts.append(f"{sql_part} as {alias_name}")
                elif proj_key in ["UnnamedExpr", "Unnamed"]:
                    (
                        expr_root_node,
                        expr_sql_str,
                        sql_part,
                        data_type,
                    ) = handle_expression(
                        query_obj=query_obj,
                        section_type=SQL.SELECT,
                        expr_dict=proj_val,
                        parent=parent,
                        sql_id=sql_id,
                        is_full_parse=is_full_parse,
                    )
                    query_obj.add_projection_node(expr_root_node)
                    if expr_root_node.get_label() != Labels.ALIAS and is_full_parse:
                        expr_root_node.add_property("expr_str", expr_sql_str)
                        expr_root_node.add_property("sql_part", sql_part)
                    proj_sql_str.append(expr_sql_str)
                    proj_sql_parts.append(sql_part)
                    data_types.extend(data_type)
                else:
                    raise Exception(
                        f'Unknown projection item type: {proj_key} in "create_projection_nodes". projection_list:\n {str(projection_list)}'
                    )
    proj_sql_str = ", ".join(proj_sql_str)
    proj_sql_parts = ", ".join(proj_sql_parts)
    data_types = prepare_data_types(data_types)
    return data_types, proj_sql_str, proj_sql_parts


def connect_wildcard_to_query_tables(
    query_obj: Query,
    parent: Node,
    sql_id: str,
    section_type: str,
    agg_function: bool = False,
    is_full_parse: bool = False,
    table_node: Node | Query = None,  ## for qualified wilcards
) -> None:
    edge_params = {Props.SQL_ID: str(sql_id)}
    schemas = query_obj.get_schemas()
    tables_list = (
        query_obj.get_tables() if not table_node else [(table_node.name, table_node)]
    )
    for table_name, table_node in tables_list:
        # query_obj.add_projection_node(wildcard_node)
        if isinstance(table_node, Query):
            query = table_node
            if not agg_function:
                # if the wildcard is not in an aggregation function, then project the columns
                proj_nodes = query.get_projection_nodes()
                for n in proj_nodes:
                    query_obj.add_projection_node(n)
                    if is_full_parse:
                        query_obj.add_edge((parent, n, edge_params), section_type)
        else:
            if not agg_function:
                # if the wildcard is not in an aggregation function, then project the columns
                schema_name_lower = table_node.get_match_props()["schema_name"].lower()
                column_names = schemas[schema_name_lower].get_table_columns(table_node)
                for column_name in column_names:
                    column_node = schemas[schema_name_lower].get_column_node(
                        column_name, table_node.get_name()
                    )
                    column_node.get_properties().update(
                        {"last_query_timestamp": query_obj.get_latest_timestamp()}
                    )
                    query_obj.add_projection_node(column_node)
                    if is_full_parse:
                        query_obj.add_edge(
                            (parent, column_node, edge_params), section_type
                        )
        if not agg_function and section_type == SQL.SELECT:
            query_obj.add_wilcarded_table(table_node)


def create_from_node(
    query_obj: Query,
    from_list: list[dict[str, Any]],
    parent: Node,
    sql_id: str,
    fks: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[str, str]:
    label = Labels.COMMAND
    from_node = query_obj.create_current_node(
        name="From",
        label=label,
        props={"name": "From", Props.SQL_ID: str(sql_id)},
        section_type=SQL.FROM,
        edge_params={Props.SQL_ID: str(sql_id)},
        parent=parent,
        is_full_parse=is_full_parse,
    )

    from_sql_str = ""
    from_sql_parts_str = ""
    previous_tables_list = []
    for from_list_item in from_list:
        for dict_item_key, dict_item_val in from_list_item.items():
            if dict_item_key == "relation":
                (
                    previous_table_node,
                    table_sql_str,
                    table_sql_parts_str,
                ) = handle_relation(
                    query_obj=query_obj,
                    section_type=SQL.FROM,
                    relation=dict_item_val,
                    parent=from_node,
                    sql_id=sql_id,
                    is_full_parse=is_full_parse,
                )
                previous_tables_list.append(previous_table_node)
                if (
                    "Derived" in dict_item_val
                    and dict_item_val["Derived"]["alias"] is not None
                ):  # add alias if exists
                    table_sql_str = (
                        table_sql_str
                        + " "
                        + dict_item_val["Derived"]["alias"]["name"]["value"]
                    )
                    table_sql_parts_str = (
                        table_sql_parts_str
                        + " "
                        + dict_item_val["Derived"]["alias"]["name"]["value"]
                    )

                elif (
                    "Table" in dict_item_val
                    and dict_item_val["Table"]["alias"] is not None
                ):
                    table_sql_str = (
                        table_sql_str
                        + " "
                        + dict_item_val["Table"]["alias"]["name"]["value"]
                    )
                    table_sql_parts_str = (
                        table_sql_parts_str
                        + " "
                        + dict_item_val["Table"]["alias"]["name"]["value"]
                    )
                from_sql_str = f"{from_sql_str}{table_sql_str} "
                from_sql_parts_str = f"{from_sql_parts_str}{table_sql_parts_str} "
            elif dict_item_key == "joins":
                if len(dict_item_val) > 0:
                    join_sql_str, join_sql_parts_str = create_joins_node(
                        query_obj,
                        SQL.FROM,
                        dict_item_val,
                        from_node,
                        sql_id,
                        previous_tables_list,
                        fks,
                        is_full_parse=is_full_parse,
                    )
                    from_sql_str = f"{from_sql_str}{join_sql_str} "
                    from_sql_parts_str = f"{from_sql_parts_str}{join_sql_parts_str} "
            else:
                raise Exception(
                    f'Unknown item key: {dict_item_key} in "create_from_node". from_list:\n {str(from_list)}'
                )
    handle_node_update(
        label=label,
        node=from_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": from_sql_str, "sql_part": from_sql_parts_str},
    )
    return from_sql_str, from_sql_parts_str


def handle_relation(
    query_obj: Query,
    section_type: str,
    relation: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Union[Node, Query], str, str]:
    relation_type = next(iter(relation))
    while relation_type.lower() in [
        "NestedJoin".lower(),
        "Pivot".lower(),
    ]:  # ignore being nested
        relation = relation[relation_type]
        relation_type = next(iter(relation))

    if relation_type == "TableFunction" or (
        relation_type == "Table"
        and not relation[relation_type]["name"][0]["value"] == "UNNEST"
    ):
        table_node, table_sql_str, table_sql_parts = create_table_node(
            query_obj,
            section_type,
            relation[relation_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse=is_full_parse,
        )
        query_obj.add_table_to_query(table_node, table_node.get_name())
        return table_node, table_sql_str, table_sql_parts
    elif (
        relation_type == "Derived"
        or (
            relation_type == "Table"
            and relation[relation_type]["name"][0]["value"] == "UNNEST"
        )
        or (
            relation_type.lower() == "Table".lower()
            and next(iter(relation[relation_type])) == "Derived"
        )
    ):
        if relation_type == "Table":
            subquery_dict = {
                "with": None,
                "body": {
                    "Select": {
                        "distinct": False,
                        "top": None,
                        "projection": [],  # relation['Table']['args'],
                        "into": None,
                        "from": [],
                        "lateral_views": [],
                        "selection": None,
                        "group_by": [],
                        "cluster_by": [],
                        "distribute_by": [],
                        "sort_by": [],
                        "having": None,
                        "qualify": None,
                    }
                },
                "order_by": [],
                "limit": None,
                "offset": None,
                "fetch": None,
                "locks": [],
            }
        elif Parser.SUBQUERY.lower() in relation[relation_type].keys():
            subquery_dict = relation[relation_type][Parser.SUBQUERY.lower()]
        elif Parser.SUBQUERY.lower() in relation[relation_type]["Derived"].keys():
            subquery_dict = relation[relation_type]["Derived"][Parser.SUBQUERY.lower()]
        else:
            raise Exception(
                f'Unknown json structure under relation[Derived] in "handle_relation". relation:\n {str(relation)}'
            )
        query_alias = None
        if relation[relation_type]["alias"] is not None:
            query_alias = relation[relation_type]["alias"]["name"]["value"]
            alias_node = add_alias(
                query_obj,
                section_type,
                parent,
                sql_id,
                query_alias,
                optional_edge_params,
                is_full_parse=is_full_parse,
            )
            parent = alias_node
            optional_edge_params = None
        elif (
            "Derived" in relation[relation_type]
            and relation[relation_type]["Derived"]["alias"] is not None
        ):
            query_alias = relation[relation_type]["alias"]["name"]["value"]
            alias_node = add_alias(
                query_obj,
                section_type,
                parent,
                sql_id,
                query_alias,
                optional_edge_params,
                is_full_parse=is_full_parse,
            )
            parent = alias_node
            optional_edge_params = None

        sql_id_increased, sql_node, subquery_obj = create_sub_sql_node(
            query_obj,
            section_type,
            optional_edge_params,
            parent,
            sql_id,
            is_sub_select=True,
            is_full_parse=is_full_parse,
        )
        previous_with_subselects = query_obj.get_subselects_by_section(SQL.WITH)
        for previous_with in previous_with_subselects:
            subquery_obj.add_subselect(previous_with, previous_with.get_id(), SQL.WITH)
            subquery_obj.add_alias_to_table(previous_with.get_name(), previous_with)
        subquery_sql_str, subquery_sql_parts, _ = create_query(
            subquery_obj,
            subquery_dict,
            parent=sql_node,
            sql_id=sql_id_increased,
            is_full_parse=is_full_parse,
        )
        query_obj.add_edges(subquery_obj.get_edges(), Parser.SUBSELECT)
        if relation[relation_type]["alias"] is not None:
            # handle the case of Values as the derived table; in such a case it is required to fetch the names
            # of the projected columns and add them to the subquery that represents the derived table.
            if (
                "columns" in relation[relation_type]["alias"]
                and len(relation[relation_type]["alias"]["columns"]) > 0
            ):
                subquery_select_node = subquery_obj.select_edges[0][1]
                for c in relation[relation_type]["alias"]["columns"]:
                    col_alias_node = add_alias(
                        subquery_obj,
                        SQL.SELECT,
                        subquery_select_node,
                        sql_id_increased,
                        c["value"],
                        optional_edge_params,
                        is_full_parse=is_full_parse,
                    )
                    handle_node_update(
                        label=Labels.ALIAS,
                        node=col_alias_node,
                        is_full_parse=is_full_parse,
                        payload={"aliased_expr": "", "expr_str": "", "sql_part": ""},
                    )

                    subquery_obj.projection_nodes.append(col_alias_node)

            query_obj.add_alias_to_table(query_alias, subquery_obj)
            handle_node_update(
                label=Labels.ALIAS,
                node=alias_node,
                is_full_parse=is_full_parse,
                payload={
                    "aliased_expr": f"{subquery_sql_str}",
                    "expr_str": f"{subquery_sql_str} as {query_alias}",
                    "sql_part": f"{subquery_sql_parts} as {query_alias}",
                },
            )
        query_obj.add_table_to_query(subquery_obj, query_alias)
        return subquery_obj, f"({subquery_sql_str})", f"({subquery_sql_parts})"

    elif relation_type == "table_with_joins":
        alias = None
        if relation["alias"] is not None:
            alias = relation["alias"]["name"]["value"]
            alias_node = add_alias(
                query_obj, section_type, parent, sql_id, alias, optional_edge_params
            )
            parent = alias_node
            optional_edge_params = None

        (
            nested_join_node,
            nested_join_str,
            nested_join_sql_str,
        ) = create_nested_join_node(
            query_obj,
            section_type,
            relation[relation_type],
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse=is_full_parse,
        )
        if relation["alias"] is not None:
            query_obj.add_alias_to_table(alias, nested_join_node)
            handle_node_update(
                query_obj=query_obj,
                label=Labels.ALIAS,
                node=alias_node,
                parent=parent,
                is_full_parse=is_full_parse,
                payload={
                    "aliased_expr": f"{nested_join_str}",
                    "expr_str": f"{nested_join_str} as {alias}",
                    "sql_part": f"{nested_join_sql_str} as {alias}",
                },
            )
        return nested_join_node, nested_join_str, nested_join_sql_str
    # elif relation_type == "Function":
    #     pass
    else:
        raise Exception(
            f'Unknown relation type: {relation_type} in "handle_relation". relation:\n {str(relation)}'
        )


def get_full_table_str(table_node: Union[Node, Query]) -> str:
    table_name = table_node.get_name()
    if isinstance(table_node, Node):
        schema_name = table_node.get_match_props()["schema_name"]
        return f"{schema_name}.{table_name}"
    else:
        return f"{table_name}"


def create_table_node(
    query_obj: Query,
    section_type: str,
    table_val: dict[str, Any],
    parent: Node,
    sql_id: str,
    optional_edge_params: Optional[dict[str, Any]] = None,
    is_full_parse: bool = False,
) -> tuple[Node, str, str, list[str]]:
    schema_name = ""
    table_name = None
    table_node = None
    if "name" in table_val:
        if len(table_val["name"]) == 1:  # no schema name in the table identifier
            table_name = table_val["name"][0]["value"]
        elif len(table_val["name"]) == 2:
            schema_name = table_val["name"][0]["value"]
            table_name = table_val["name"][1]["value"]
        elif len(table_val["name"]) == 3:
            # the first name, which we ignore, is the db_name
            schema_name = table_val["name"][1]["value"]
            table_name = table_val["name"][2]["value"]
        elif len(table_val["name"]) == 4:
            # the first name is unknown, the second name is the db_name - we currently ignore both
            schema_name = table_val["name"][2]["value"]
            table_name = table_val["name"][3]["value"]
        table_node = query_obj.get_table_node_from_schema(table_name, schema_name)
    if not table_node:
        # This is a TableFunction
        if "expr" in table_val:
            table_val_expr = table_val["expr"]
            expr_type = next(iter(table_val_expr))
            if expr_type == Parser.FUNCTION:
                table_val_expr = table_val_expr[expr_type]
                table_name = table_val_expr["name"][0]["value"]
            if (
                table_name is not None
                and table_name in SQLFunctions.relation_functions
                and len(table_val_expr["name"]) == 1
            ):
                # parse the table as a function
                function_node, func_sql_str, func_sql_parts_str, _ = (
                    create_function_node(
                        query_obj,
                        section_type,
                        table_val_expr,
                        parent,
                        sql_id,
                        is_full_parse=is_full_parse,
                    )
                )
                return function_node, func_sql_str, func_sql_parts_str
            if (
                Parser.VALUE in table_val_expr
                and "SingleQuotedString" in table_val_expr[Parser.VALUE]
            ):
                # try to retrieve the table name
                unparsed_table_name = table_val_expr[expr_type]["SingleQuotedString"]
                if "." in unparsed_table_name:
                    schema_name = unparsed_table_name[: unparsed_table_name.find(".")]
                    table_name = unparsed_table_name[
                        unparsed_table_name.find(".") + 1 :
                    ]
                else:
                    table_name = unparsed_table_name
                table_node = query_obj.get_table_node_from_schema(
                    table_name, schema_name
                )
        if not table_node:
            raise MissingDataError(
                f'"create_table_node": Table "{table_name}" not in schema {schema_name}.'
            )
    table_sql_str = get_full_table_str(table_node)
    # try:
    #     match_props = table_node.get_match_props()
    #     if 'schema_name' in match_props:
    #         table_sql_parts = f"{match_props['schema_name']}.{match_props['name']}"
    #     else:
    #         table_sql_parts = f"{match_props['name']}"
    # except:
    table_sql_parts = get_full_table_str(table_node)
    if "alias" not in table_val or table_val["alias"] is None:
        edge_params = create_edge_params(optional_edge_params, sql_id)
        query_obj.add_edge((parent, table_node, edge_params), section_type)
    else:
        table_alias = table_val["alias"]["name"]["value"]
        query_obj.add_alias_to_table(table_alias, table_node)
        alias_node = add_alias(
            query_obj,
            section_type,
            parent,
            sql_id,
            table_alias,
            optional_edge_params,
            is_full_parse=is_full_parse,
        )
        # handle_node_update(
        #     query_obj=query_obj,
        #     label=Labels.ALIAS,
        #     node=alias_node,
        #     parent=parent,
        #     is_full_parse=is_full_parse,
        #     payload={
        #         "aliased_expr": f"{table_sql_str}",
        #         "expr_str": f"{table_sql_str} as {table_alias}",
        #         "sql_part": f"{table_sql_str} as {table_alias}",
        #     },
        # )
        edge_params = create_edge_params(None, sql_id)
        query_obj.add_edge((alias_node, table_node, edge_params), section_type)
    return table_node, table_sql_str, table_sql_parts


def add_alias(
    query_obj: Query,
    section_type: str,
    parent: Node,
    sql_id: str,
    alias_name: str,
    optional_edge_params: dict = None,
    is_full_parse: bool = False,
    is_cte_table: bool = False,
):
    ## in the slim version we want the main sql node to be connected directly to the projection aliases (not the cte table aliases)
    if is_cte_table and not is_full_parse:
        # make sure to count the cte table alias also in the slim version
        query_obj.increment_nodes_counter()
        return parent
    alias_node = query_obj.create_current_node(
        name=alias_name,
        label=Labels.ALIAS,
        props={
            "name": alias_name,
            Props.SQL_ID: str(sql_id),
            "is_cte_table": is_cte_table,
        },
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )
    return alias_node


def create_join_operator_node(
    query_obj,
    section_type,
    join_operator,
    parent,
    sql_id,
    previous_tables_list,
    current_table,
    table_sql_str,
    table_sql_part_str,
    optional_edge_params=None,
    fks=None,
    is_full_parse=False,
):
    join_type = (
        join_operator if isinstance(join_operator, str) else next(iter(join_operator))
    )
    label = Labels.COMMAND
    join_node = query_obj.create_current_node(
        name=join_type,
        label=label,
        props={"name": join_type, Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    joined_objects = previous_tables_list + [current_table]
    operator_string = "True"
    join_op_sql_str = JoinNodes.join_str_map[join_type] + " join"
    join_op_sql_parts = JoinNodes.join_str_map[join_type] + " join "

    if join_type == "CrossJoin" or join_operator[join_type] == "Natural":
        join_name = "CrossJoin" if isinstance(join_operator, str) else "Natural"
        _ = query_obj.create_current_node(
            name=join_name,
            label=Labels.COMMAND,
            props={"name": join_name, Props.SQL_ID: str(sql_id)},
            section_type=section_type,
            edge_params=create_edge_params(optional_edge_params, sql_id),
            parent=join_node,
            is_full_parse=is_full_parse,
        )
        join_op_sql_str = (
            "cross join" if isinstance(join_operator, str) else "natural join "
        )
        join_op_sql_parts = join_op_sql_str
    elif join_operator[join_type] == "None":
        pass  # Nothing else to parse here
    elif "On" in join_operator[join_type].keys():
        _, expr_sql_str, sql_part, data_type = handle_expression(
            query_obj=query_obj,
            section_type=section_type,
            expr_dict=join_operator[join_type]["On"],
            parent=join_node,
            sql_id=sql_id,
            is_full_parse=is_full_parse,
        )
        joined_objects = get_joined_objects(
            query_obj, join_operator[join_type]["On"], fks, is_full_parse=is_full_parse
        )
        join_op_sql_str = f"{join_op_sql_str} {table_sql_str} on {expr_sql_str} "
        join_op_sql_parts = f"{join_op_sql_parts} {table_sql_part_str} on {sql_part} "
        operator_string = expr_sql_str
    elif "Using" in join_operator[join_type].keys():
        # create the same structure as would have been created if instead of "Using" would have been "On"
        op_node_parent = join_node
        columns_list = join_operator[join_type]["Using"]
        columns_str_list = []

        if len(columns_list) > 1:
            op_node_parent = query_obj.create_current_node(
                name="And",
                label=Labels.OPERATOR,
                props={"name": "And", Props.SQL_ID: str(sql_id)},
                section_type=section_type,
                edge_params=create_edge_params(None, sql_id),
                parent=join_node,
                is_full_parse=is_full_parse,
            )

        for i, col_item in zip(range(len(columns_list)), columns_list):
            op_node = query_obj.create_current_node(
                name="Eq",
                label=Labels.OPERATOR,
                props={"name": "Eq", Props.SQL_ID: str(sql_id)},
                section_type=section_type,
                edge_params=create_edge_params(None, sql_id),
                parent=op_node_parent,
                is_full_parse=is_full_parse,
            )

            col_name = col_item["value"]
            columns_str_list.append(col_name)
            op_strs = []
            for table_node in previous_tables_list:
                if query_obj.is_column_in_table(table_node, col_name):
                    op_strs.append(
                        f"{table_node.get_name()}.{col_name}={current_table.get_name()}.{col_name}"
                    )
            col_node_current = query_obj.get_column_node(col_name, current_table)
            edge_params = create_edge_params(None, sql_id)
            query_obj.add_edge((op_node, col_node_current, edge_params), section_type)
        if op_strs:
            operator_string = " and ".join(op_strs)
        else:
            operator_string = "True"
        columns_list_str = ", ".join(columns_str_list)
        join_op_sql_str = f"{join_op_sql_str} {table_sql_str} using({columns_list_str}) "  # to be tested with sql parts
    else:
        raise Exception(
            f'Unknown join_operator type in "create_join_operator_node". join_operator:\n {str(join_operator)}'
        )
    for table_or_sql_node in joined_objects:
        edge_params = create_edge_params(None, sql_id)
        query_obj.add_edge((join_node, table_or_sql_node, edge_params), section_type)
    if joined_objects:
        if len(joined_objects) > 1:
            reduce(
                lambda prev_table_node, current_table_node: reduce_update_operator_str(
                    join_node,
                    label,
                    operator_string,
                    prev_table_node,
                    current_table_node,
                    is_full_parse,
                ),
                joined_objects,
            )
        else:
            reduce_update_operator_str(
                join_node,
                label,
                operator_string,
                joined_objects[0],
                joined_objects[0],
                is_full_parse,
            )
    else:
        pass
        # logger.error(f"No joined objects for join operator {join_type}")

    handle_node_update(
        label=label,
        node=join_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": join_op_sql_str,
            "sql_part": join_op_sql_parts,
        },
    )
    return join_op_sql_str, join_op_sql_parts


def reduce_update_operator_str(
    join_node: Node,
    label: Labels,
    current_operator: str,
    prev_table_node: Node,
    current_table_node: Node,
    is_full_parse: bool = False,
):
    operator_prop = [
        prev_table_node.get_id(),
        current_table_node.get_id(),
        current_operator,
    ]
    existing_operators = join_node.get_properties().get("operators", [])
    existing_operators.append(operator_prop)
    handle_node_update(
        label=label,
        node=join_node,
        is_full_parse=is_full_parse,
        payload={
            "operators": existing_operators,
        },
    )
    return current_table_node


def get_joined_objects(
    query_obj: Query,
    join_on_dict: dict,
    fks: Optional[list] = None,
    is_full_parse: bool = False,
):
    # return the table nodes or derived tables (subquery nodes)
    if fks:
        try:
            node_to_test_left, _, _ = create_column_node(
                query_obj=query_obj,
                section_type="select",
                column_dict=join_on_dict["BinaryOp"]["left"],
                parent=None,
                sql_id=None,
                is_full_parse=is_full_parse,
            )
            node_to_test_right, _, _ = create_column_node(
                query_obj=query_obj,
                section_type="select",
                column_dict=join_on_dict["BinaryOp"]["right"],
                parent=None,
                sql_id=None,
                is_full_parse=is_full_parse,
            )
            table_left = node_to_test_left.match_props["table_name"].lower()
            column_left = node_to_test_left.match_props["name"].lower()

            table_right = node_to_test_right.match_props["table_name"].lower()
            column_right = node_to_test_right.match_props["name"].lower()

            found = False
            for fk in fks:
                if table_left == table_right and column_left == column_right:
                    found = True
                else:
                    t_fk_left = fk["table1"].lower()
                    c_fk_left = fk["column1"].lower()
                    t_fk_right = fk["table2"].lower()
                    c_fk_right = fk["column2"].lower()
                    if (
                        t_fk_left == table_left
                        and c_fk_left == column_left
                        and t_fk_right == table_right
                        and c_fk_right == column_right
                    ):
                        found = True
                    elif (
                        t_fk_left == table_right
                        and c_fk_left == column_right
                        and t_fk_right == table_left
                        and c_fk_right == column_left
                    ):
                        found = True
            if not found:
                raise NoFKError(
                    f"Foreign key not found between {table_right}.{column_right} and {table_left}.{column_left}"
                )
        except KeyError as ke:  # TODO if some exception consider fks are ok. recheck
            logger.error(ke)

    columns = get_key_recursive(join_on_dict, "Identifier")
    columns.extend(get_key_recursive(join_on_dict, "CompoundIdentifier"))

    joined_objects = []
    for c_dict in columns:
        if isinstance(c_dict, dict):
            column_name = c_dict["value"]
            jo_node = query_obj.find_the_table_node_of_the_column(column_name)
        else:
            if len(c_dict) == 2:
                schema_name = ""
                table_name = c_dict[0]["value"]
            elif len(c_dict) == 3:
                schema_name = c_dict[0]["value"]
                table_name = c_dict[1]["value"]
            jo_node = query_obj.get_table_node_from_schema(table_name, schema_name)
        if not is_full_parse and isinstance(jo_node, Query):
            tables_of_subselect = query_obj.get_tables_of_subselect(jo_node.id)
            joined_objects.extend(tables_of_subselect)
        elif jo_node and jo_node not in joined_objects:
            joined_objects.append(jo_node)
    return joined_objects


def create_nested_join_node(
    query_obj: Query,
    section_type: str,
    nestedjoin_dict: dict,
    parent: Node,
    sql_id: str,
    optional_edge_params: dict = None,
    is_full_parse: bool = False,
):
    join_node = query_obj.create_current_node(
        name="Joins",
        label=Labels.COMMAND,
        props={"name": "Joins", Props.SQL_ID: str(sql_id), "is_nested": True},
        section_type=section_type,
        edge_params=create_edge_params(optional_edge_params, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    from_sql_str = ""
    from_sql_parts_str = ""
    previous_tables_list = []

    relation = nestedjoin_dict["relation"]
    previous_table_node, table_sql_str, table_sql_parts_str = handle_relation(
        query_obj,
        section_type,
        relation,
        join_node,
        sql_id,
        is_full_parse=is_full_parse,
    )
    previous_tables_list.append(previous_table_node)
    if (
        "Derived" in relation and relation["Derived"]["alias"] is not None
    ):  # add alias if exists
        table_sql_str = (
            table_sql_str + " " + relation["Derived"]["alias"]["name"]["value"]
        )
        table_sql_parts_str = (
            table_sql_parts_str + " " + relation["Derived"]["alias"]["name"]["value"]
        )

    elif "Table" in relation and relation["Table"]["alias"] is not None:
        table_sql_str = (
            table_sql_str + " " + relation["Table"]["alias"]["name"]["value"]
        )
        table_sql_parts_str = (
            table_sql_parts_str + " " + relation["Table"]["alias"]["name"]["value"]
        )
    from_sql_str = f"{from_sql_str}{table_sql_str} "
    from_sql_parts_str = f"{from_sql_parts_str}{table_sql_parts_str} "

    joins = nestedjoin_dict["joins"]
    if len(joins) > 0:
        join_sql_str, join_sql_parts_str = create_joins_node(
            query_obj,
            section_type,
            joins,
            join_node,
            sql_id,
            previous_tables_list,
            is_full_parse=is_full_parse,
        )
        from_sql_str = f"{from_sql_str}{join_sql_str} "
        from_sql_parts_str = f"{from_sql_parts_str}{join_sql_parts_str} "
    handle_node_update(
        label=Labels.COMMAND,
        node=join_node,
        is_full_parse=is_full_parse,
        payload={
            "expr_str": from_sql_str,
            "sql_part": from_sql_parts_str,
            "tables": previous_tables_list,
        },
    )
    return join_node, from_sql_str, from_sql_parts_str


def create_joins_node(
    query_obj: dict,
    section_type: str,
    joins_list: list,
    parent: Node,
    sql_id: str,
    previous_tables_list: list,
    fks=None,
    is_full_parse=False,
):
    label = Labels.COMMAND
    join_node = query_obj.create_current_node(
        name="Joins",
        label=label,
        props={"name": "Joins", Props.SQL_ID: str(sql_id)},
        section_type=section_type,
        edge_params={Props.SQL_ID: str(sql_id)},
        parent=parent,
        is_full_parse=is_full_parse,
    )

    join_sql_str = ""
    join_sql_parts_str = ""
    for child_idx, join_list_item in enumerate(joins_list):
        optional_edge_params = {"child_idx": child_idx}
        for dict_item_key, dict_item_val in join_list_item.items():
            if dict_item_key == "relation":
                (
                    current_table_node,
                    table_sql_str,
                    table_sqls_parts_str,
                ) = handle_relation(
                    query_obj,
                    section_type,
                    dict_item_val,
                    parent,
                    sql_id,
                    optional_edge_params,
                )
                if (
                    "Derived" in dict_item_val
                    and dict_item_val["Derived"]["alias"] is not None
                ):
                    table_sql_str = (
                        table_sql_str
                        + " "
                        + dict_item_val["Derived"]["alias"]["name"]["value"]
                    )
                    table_sqls_parts_str = (
                        table_sqls_parts_str
                        + " "
                        + dict_item_val["Derived"]["alias"]["name"]["value"]
                    )
                elif (
                    "Table" in dict_item_val
                    and dict_item_val["Table"]["alias"] is not None
                ):
                    table_sql_str = (
                        table_sql_str
                        + " "
                        + dict_item_val["Table"]["alias"]["name"]["value"]
                    )
                    table_sqls_parts_str = (
                        table_sqls_parts_str
                        + " "
                        + dict_item_val["Table"]["alias"]["name"]["value"]
                    )
            elif dict_item_key == "join_operator":
                join_op_sql_str, join_op_sql_parts = create_join_operator_node(
                    query_obj,
                    section_type,
                    dict_item_val,
                    join_node,
                    sql_id,
                    previous_tables_list,
                    current_table_node,
                    table_sql_str,
                    table_sqls_parts_str,
                    optional_edge_params,
                    fks,
                    is_full_parse,
                )
                previous_tables_list.append(current_table_node)
                # join_op_sql_str = join_op_sql_str.replace("PLACEHOLDER", table_sql_str)
            elif dict_item_key == "global":
                continue
            else:
                raise Exception(
                    f'Unknown join item type: {dict_item_key} in "create_joins_node". joins_list:\n {str(joins_list)}'
                )

        child_idx += 1
        join_sql_str = f"{join_sql_str}{join_op_sql_str}"
        join_sql_parts_str = f"{join_sql_parts_str}{join_op_sql_parts}"

    handle_node_update(
        label=label,
        node=join_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": join_sql_str, "sql_part": join_sql_parts_str},
    )
    return join_sql_str, join_sql_parts_str


def create_query(query_obj, query_dict, parent, sql_id, fks=None, is_full_parse=False):
    # No support for FETCH and OFFSET
    with_sql_str, with_sql_parts_str = create_with_node(
        query_obj, query_dict[SQL.WITH], parent, sql_id, is_full_parse=is_full_parse
    )
    data_type, body_sql_str, body_sql_part = create_body(
        query_obj,
        query_dict[Parser.BODY],
        parent,
        sql_id,
        fks,
        is_full_parse=is_full_parse,
    )
    data_type = prepare_data_types(data_type)
    limit_sql_str = create_limit_node(
        query_obj=query_obj,
        limit_dict=query_dict[SQL.LIMIT],
        parent=parent,
        sql_id=sql_id,
        is_full_parse=is_full_parse,
    )
    order_by_sql_str, order_by_sql_parts_str = create_order_by_node(
        query_obj, query_dict[SQL.ORDER_BY], parent, sql_id, is_full_parse=is_full_parse
    )
    with_sql_str = "" if with_sql_str is None else with_sql_str
    with_sql_parts_str = "" if with_sql_parts_str is None else with_sql_parts_str
    body_sql_str = "" if body_sql_str is None else body_sql_str
    body_sql_part = "" if body_sql_part is None else body_sql_part
    limit_sql_str = "" if limit_sql_str is None else limit_sql_str
    order_by_sql_str = "" if order_by_sql_str is None else order_by_sql_str
    order_by_sql_parts_str = (
        "" if order_by_sql_parts_str is None else order_by_sql_parts_str
    )
    parent.add_property("data_type", data_type)
    if query_obj.is_subselect:
        subselect_exp = (
            "("
            + " ".join(
                f"{with_sql_str} {body_sql_str} {limit_sql_str} {order_by_sql_str}".split()
            )
            + ")"
        )
        # query_obj.sql_node.props["expr_str"] = subselect_exp
        subselect_sql_part = (
            "("
            + " ".join(
                f"{with_sql_parts_str} {body_sql_part} {limit_sql_str} {order_by_sql_parts_str}".split()
            )
            + ")"
        )
        # query_obj.sql_node.props["sql_part"] = subselect_sql_part
        return subselect_exp, subselect_sql_part, data_type
    full_select_str = " ".join(
        f"{with_sql_str} {body_sql_str} {limit_sql_str} {order_by_sql_str}".split()
    )
    return full_select_str, full_select_str, data_type


def create_with_node(
    query_obj: Query,
    with_dict: dict,
    parent: Node,
    sql_id: str,
    is_full_parse: bool = False,
):
    """
    The structure of the with_dict is as follows:
    list of tables, sometimes with columns, and for each table,
    the select query from the schema that constructs it.
    """
    if with_dict is None:
        return "", ""
    label = Labels.COMMAND
    with_node = query_obj.create_current_node(
        name="With",
        label=label,
        props={"name": "With", Props.SQL_ID: str(sql_id)},
        section_type=SQL.WITH,
        edge_params={Props.SQL_ID: str(sql_id)},
        parent=parent,
        is_full_parse=is_full_parse,
    )

    cte_tables = with_dict["cte_tables"]
    for cte_table in cte_tables:
        alias = cte_table["alias"]
        # parse alias to add the table and the columns (if they exist) to the with_obj's "schema"
        query_alias = alias["name"]["value"]
        query_alias_node = add_alias(
            query_obj,
            SQL.WITH,
            with_node,
            sql_id,
            query_alias,
            is_cte_table=True,
            is_full_parse=is_full_parse,
        )

        query = cte_table["query"]
        # connect the table defined in the WITH section to the query that feeds it
        sql_id_increased, sql_node, subquery_obj = create_sub_sql_node(
            query_obj,
            SQL.WITH,
            {"type": "with_alias"},
            parent=query_alias_node,
            sql_id=sql_id,
            is_sub_select=True,
            is_full_parse=is_full_parse,
        )
        previous_with_subselects = query_obj.get_subselects_by_section(SQL.WITH)
        for previous_with in previous_with_subselects:
            if previous_with.get_id() != subquery_obj.get_id():
                subquery_obj.add_subselect(
                    previous_with, previous_with.get_id(), SQL.WITH
                )
                subquery_obj.add_alias_to_table(previous_with.get_name(), previous_with)
        subquery_sql_str, subquery_sql_parts_str, data_type = create_query(
            subquery_obj,
            query,
            parent=sql_node,
            sql_id=sql_id_increased,
            is_full_parse=is_full_parse,
        )
        handle_node_update(
            label=Labels.ALIAS,
            node=query_alias_node,
            is_full_parse=is_full_parse,
            payload={
                "aliased_expr": f"{subquery_sql_str}",
                "expr_str": f"{subquery_sql_str} as {query_alias}",
                "sql_part": subquery_sql_parts_str,
            },
        )
        query_obj.add_edges(subquery_obj.get_edges(), Parser.SUBSELECT)

        subquery_obj.set_name(query_alias)
        query_obj.add_alias_to_table(query_alias, subquery_obj)

        for proj_node, column_alias in zip(
            subquery_obj.get_projection_nodes(), alias["columns"]
        ):
            column_alias_name = column_alias["value"]
            column_alias_node = add_alias(
                query_obj,
                SQL.WITH,
                query_alias_node,
                sql_id,
                column_alias_name,
                is_full_parse=is_full_parse,
            )
            proj_str = (
                proj_node.get_properties()["expr_str"]
                if "expr_str" in proj_node.get_properties().keys()
                else proj_node.get_name()
            )
            handle_node_update(
                label=Labels.ALIAS,
                node=column_alias_node,
                is_full_parse=is_full_parse,
                payload={
                    "expr_str": f"{proj_str} as {column_alias_name}",
                    "aliased_expr": f"{proj_str}",
                },
            )
            query_obj.add_expression_alias(column_alias_name, column_alias_node)
            subquery_obj.add_expression_alias(column_alias_name, column_alias_node)
            proj_str_sql_part = (
                proj_node.get_properties()["sql_part"]
                if "sql_part" in proj_node.get_properties()
                else proj_node.get_name()
            )
            handle_node_update(
                label=Labels.ALIAS,
                node=column_alias_node,
                is_full_parse=is_full_parse,
                payload={
                    "sql_part": proj_str_sql_part,
                    "data_type": proj_node.props["data_type"],
                },
            )

            edge_params = create_edge_params(None, sql_id)
            query_obj.add_edge((column_alias_node, proj_node, edge_params), SQL.WITH)

    handle_node_update(
        label=label,
        node=with_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": "", "sql_part": ""},
    )
    return "", ""


def create_limit_node(
    query_obj: Query,
    limit_dict: dict,
    parent: Node,
    sql_id: str,
    is_full_parse: bool = False,
):
    if limit_dict is None:
        return
    label = Labels.COMMAND
    limit_node = query_obj.create_current_node(
        name="Limit",
        label=label,
        props={"name": "Limit", Props.SQL_ID: str(sql_id)},
        section_type=SQL.LIMIT,
        edge_params=create_edge_params(None, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )
    _, value_sql_str, sql_part, _ = create_value_node(
        query_obj=query_obj,
        section_type=SQL.LIMIT,
        value_dict=limit_dict["Value"],
        parent=limit_node,
        sql_id=sql_id,
        is_full_parse=is_full_parse,
    )
    pointer_str = "limit " + value_sql_str
    pointer_parts_str = "limit " + sql_part
    handle_node_update(
        label=label,
        node=limit_node,
        is_full_parse=is_full_parse,
        payload={"expr_str": pointer_str, "sql_part": pointer_parts_str},
    )

    return pointer_str, pointer_parts_str


def create_order_by_node(query_obj, order_by_list, parent, sql_id, is_full_parse=False):
    if order_by_list is None or len(order_by_list) == 0:
        return "", ""
    if "exprs" in order_by_list:
        order_by_list = order_by_list["exprs"]
    label = Labels.COMMAND
    order_by_node = query_obj.create_current_node(
        name="Order_by",
        label=label,
        props={"name": "Order_by", Props.SQL_ID: str(sql_id)},
        section_type=SQL.ORDER_BY,
        edge_params=create_edge_params(None, sql_id),
        parent=parent,
        is_full_parse=is_full_parse,
    )

    order_by_exprs_sql_str = ""
    order_by_exprs_sql_parts_str = ""
    for order_by_item in order_by_list:
        expr = order_by_item["expr"]
        if (
            order_by_item["asc"] is None
            or order_by_item["asc"] == "true"
            or order_by_item["asc"] is True
        ):  # the default is asc order
            order_direction = "Asc"
        else:  # in this case holds: order_by_item["asc"] == "false"
            order_direction = "Desc"

        order_direction_node = query_obj.create_current_node(
            name=order_direction,
            label=Labels.COMMAND,
            props={"name": order_direction, Props.SQL_ID: str(sql_id)},
            section_type=SQL.ORDER_BY,
            edge_params=create_edge_params(None, sql_id),
            parent=order_by_node,
            is_full_parse=is_full_parse,
        )

        _, expr_sql_str, sql_part, data_type = handle_expression(
            query_obj,
            SQL.ORDER_BY,
            expr,
            order_direction_node,
            sql_id,
            None,
            is_full_parse=is_full_parse,
        )
        order_by_exprs_sql_str = (
            f"{order_by_exprs_sql_str}{expr_sql_str} {order_direction}, "
        )
        order_by_exprs_sql_parts_str = (
            f"{order_by_exprs_sql_parts_str}{sql_part} {order_direction}, "
        )
    order_by_exprs_sql_str = order_by_exprs_sql_str[:-2]
    order_by_exprs_sql_parts_str = order_by_exprs_sql_parts_str[:-2]
    query_obj.get_sql_node().add_property("orderby_str", order_by_exprs_sql_str.strip())
    pointer_str = "order by " + order_by_exprs_sql_str
    pointer_str_sql_parts = "order by " + order_by_exprs_sql_parts_str
    # handle_node_update(
    #     query_obj=query_obj,
    #     label=label,
    #     node=order_by_node,
    #     parent=parent,
    #     is_full_parse=is_full_parse,
    #     payload={"expr_str": pointer_str, "sql_part": pointer_str_sql_parts},
    # )
    return pointer_str, pointer_str_sql_parts


def create_set_op_node(
    query_obj: Query,
    section_type: str,
    set_op_dict: dict,
    parent: Node,
    sql_id: str,
    optional_edge_params: dict = None,
    is_full_parse=False,
):
    operator = next(iter(set_op_dict))
    if operator == "op":
        set_operation = set_op_dict[operator]
        # version 0.1.26 & 0.2.28
        if (
            (
                "all" in set_op_dict
                and (set_op_dict["all"] == "true" or set_op_dict["all"] is True)
            )
            or (
                "set_quantifier" in set_op_dict
                and set_op_dict["set_quantifier"] == "All"
            )
        ) and set_op_dict[operator].lower().find("all") == -1:
            set_op_dict[operator] = f"{set_operation} All"
        (
            binaryop_node,
            binaryop_sql_str,
            binary_op_sql_part,
            data_type,
        ) = create_binary_op_node(
            query_obj,
            section_type,
            set_op_dict,
            parent,
            sql_id,
            optional_edge_params,
            is_full_parse=is_full_parse,
        )
        query_obj.add_sql_strs_to_node("", binary_op_sql_part, "")

        subquery_obj_0 = query_obj.get_subselects_by_section(section_type)[0]
        sorted_projection_list_0 = sorted(
            subquery_obj_0.get_projection_nodes(), key=lambda proj: proj.get_name()
        )
        subquery_obj_1 = query_obj.get_subselects_by_section(section_type)[1]
        sorted_projection_list_1 = sorted(
            subquery_obj_1.get_projection_nodes(), key=lambda proj: proj.get_name()
        )
        for col_or_alias_0, col_or_alias_1 in zip(
            sorted_projection_list_0, sorted_projection_list_1
        ):
            column_name = col_or_alias_0.get_name()
            column_data_type = col_or_alias_0.get_properties().get("data_type")
            label = Labels.SET_OP_COLUMN
            set_op_node = query_obj.create_current_node(
                name=column_name,
                label=label,
                props={
                    "name": column_name,
                    "table_name": "",
                    "schema_name": "",
                    "data_type": column_data_type,
                },
                section_type=section_type,
                edge_params=create_edge_params(None, sql_id),
                parent=binaryop_node,
                is_full_parse=is_full_parse,
            )
            handle_node_update(
                label=label,
                node=set_op_node,
                is_full_parse=is_full_parse,
                payload={"data_type": data_type},
            )
            query_obj.add_projection_node(set_op_node)

            edge_params_0 = create_edge_params(None, sql_id)
            query_obj.add_edge(
                (set_op_node, col_or_alias_0, edge_params_0), section_type
            )
            edge_params_1 = create_edge_params(None, sql_id)
            query_obj.add_edge(
                (set_op_node, col_or_alias_1, edge_params_1), section_type
            )
        if len(subquery_obj_0.get_projection_nodes()) == 1:
            subquery_obj_0.sql_node.add_property(
                "data_type", prepare_data_types(col_or_alias_0.props.get("data_type"))
            )
            subquery_obj_1.sql_node.add_property(
                "data_type", prepare_data_types(col_or_alias_1.props.get("data_type"))
            )
        return binaryop_node, binaryop_sql_str, binary_op_sql_part, data_type
    else:
        raise Exception(
            f'Unknown operator type: {operator} in "create_set_op_node". set_op_dict:\n {str(set_op_dict)}'
        )


def create_body(
    query_obj: Query,
    body: dict,
    parent: Node,
    sql_id: str,
    fks=None,
    is_full_parse=False,
):
    key = next(iter(body))
    sql_part = ""
    if key == "Select":
        select_node, body_sql_str, sql_part, data_type = create_select_node(
            query_obj, body[key], parent, sql_id, fks, is_full_parse=is_full_parse
        )
    elif key == Parser.SET_OPERATION:
        set_op_node, body_sql_str, sql_part, data_type = create_set_op_node(
            query_obj, SQL.FROM, body[key], parent, sql_id, is_full_parse=is_full_parse
        )
    elif key == "Query":
        sql_id_increased, sql_node, subquery_obj = create_sub_sql_node(
            query_obj,
            Parser.SUBSELECT,
            None,
            parent,
            sql_id,
            is_sub_select=True,
            is_full_parse=is_full_parse,
        )
        previous_with_subselects = query_obj.get_subselects_by_section(SQL.WITH)
        for previous_with in previous_with_subselects:
            subquery_obj.add_subselect(previous_with, previous_with.get_id(), SQL.WITH)
            subquery_obj.add_alias_to_table(previous_with.get_name(), previous_with)
        body_sql_str, sql_part, data_type = create_query(
            subquery_obj, body[key], parent=sql_node, sql_id=sql_id_increased
        )
        query_obj.add_edges(subquery_obj.get_edges(), Parser.SUBSELECT)
        query_obj.projection_nodes = subquery_obj.projection_nodes
    elif key == "Values":
        select_dict = {
            "distinct": False,
            "top": None,
            # 'projection': [{'UnnamedExpr': {'Value': {'Number': ('0', False)}}}],
            "projection": [],
            "into": None,
            "from": [],
            "lateral_views": [],
            "selection": None,
            "group_by": [],
            "cluster_by": [],
            "distribute_by": [],
            "sort_by": [],
            "having": None,
            "qualify": None,
        }
        select_node, body_sql_str, sql_part, data_type = create_select_node(
            query_obj, select_dict, parent, sql_id, fks, is_full_parse=is_full_parse
        )
        # raise ValuesBodyError("No need to parse values.")
    else:
        raise Exception(
            f'Unknown body type: {key} in "create_body". body_dict:\n {str(body)}'
        )
    return data_type, body_sql_str, sql_part


def create_sub_sql_node(
    query_obj: Query,
    section_type: str,
    optional_edge_params: dict,
    parent: Node,
    sql_id: str,
    is_sub_select: bool,
    is_full_parse: bool = False,
):
    current_sql_id = str(uuid.uuid4())
    edge_params = create_edge_params(optional_edge_params, sql_id)
    sql_node = query_obj.create_current_node(
        name="query_" + str(current_sql_id),
        label=Labels.SQL,
        props={
            Props.SQL_ID: str(sql_id),
            "is_sub_select": is_sub_select,
            "id": current_sql_id,
        },
        section_type=section_type,
        edge_params=edge_params,
        parent=parent,
        is_full_parse=is_full_parse,
    )
    subquery_obj = Query(
        schemas=query_obj.get_schemas(),
        id=current_sql_id,
        q="",
        utterance="",
        ltimestamp=query_obj.get_latest_timestamp(),
        count=0,
        is_subselect=is_sub_select,
        sql_node=sql_node,
        direct_parent_query=query_obj,
        default_schema=query_obj.default_schema,
        dialect=query_obj.dialect,
    )
    query_obj.add_subselect(subquery_obj, current_sql_id, section_type)

    return current_sql_id, sql_node, subquery_obj


def build_query_obj(
    query_obj: Query,
    parsed_query: dict,
    keep_string_vals: bool,
    fks=None,
    is_full_parse=False,
):
    try:
        global keep_string_values
        keep_string_values = keep_string_vals
        _, _, _ = create_query(
            query_obj,
            parsed_query,
            parent=query_obj.get_sql_node(),
            sql_id=query_obj.get_id(),
            fks=fks,
            is_full_parse=is_full_parse,
        )
        if not is_full_parse:
            filter_edges_for_slim_parsing(query_obj)
    except MissingDataError as e:
        # trace = traceback.format_exc()
        # logger.error(
        #     f"Error in build_query_obj: {query_obj.string_query}\n {trace} \n{e}"
        # )
        raise e
    except Exception as e:
        # trace = traceback.format_exc()
        # logger.error(
        #     f"Error in build_query_obj: {query_obj.string_query}\n {trace} \n{e}"
        # )
        raise e


# in this function we filter all the edges that are not needed for the slim parsing (Join nodes, alias nodes, )
# so we remove wil
def filter_edges_for_slim_parsing(query_obj: Query):
    if len(query_obj.get_edges()) == 0:
        raise MissingDataError("A query without data (only constants). Skip it.")

    select_edges = query_obj.get_edges_dict()[SQL.SELECT].copy()
    select_edges.extend(query_obj.get_edges_by_section(Parser.SUBSELECT))
    get_wildcards_paths(query_obj, select_edges)
    edges_set: set[tuple[str, str]] = set()
    (
        filtered_edges,
        edges_set,
        connected_leafs_ids,
    ) = handle_columns_and_aliases(select_edges, root_id=str(query_obj.get_id()))
    all_edges = query_obj.get_edges()
    (
        all_edges_filtered,
        connected_leafs_ids,
        connected_columns_ids,
    ) = handle_tables_and_joins(
        all_edges, edges_set, connected_leafs_ids, root_id=str(query_obj.get_id())
    )

    if len(connected_leafs_ids) == 0 and len(connected_columns_ids) == 0:
        raise MissingDataError("A query without data (only constants). Skip it.")

    filtered_edges.extend(all_edges_filtered)
    query_obj.set_filtered_edges(filtered_edges)
    query_obj.set_reached_columns_ids(connected_columns_ids)


# TODO: write a comment that explains this intelligent bullshit
def get_wildcards_paths(
    query_obj: Query,
    edges: list[tuple[Node, Node, dict, str]],
    parent_wildcard: Optional[Node] = None,
):
    if not query_obj.get_wildcarded_tables():
        return
    edge_params = create_edge_params(
        optional_edge_params=None, sql_id=query_obj.get_root_query().get_id()
    )
    wildcard_node = Node(
        name=Parser.WILDCARD,
        label=Labels.CONSTANT,
        props=edge_params | {"name": Parser.WILDCARD},
    )

    if parent_wildcard is None:
        edges.append(
            (
                query_obj.sql_node,
                wildcard_node,
                edge_params,
            )
        )
    for table in query_obj.get_wildcarded_tables():
        if isinstance(table, Query):
            get_wildcards_paths(table, edges, wildcard_node)
        else:
            if parent_wildcard:
                edges.append((parent_wildcard, wildcard_node))
            edges.append((wildcard_node, table))
    return


def handle_columns_and_aliases(edges: list[tuple[Node, Node, dict, str]], root_id: str):
    if not edges:
        return [], set(), set()
    g = get_graph_from_edges(
        edges=edges,
        get_is_allowed_node=lambda node: node.label == Labels.SQL
        or (node.label == Labels.CONSTANT and node.name == Parser.WILDCARD)
        or (
            get_should_update(node.label, False, node.name)
            and not is_join(node.label, node.name)
        ),
    )
    edge_props = edges[0][2]
    filtered_edges, edges_set, connected_leafs_ids = filter_alias_chains(
        g=g,
        root_id=root_id,
        props=edge_props,
    )
    return filtered_edges, edges_set, connected_leafs_ids


def handle_tables_and_joins(
    edges: list[tuple[Node, Node, dict, str]],
    edges_set: set[tuple[str, str]],
    connected_leafs_ids: set[str],
    root_id: str,
):
    g = get_graph_from_edges(edges)
    edge_props = edges[0][2]
    (
        filtered_edges,
        connected_columns_ids,
    ) = connect_and_join_tables(
        g,
        root_id,
        edge_props,
        edges_set,
        connected_leafs_ids,
    )
    return filtered_edges, connected_leafs_ids, connected_columns_ids


## when receiving a graph with a branch such as (sql) -> (alias1) -> (alias2) -> (alias3) -> (column)
## we want to keep only the root alias (sql) -> (alias1) -> (column)
def filter_alias_chains(
    g: nx.DiGraph,
    root_id: str,
    props: dict,
):
    edges = []
    edges_set = set()
    connected_leafs_ids = set()
    nodes_to_remove = []
    allowed_leafs = [
        node
        for node in g.nodes
        if g.nodes[node]["node"].label
        in [
            Labels.COLUMN,
            Labels.TEMP_COLUMN,
            Labels.TABLE,
            Labels.TEMP_TABLE,
        ]
    ]
    for n in g.nodes:
        node_label = g.nodes[n]["node"].label
        node_name = g.nodes[n]["node"].name
        connected_to_root = g.has_edge(root_id, n)
        num_of_children = g.out_degree(n)
        if node_label not in [
            Labels.SQL,
            Labels.COLUMN,
            Labels.TEMP_COLUMN,
            Labels.TABLE,
            Labels.TEMP_TABLE,
        ]:
            if (
                (  ## keeping only the top level aliases, set op and wildcards
                    node_label in [Labels.ALIAS, Labels.SET_OP_COLUMN]
                    or node_name == Parser.WILDCARD
                )
                and (connected_to_root and g.in_degree(n) == 1)
                and num_of_children > 0
            ):
                has_allowed_leafs = any(
                    nx.has_path(g, n, col_id) for col_id in allowed_leafs
                )  # if the alias is not connected to any column, we want to remove it
                if has_allowed_leafs:
                    pass
                else:
                    nodes_to_remove.append(n)
            else:
                nodes_to_remove.append(n)
    g = remove_aliases_from_graph(g, set(nodes_to_remove))

    leaf_labels = [
        Labels.COLUMN,
        Labels.TEMP_COLUMN,
        Labels.TABLE,
        Labels.TEMP_TABLE,
    ]
    for edge in g.edges():
        if edge[0] != edge[1] and (edge[0], edge[1]) not in edges_set:
            edges_set.add((edge[0], edge[1]))
            edges.append([g.nodes[edge[0]]["node"], g.nodes[edge[1]]["node"], props])

            node_0 = g.nodes[edge[0]]["node"]
            node_1 = g.nodes[edge[1]]["node"]
            if node_0.label in leaf_labels and nx.has_path(g, root_id, node_0.id):
                connected_leafs_ids.add(node_0.id)
            if node_1.label in leaf_labels and nx.has_path(g, root_id, node_1.id):
                connected_leafs_ids.add(node_1.id)
    return edges, edges_set, connected_leafs_ids


## when receiving a graph with a branch such as
#               (sql)
#                 |
#                 v
# (table) <- (Join|Union) -> (table)
#               /   \
#        (column)    (column)
#
# we want to join the tables and connect them to the root, for a result like:
#                (sql)
#             / /      \ \
#            / /        \  \
#         /   /          \   \
#      /     /            \    \
# (col)  (t)<-join|union->(t) (col)


def connect_and_join_tables(
    g: nx.DiGraph,
    root_id: str,
    props: dict,
    edges_set: set[tuple[str, str]],
    connected_leafs_ids: set[str],
):
    edges = []
    tables_ids = []
    column_ids = []
    join_ids = []
    connected_columns_ids = []
    for n in g.nodes:
        if g.nodes[n]["node"].label in [Labels.TABLE, Labels.TEMP_TABLE]:
            tables_ids.append(n)
        elif g.nodes[n]["node"].label in [Labels.COLUMN, Labels.TEMP_COLUMN]:
            column_ids.append(n)
        elif is_join(g.nodes[n]["node"].label, g.nodes[n]["node"].props["name"]):
            join_ids.append(n)
    for join_id in join_ids:
        reached_leafs_ids = [
            leaf_id
            for leaf_id in connected_leafs_ids
            if nx.has_path(g, join_id, leaf_id)
        ]
        for leaf_id in reached_leafs_ids:
            if (
                root_id,
                leaf_id,
            ) not in edges_set and leaf_id not in connected_leafs_ids:
                edges.append(
                    (g.nodes[root_id]["node"], g.nodes[leaf_id]["node"], props)
                )
                edges_set.add((root_id, leaf_id))

        operators = g.nodes[join_id]["node"].props.get("operators", [])
        for operator in operators:
            table_id, other_table_id, join_operator = operator
            ## We sort the tables by ids to ensure consistent joins
            joined_tables_ids = sorted([table_id, other_table_id])
            try:
                edges.append(
                    (
                        g.nodes[joined_tables_ids[0]]["node"],
                        g.nodes[joined_tables_ids[1]]["node"],
                        {
                            Props.JOIN: join_operator,
                            Props.JOIN_SQL_ID: [props["sql_id"]],
                            **props,
                        },
                    )
                )
            except KeyError:
                logger.error(f"Couldn't join {join_operator}.")
    for table_column in tables_ids + column_ids:
        if (
            table_column not in connected_leafs_ids
            and (root_id, table_column) not in edges_set
        ):
            edges_set.add((root_id, table_column))
            connected_leafs_ids.add(table_column)
            edges.append(
                (g.nodes[root_id]["node"], g.nodes[table_column]["node"], props)
            )
        if g.nodes[table_column]["node"].label in [Labels.COLUMN, Labels.TEMP_COLUMN]:
            connected_columns_ids.append(table_column)

    return edges, connected_columns_ids


def get_graph_from_edges(
    edges: list[tuple[Node, Node, dict, str]],
    get_is_allowed_node: Callable[[Node], bool] = None,
):
    g = nx.DiGraph()
    for e in edges:
        ## Prevent infinite loops and circular relationships
        if isinstance(e[1], Query) or e[1].label == Labels.SQL:
            continue
        node_e0 = e[0].get_sql_node() if isinstance(e[0], Query) else e[0]
        node_e1 = e[1]
        if get_is_allowed_node and (
            (not get_is_allowed_node(node_e0)) or not get_is_allowed_node(node_e1)
        ):
            continue
        g.add_nodes_from(
            [(node_e0.id, {"node": node_e0}), (node_e1.id, {"node": node_e1})]
        )
        g.add_edge(node_e0.id, node_e1.id)
    return g


def change_to_output_dtype(function_name: str, arg_data_type: list[str] = []):
    if function_name in ["sum", "avg", "median", "nullifzero"]:
        return ["float"]
    if function_name in [
        "quarter",
        "hour",
        "bitwise_and",
        "date_diff",
        "dow",
        "to_unixtime",
        "row_number",
        "to_big_endian_64",
        "rank",
        "day_of_week",
        "length",
        "ceiling",
        "month",
        "week",
        "minute",
        "year",
        "floor",
        "datediff",
        "count",
        "dayofweek",
        "day",
        "regexp_count",
        "ln",
    ]:
        return ["int"]
    if function_name in [
        "coalesce",
        "nullif",
        "lag",
        "min",
        "max",
        "try",
        "index",
        "lead",
        "ListAgg",
        "contact_idin",
        "decode",
        "tungsten_opcodein",
        "greatest",
        "nvl",
        "date_add",
        "dateadd",
        "lower",
        "upper",
        "concat",
        "interval",
        "split_part",
        "replace",
        "regexp_replace",
        "substr",
        "reverse",
        "convert_timezone",
        "left",
        "right",
        "last_value",
        "first_value",
        "data_part",
    ]:
        return arg_data_type
    if function_name in ["round", "abs", "approx_percentile", "approx_distinct"]:
        return ["float"]
    if function_name in ["if", "regexp_like"]:
        return ["boolean"]
    if function_name in ["array_distinct", "array_agg"]:
        return arg_data_type
    if function_name in ["date_trunc", "current_timestamp", "sysdate"]:
        return ["timestamp"]
    if function_name in [
        "current_date",
        "to_hex",
        "array_join",
        "format_datetime",
        "extract",
        "regexp_extract",
        "date_format",
        "json_extract_path_text",
        "monthname",
    ]:
        return ["string"]
    if function_name in ["date"]:
        return ["date"]
    if function_name in ["from_unixtime", "date_parse", "getdate"]:
        return ["datetime"]
    if function_name in ["to_char"]:
        return ["varchar2"]
    if function_name in ["current_time"]:
        return ["time"]
    if function_name in ["to_timestamp"]:
        return ["timestamp"]
    if function_name in ["to_date"]:
        return ["date"]
    if function_name in ["to_number"]:
        return ["numeric", "try_to_numeric"]
    # logger.info(f"Unrecognized function. Function name: {function_name}")
    return []
