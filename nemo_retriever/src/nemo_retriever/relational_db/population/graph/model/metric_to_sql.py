import pandas as pd
from shared.graph.model.snippet import Snippet
from shared.graph.services.sql_snippet import get_all_joins
from shared.graph.dal.metrics_dal import (
    retrieve_metrics_subgraphs,
    connect_metric_sql_edges,
    get_snippets_of_bi_fields,
)
from shared.graph.utils import remove_redundant_parentheses
import networkx as nx
from shared.graph.parsers.sql.op_name_to_symbol import get_symbol
import logging
from shared.graph.model.reserved_words import unimportant_funcs_for_description
from shared.graph.model.node import clean_phrase
from shared.graph.model.metric import Metric
from shared.graph.cte.common import construct_sql_from_field

from shared.graph.dal.terms_dal import get_bt_name_by_attr_id
from functools import reduce


logger = logging.getLogger("metric_to_sql.py")


class MetricSQL:
    def __init__(self, id):
        self.id = id
        self.select_str = []
        self.select_tab_col_str = []
        self.where_tab_col_str = []
        self.where_str = []
        self.groupby_str = []
        self.groupby_tab_col_str = []
        self.over_str = []
        self.over_tab_col = []
        self.from_str = []
        self.tables = []

        self.source_select_str = []
        self.source_select_tab_col_str = []
        self.source_where_tab_col_str = []
        self.source_where_str = []
        self.source_groupby_str = []
        self.source_groupby_tab_col_str = []
        self.source_over_str = []
        self.source_over_tab_col = []
        self.source_from_str = []
        self.source_tables = []

    def add_to_field(self, field_name, to_add, source_to_add=None):
        if field_name is None:
            field_name = "select_str"
        if field_name == "select_str":
            self.select_str.append(to_add)
            self.select_tab_col_str.append(to_add)
            if source_to_add:
                self.source_select_str.append(source_to_add)
                self.source_select_tab_col_str.append(source_to_add)
        elif field_name == "where_str":
            self.where_str.append(to_add)
            self.where_tab_col_str.append(to_add)
            if source_to_add:
                self.source_where_str.append(source_to_add)
                self.source_where_tab_col_str.append(source_to_add)
        elif field_name == "groupby_str":
            self.groupby_str.append(to_add)
            self.groupby_tab_col_str.append(to_add)
            if source_to_add:
                self.source_groupby_str.append(source_to_add)
                self.source_groupby_tab_col_str.append(source_to_add)
        elif field_name == "over_str":
            self.over_str.append(to_add)
            self.over_tab_col.append(to_add)
            if source_to_add:
                self.source_over_str.append(source_to_add)
                self.source_over_tab_col.append(source_to_add)
        else:
            raise "Unknown field."

    def add_to_tc_field(self, field_name, to_add):
        if field_name is None:
            field_name = "select_str"
        if field_name == "select_str":
            self.select_tab_col_str = self.select_tab_col_str[:-1]
            self.select_tab_col_str.append(to_add)
        elif field_name == "where_str":
            self.where_tab_col_str = self.where_tab_col_str[:-1]
            self.where_tab_col_str.append(to_add)
        elif field_name == "groupby_str":
            self.groupby_tab_col_str = self.groupby_tab_col_str[:-1]
            self.groupby_tab_col_str.append(to_add)
        elif field_name == "over_str":
            self.over_tab_col = self.over_tab_col[:-1]
            self.over_tab_col.append(to_add)
        else:
            raise "Unknown field."

    def add_table(self, table_name_id, source_table_name_ids=None):
        if table_name_id not in self.tables:
            self.tables.append(table_name_id)
        if source_table_name_ids:
            for source_table_name_id in source_table_name_ids:
                if source_table_name_id not in self.source_tables:
                    self.source_tables.append(source_table_name_id)

    def get_select_str(self):
        return " ".join(self.select_str)

    def get_source_select_str(self):
        return " ".join(self.source_select_str)

    def get_select_tab_col_str(self):
        return " ".join(self.select_tab_col_str)

    def get_source_select_tab_col_str(self):
        return " ".join(self.source_select_tab_col_str)

    def get_where_str(self):
        return " ".join(self.where_str)

    def get_over_str(self):
        return " ".join(self.over_str)

    def get_over_tab_col_str(self):
        return " ".join(self.over_tab_col)

    def get_where_tab_col_str(self):
        return " ".join(self.where_tab_col_str)

    def get_groupby_str(self):
        return " ".join(self.groupby_str)

    def get_groupby_tab_col_str(self):
        return " ".join(self.groupby_tab_col_str)

    def get_from_str(self):
        return " ".join(self.from_str)

    def get_source_from_str(self):
        return " ".join(self.source_from_str)


def build_nx_metrics_graphs(account_id, metric_id):
    metrics = retrieve_metrics_subgraphs(account_id, metric_id)
    metrics_graphs = []

    for idx, func in metrics.iterrows():
        metrics_nx_graph = nx.DiGraph(name=func["name"] + "_" + str(idx))
        metrics_nx_graph.add_nodes_from([(x["id"], x) for x in func.nodes_props])
        metrics_nx_graph.add_edges_from(
            [(x[1][0]["id"], x[1][2]["id"], {"child_idx": x[0]}) for x in func.edges]
        )
        metrics_graphs.append(metrics_nx_graph)
    metrics["subgraphs"] = metrics_graphs
    return metrics


def construct_sql(metric_sql_obj: MetricSQL, account_id: str):
    sql_str = construct_sql_specific(
        metric_sql_obj.tables,
        metric_sql_obj.select_str,
        metric_sql_obj.select_tab_col_str,
        metric_sql_obj.from_str,
        metric_sql_obj.where_str,
        metric_sql_obj.where_tab_col_str,
        metric_sql_obj.groupby_str,
        metric_sql_obj.groupby_tab_col_str,
        account_id,
    )
    definition_sql_str = None
    if len(metric_sql_obj.source_tables) > 0:
        try:
            definition_sql_str = construct_sql_specific(
                metric_sql_obj.source_tables,
                metric_sql_obj.source_select_str,
                metric_sql_obj.source_select_tab_col_str,
                metric_sql_obj.source_from_str,
                metric_sql_obj.source_where_str,
                metric_sql_obj.source_where_tab_col_str,
                metric_sql_obj.source_groupby_str,
                metric_sql_obj.source_groupby_tab_col_str,
                account_id,
            )
        except Exception:
            logger.error(
                f"error in creating definition sql for metric: {metric_sql_obj.id}"
            )
    return sql_str, definition_sql_str


def construct_sql_specific(
    tables,
    select_list,
    select_tab_col_list,
    from_list,
    where_list,
    where_tab_col_list,
    groupby_list,
    groupby_tab_col_list,
    account_id,
):
    if len(tables) == 0:
        if len(where_list) > 0:
            raise Exception(
                f"when composing the sql {select_list} an empty 'from' tables list was recevied with a 'where' tables list: {where_list}"
            )
        else:
            select_str = " ".join(select_list)
            if len(from_list) > 0:
                from_str = " ".join(from_list)
                sql_str = f"SELECT {select_str} FROM {from_str}"
            else:
                sql_str = select_str
    else:
        groupby_str = ""
        where_str = ""
        from_str = tables[0]["name"]
        select_str = " ".join(select_list)
        if len(where_list) > 0:
            where_str = " ".join(where_list)
        if len(groupby_list) > 0:
            groupby_str = " ".join(groupby_list)
        if len(tables) > 1:
            tables_sorted_by_name = sorted(tables, key=lambda x: x["name"])
            tables_ids = [table["id"] for table in tables_sorted_by_name]
            from_str = get_all_joins(tables_ids, account_id)
            select_str = " ".join(select_tab_col_list)
        if len(where_tab_col_list) > 0:
            where_str = " ".join(where_tab_col_list)
        if len(groupby_tab_col_list) > 0:
            groupby_str = " ".join(groupby_tab_col_list)

        sql_str = f"SELECT {select_str} FROM {from_str}"
        if len(where_str) > 0:
            sql_str += f" WHERE {where_str}"
        if len(groupby_str) > 0:
            sql_str += f" GROUP BY {groupby_str}"
    return sql_str


def extract_aliased_expression(expr_str):
    as_index = expr_str.lower().find(" as ")
    if as_index == -1:
        return expr_str
    aliased_expr = expr_str[:as_index]
    return aliased_expr


def retrieve_alias_name(metric_sql):
    func_name = metric_sql.select_str[
        -4
    ]  # consider select str is [func_name, (, column name, )]
    col_name = metric_sql.select_str[-2]
    clean_col_name = "_".join(
        clean_phrase(col_name).split()
    )  # clean first and then write with _, shoul be no spaces between words, not compiling by sql
    return f"{func_name}_of_{clean_col_name}"


def traverse_graph(
    account_id: str,
    metric_snippets: dict[str, Snippet],  ##attrId: Snippet
    graph_f,
    current_node,
    parent: any,
    metric_sql: MetricSQL,
    field_name=None,
    simple_formula="",
    to_skip_for_simple=False,
):
    neibrs = [
        x
        for x in sorted(
            graph_f[current_node["id"]].items(), key=lambda edge: edge[1]["child_idx"]
        )
    ]

    if current_node["label"] in ["metric", "agg", "cond"]:
        for neibr in neibrs:
            metric_sql, simple_formula = traverse_graph(
                account_id,
                metric_snippets,
                graph_f,
                graph_f.nodes.get(neibr[0]),
                current_node,
                metric_sql,
                field_name,
                simple_formula,
            )

    elif current_node["label"] == "item":
        sub_metric_sql = MetricSQL("sub_item")

        # sub_metric_sql = MetricSQL(current_node['id'])
        for neibr in neibrs:
            if graph_f.nodes[neibr[0]]["label"] == "agg":
                sub_metric_sql, simple_formula = traverse_graph(
                    account_id,
                    metric_snippets,
                    graph_f,
                    graph_f.nodes.get(neibr[0]),
                    current_node,
                    sub_metric_sql,
                    field_name="select_str",
                    simple_formula=simple_formula,
                )
            elif graph_f.nodes[neibr[0]]["label"] == "cond":
                if graph_f.nodes[neibr[0]]["name"].lower() == "filter":
                    simple_formula += ", FILTER( "
                    sub_metric_sql, simple_formula = traverse_graph(
                        account_id,
                        metric_snippets,
                        graph_f,
                        graph_f.nodes.get(neibr[0]),
                        current_node,
                        sub_metric_sql,
                        field_name="where_str",
                        simple_formula=simple_formula,
                    )
                    simple_formula += ") "
                if graph_f.nodes[neibr[0]]["name"].lower() == "over":
                    sub_metric_sql, simple_formula = traverse_graph(
                        account_id,
                        metric_snippets,
                        graph_f,
                        graph_f.nodes.get(neibr[0]),
                        current_node,
                        sub_metric_sql,
                        field_name="over_str",
                        simple_formula=simple_formula,
                    )
                if graph_f.nodes[neibr[0]]["name"].lower() == "groupby":
                    sub_metric_sql, simple_formula = traverse_graph(
                        account_id,
                        metric_snippets,
                        graph_f,
                        graph_f.nodes.get(neibr[0]),
                        current_node,
                        sub_metric_sql,
                        field_name="groupby_str",
                        simple_formula=simple_formula,
                    )
            else:
                raise

        sub_sql_text, sub_source_sql_text = construct_sql(sub_metric_sql, account_id)
        sub_sql_text = "(" + sub_sql_text + ")"
        if sub_source_sql_text:
            sub_source_sql_text = "(" + sub_source_sql_text + ")"
        if "nested_aggr_for_groupby" in parent and parent["nested_aggr_for_groupby"]:
            alias = retrieve_alias_name(sub_metric_sql)
            over_str = ""
            if len(sub_metric_sql.over_str) > 0:
                over_str = sub_metric_sql.get_over_str()
                over_tab_col = sub_metric_sql.get_over_tab_col_str()
            sub_metric_sql.add_to_field(
                "select_str", f"OVER (PARTITION BY {over_str}) as {alias}"
            )  # consider only one column to select and only one alias a
            sub_metric_sql.add_to_tc_field(
                "select_str", f"OVER (PARTITION BY {over_tab_col}) as {alias}"
            )
            sub_sql_text, sub_source_sql_text = construct_sql(
                sub_metric_sql, account_id
            )
            sub_sql_text = "(" + sub_sql_text + ")"
            sub_source_sql_text = "(" + sub_source_sql_text + ")"
            metric_sql.select_str.append(f"{alias}")
            metric_sql.source_select_str.append(f"{alias}")
            metric_sql.from_str.append(sub_sql_text)
            metric_sql.source_from_str.append(sub_source_sql_text)

        else:
            metric_sql.select_str.append(sub_sql_text)
            metric_sql.select_tab_col_str.append(sub_sql_text)
            metric_sql.source_select_str.append(sub_source_sql_text)
            metric_sql.source_select_tab_col_str.append(sub_source_sql_text)

    elif current_node["label"] == "function" and current_node["name"] != "if":
        args_to_skip = []
        metric_sql.add_to_field(field_name, current_node["name"], current_node["name"])
        metric_sql.add_to_field(field_name, "(", "(")
        if current_node["name"] not in unimportant_funcs_for_description:
            simple_formula += f"{current_node['name']} ("
        else:
            args_to_skip = unimportant_funcs_for_description[current_node["name"]]
        if "distinct" in current_node:
            metric_sql.add_to_field(field_name, "distinct", "distinct")
            simple_formula += "distinct "
        if len(neibrs) == 0:  # no arguments, add )
            metric_sql.add_to_field(field_name, ")", ")")
            if current_node["name"] not in unimportant_funcs_for_description:
                simple_formula += " )"

        else:
            for idx, neibr in enumerate(neibrs):
                metric_sql, simple_formula = traverse_graph(
                    account_id,
                    metric_snippets,
                    graph_f,
                    graph_f.nodes.get(neibr[0]),
                    current_node,
                    metric_sql,
                    field_name,
                    simple_formula,
                    to_skip_for_simple=True if idx in args_to_skip else False,
                )
                if idx < len(neibrs) - 1:
                    metric_sql.add_to_field(field_name, ", ", ", ")
                    if len(args_to_skip) == 0:
                        simple_formula += ", "
            metric_sql.add_to_field(field_name, ")", ")")
            if current_node["name"] not in unimportant_funcs_for_description:
                simple_formula += " )"

    elif current_node["label"] == "operator" or (
        current_node["label"] == "function" and current_node["name"] == "if"
    ):
        if current_node["name"] == "Case" or current_node["name"] == "if":
            metric_sql.add_to_field(field_name, "CASE ", "CASE ")
            simple_formula += " CASE "
            for neibr in neibrs:
                if "cond" in neibr[1]["child_idx"]:
                    metric_sql.add_to_field(field_name, "WHEN ", "WHEN ")
                    simple_formula += " WHEN "
                    metric_sql, simple_formula = traverse_graph(
                        account_id,
                        metric_snippets,
                        graph_f,
                        graph_f.nodes.get(neibr[0]),
                        current_node,
                        metric_sql,
                        field_name,
                        simple_formula,
                    )
                elif "res" in neibr[1]["child_idx"]:
                    metric_sql.add_to_field(field_name, "THEN ", "THEN ")
                    simple_formula += " THEN "
                    metric_sql, simple_formula = traverse_graph(
                        account_id,
                        metric_snippets,
                        graph_f,
                        graph_f.nodes.get(neibr[0]),
                        current_node,
                        metric_sql,
                        field_name,
                        simple_formula,
                    )
                elif "else" in neibr[1]["child_idx"]:
                    metric_sql.add_to_field(field_name, "ELSE ", "ELSE ")
                    simple_formula += " ELSE "
                    metric_sql, simple_formula = traverse_graph(
                        account_id,
                        metric_snippets,
                        graph_f,
                        graph_f.nodes.get(neibr[0]),
                        current_node,
                        metric_sql,
                        field_name,
                        simple_formula,
                    )
                else:
                    raise "case smth not supported."
            metric_sql.add_to_field(field_name, "END ", "END ")
            simple_formula += " END "

        elif current_node["name"] == "Between":
            for neibr in neibrs:
                if neibr[1]["child_idx"] == 0:
                    metric_sql, simple_formula = traverse_graph(
                        account_id,
                        metric_snippets,
                        graph_f,
                        graph_f.nodes.get(neibr[0]),
                        current_node,
                        metric_sql,
                        field_name,
                        simple_formula,
                    )
                elif neibr[1]["child_idx"] == 1:
                    metric_sql.add_to_field(field_name, "BETWEEN ", "BETWEEN ")
                    simple_formula += " BETWEEN "
                    metric_sql, simple_formula = traverse_graph(
                        account_id,
                        metric_snippets,
                        graph_f,
                        graph_f.nodes.get(neibr[0]),
                        current_node,
                        metric_sql,
                        field_name,
                        simple_formula,
                    )
                elif neibr[1]["child_idx"] == 2:
                    metric_sql.add_to_field(field_name, "AND ", "AND ")
                    simple_formula += " AND "
                    metric_sql, simple_formula = traverse_graph(
                        account_id,
                        metric_snippets,
                        graph_f,
                        graph_f.nodes.get(neibr[0]),
                        current_node,
                        metric_sql,
                        field_name,
                        simple_formula,
                    )
                else:
                    raise "between smth not supported."

        elif current_node["name"] == "Interval":
            metric_sql.add_to_field(field_name, "Interval ", "Interval ")
            simple_formula += "Interval "
            for neibr in neibrs:
                metric_sql, simple_formula = traverse_graph(
                    account_id,
                    metric_snippets,
                    graph_f,
                    graph_f.nodes.get(neibr[0]),
                    current_node,
                    metric_sql,
                    field_name,
                    simple_formula,
                )

        elif len(neibrs) == 1:
            symbol, _ = get_symbol(current_node["name"])
            if symbol.lower() in ["is null", "is not null"]:
                metric_sql, simple_formula = traverse_graph(
                    account_id,
                    metric_snippets,
                    graph_f,
                    graph_f.nodes.get(neibrs[0][0]),
                    current_node,
                    metric_sql,
                    field_name,
                    simple_formula,
                )
                metric_sql.add_to_field(field_name, symbol, symbol)
                simple_formula += f"{symbol} "
            else:
                metric_sql.add_to_field(field_name, symbol, symbol)
                simple_formula += f"{symbol} "
                metric_sql, simple_formula = traverse_graph(
                    account_id,
                    metric_snippets,
                    graph_f,
                    graph_f.nodes.get(neibrs[0][0]),
                    current_node,
                    metric_sql,
                    field_name,
                    simple_formula,
                )

        elif len(neibrs) == 2:
            symbol, _ = get_symbol(current_node["name"])
            if symbol in ["/", "*", "-", "+"]:
                metric_sql.add_to_field(field_name, "(", "(")
                simple_formula += "("
                if symbol == "/":
                    metric_sql.add_to_field(field_name, "cast(", "cast(")

            metric_sql, simple_formula = traverse_graph(
                account_id,
                metric_snippets,
                graph_f,
                graph_f.nodes.get(neibrs[0][0]),
                current_node,
                metric_sql,
                field_name,
                simple_formula,
            )
            if symbol == "/":
                metric_sql.add_to_field(field_name, "as float)", "as float)")

            metric_sql.add_to_field(field_name, symbol, symbol)
            simple_formula += f" {symbol} "
            metric_sql, simple_formula = traverse_graph(
                account_id,
                metric_snippets,
                graph_f,
                graph_f.nodes.get(neibrs[1][0]),
                current_node,
                metric_sql,
                field_name,
                simple_formula,
            )
            if symbol in ["/", "*", "-", "+"]:
                metric_sql.add_to_field(field_name, ")", ")")
                simple_formula += ")"

        else:
            logger.error("Not supportable number of neighbors in operator")
            raise

    elif current_node["label"] == "attribute":
        snippet = metric_snippets[current_node["id"]]
        try:
            for snippet_table_name_id in snippet.tables_names_ids:
                metric_sql.add_table(
                    snippet_table_name_id, snippet.source_table_name_ids
                )
        except Exception as e:
            logger.error(f"Error in adding table to metric sql: {e}")
            logger.error(e)
        metric_sql.add_to_field(
            field_name,
            extract_aliased_expression(snippet.props["snippet_select"]),
            snippet.props["source_expr"],
        )
        if len(snippet.tables_names_ids) == 1:
            metric_sql.add_to_tc_field(
                field_name,
                snippet.tables_names_ids[0]["name"]
                + "."
                + snippet.props["snippet_select"],
            )
        term_name = get_bt_name_by_attr_id(current_node["id"], account_id)
        simple_formula += f"[{term_name}.{current_node['name']}] "

    elif current_node["label"] in ["constant"]:
        if current_node["data_type"] in ["SingleQuotedString"]:
            metric_sql.add_to_field(
                field_name,
                f"'{str(current_node['name'])}'",
                f"'{str(current_node['name'])}'",
            )
            if not to_skip_for_simple:
                simple_formula += f"'{str(current_node['name'])}'"
        else:
            metric_sql.add_to_field(
                field_name, str(current_node["name"]), str(current_node["name"])
            )
            if not to_skip_for_simple:
                simple_formula += str(current_node["name"])

    else:
        logger.error(
            f"Not supportable label {graph_f.nodes.get(current_node)['label']}"
        )
        raise

    return metric_sql, simple_formula


def build_sqls_from_metric(
    account_id: str, metric_obj: Metric, from_bi_field: str = None
):
    metric_id = str(metric_obj.id)
    try:
        metric_graph = build_nx_metrics_graphs(account_id, metric_id).iloc[0]
        metrics_sqls = []
        combinations: list[dict[str, Snippet]] = metric_obj.snippets_df.to_dict(
            orient="records"
        )
        for combination in combinations:
            sql, definition_sql = build_sql_from_metric(
                account_id, metric_id, metric_graph, combination
            )
            snippets = list(combination.values())
            tables_ids = reduce(
                lambda tables, current_snippet: list(
                    set(tables + current_snippet.tables_ids)
                ),
                snippets,
                [],
            )
            metrics_sqls.append(
                {
                    "sql": sql,
                    "definition_sql": definition_sql,
                    "snippets": [snippet.id for snippet in snippets],
                    "tables_ids": tables_ids,
                }
            )
        metric_attributes_ids = metric_obj.snippets_df.keys().to_list()
        for metric_sql in metrics_sqls:
            metric_sql_id = connect_metric_sql_edges(
                account_id, metric_id, metric_sql, metric_attributes_ids
            )
            metric_sql["snippet_id"] = metric_sql_id
        return metrics_sqls
    except Exception as error:
        raise error


def build_sql_from_metric(
    account_id: str,
    metric_id: str,
    metric_graph: pd.DataFrame,
    metric_snippets: dict[str, Snippet],
):
    metric_sql = MetricSQL(metric_id)
    metric_sql, _ = traverse_graph(
        account_id,
        metric_snippets,
        metric_graph["subgraphs"],
        list(metric_graph.nodes)[0],
        None,
        metric_sql,
        simple_formula="",
    )
    try:
        final_sql, definition_final_sql = construct_sql(metric_sql, account_id)
    except Exception as e:
        logger.error(f"[{metric_graph['name']}]: {e}")
        raise e

    final_sql = remove_redundant_parentheses(final_sql).strip()
    if definition_final_sql:
        definition_final_sql = remove_redundant_parentheses(
            definition_final_sql
        ).strip()
    return final_sql, definition_final_sql


def build_sql_from_bi_fields(
    account_id: str, metric_obj: Metric, from_bi_fields: list[str], snippet_id: str
):
    try:
        metric_id = str(metric_obj.id)
        sql = construct_sql_from_field(account_id, field_id=from_bi_fields[0])
        snippets_ids, attributes_ids = get_snippets_of_bi_fields(
            account_id, from_bi_fields, snippet_id
        )
        metric_sql_id = connect_metric_sql_edges(
            account_id,
            metric_id,
            {"snippets": snippets_ids, "sql": sql},
            attributes_ids,
        )
        return metric_sql_id
    except Exception as error:
        raise f"Error in building sql from bi field in metric {metric_id} , error:{str(error)}"
