import networkx as nx
import logging
import pandas as pd
from infra.Neo4jConnection import get_neo4j_conn
from collections import ChainMap
import ast
from shared.graph.parsers import sql

from shared.graph.utils import flat_list_recursive

conn = get_neo4j_conn()

logger = logging.getLogger("metrics_dal.py")

sql_math_functions = [
    "abs",
    "acos",
    "asin",
    "atan",
    "atn2",
    "avg",
    "ceiling",
    "count",
    "cos",
    "cot",
    "degrees",
    "exp",
    "floor",
    "log",
    "log10",
    "max",
    "min",
    "pi",
    "power",
    "radians",
    "rand",
    "round",
    "sign",
    "sin",
    "sqrt",
    "square",
    "sum",
    "tan",
]


# def set_conditions(conds: list):
#     conds = [i.replace('where','') for i in conds if i]
#     conds = sorted(conds, key=len)
#     full_list = []
#     for cond in conds:
#         if not any(cond in s for s in full_list):
#             full_list.append(cond)
#     return  ' and '.join(full_list)


# def group_kpis(formulas_trees,df):
#     sim_dict = {g.name: [] for g in formulas_trees}
#     for i, g in enumerate(tqdm(formulas_trees)):
#         for j, g2 in enumerate(formulas_trees):
#             if g != g2 and i < j:
#                 if [g.nodes[e[1]]['name'] for e in g.edges(['start'])] == [g2.nodes[e[1]]['name'] for e in g2.edges(['start'])] and\
#                         abs((len(g.nodes()) - len(g2.nodes()))) < 2 and check_same_length_conditions(df,i,j):
#                     score = nx.graph_edit_distance(g, g2, node_match=lambda a, b: a['name'] == b['name'], roots=('start', 'start'), timeout=5)
#                     if score < 3:
#                         sim_dict[g.name].append(g2.name)
#                         sim_dict[g2.name].append(g.name)
#
#     # new_dict = [{'formula': k, 'similar': v} for k, v in sim_dict.items()]
#     return sim_dict


# def check_same_length_conditions(df,i,j):
#     # return true if both conditions have same length
#     first=df.iloc[i]['conditions']
#     second=df.iloc[j]['conditions']
#     return len(first.split('and')) == len(second.split('and'))

#
# def check_same_conjuctions(df,i,j):
#     # return true if both conditions compose of same sub-conditions
#     first=df.iloc[i]['conditions']
#     second=df.iloc[j]['conditions']
#     return set(first.split('and')) == set(second.split('and'))
#
#
# def check_sub_conditions(df,i,j):
#     # check if conditions are subsets
#     first=df.iloc[i]['conditions'].split('and')
#     second=df.iloc[j]['conditions'].split('and')
#     return set(first).issubset(second) or set(first).issuperset(second)


# def check_differnce_jaccard_conditions(df,i,j):
#     #estimate the diffrence by 1-jaccard distance, return true if it is less than 0.3*set size. (0.3 is completely hyper parameter)
#     first=set(df.iloc[i]['conditions'].split('and'))
#     second=set(df.iloc[j]['conditions'].split('and'))
#     return (1-(len(first.intersection(second))/len(first.union(second)))) < 0.3*min(len(first),len(second))
#
# def check_out_of_set_not_appear(df,i,j):
#     # calculate the difference and check if the columns belong there appear with different context in the second where clause
#     opp=['=','<=''>=','<>','<','>']
#     first=set(df.iloc[i]['conditions'].split('and'))
#     second=set(df.iloc[j]['conditions'].split('and'))
#     short= copy.deepcopy(first) if len(first)<len(second) else copy.deepcopy(second)
#     long=copy.deepcopy(first) if len(first)>=len(second) else copy.deepcopy(second)
#     diff=short.difference(long)
#     diff2=long.difference(short)
#     if len(diff)==0:
#         return True # in case they are the same
#     diff=list(diff)
#     diff2=list(diff2)
#     for item in diff:
#         for op in opp:
#             if item.find(op)!=-1:
#                 item=item[:item.find(op)]
#                 for item2 in diff2:
#                     if item2.find(item)!=-1: #found same column with different condition
#                         return False
#     return True
def clean_functions(df):
    # df['conditions'] = df['conditions'].apply(lambda x: set_conditions(x))
    df["conditions"] = df["conditions"].apply(lambda x: x.replace("where", ""))
    df = (
        df.groupby(
            [
                "formula",
                df["conditions"].astype(str),
                "id",
                "main_sql_text",
                "counter",
                df["f_sub_sql"].astype(str),
            ]
        )
        .agg({"terms_attrs": list, "sub_terms_attrs": list})
        .reset_index()
        .sort_values("counter", ascending=False)
    )
    df = (
        df.groupby(
            [
                "formula",
                "conditions",
                df["terms_attrs"].astype(str),
                df["sub_terms_attrs"].astype(str),
                "f_sub_sql",
            ]
        )
        .agg({"main_sql_text": list, "id": list, "counter": "sum"})
        .reset_index()
        .sort_values("counter", ascending=False)
    )
    df = df.loc[df.formula != "count(*)"]
    df = df.loc[
        ~(
            (df.conditions.str.lower().str.contains("over"))
            | (df.conditions.str.lower().str.contains("group by"))
        )
    ].copy()  # Currently not supported
    df = df.loc[
        df.formula.apply(
            lambda x: any(
                f"{math_func}(" in x.replace(" ", "")
                for math_func in sql_math_functions
            )
        )
    ]
    df["terms_attrs"] = df["terms_attrs"].apply(
        lambda x: flat_list_recursive(ast.literal_eval(x))
    )
    df["sub_terms_attrs"] = df["sub_terms_attrs"].apply(
        lambda x: flat_list_recursive(ast.literal_eval(x))
    )
    df["f_sub_sql"] = df["f_sub_sql"].apply(
        lambda x: flat_list_recursive(ast.literal_eval(x))
    )
    return df


def replace_col_name(x, names_dict):
    for col, bt_attr in names_dict.items():
        try:
            if col.lower() in x.lower():
                x = x.replace(col, "[" + bt_attr + "]")
        except Exception as e:
            raise e
    return x


def remove_alias(text):
    # remove the first word after as
    to_return = []
    text_spl = text.split(" as ")
    if len(text_spl) == 1:
        return text
    else:
        to_return.append(text_spl[0])
        for idx, item in enumerate(text_spl):
            if idx > 0:
                item_spl = item.split(" ", 1)
                del item_spl[0]
                if len(item_spl) > 0 and len(item_spl[0]) > 0:
                    to_return.append(item_spl[0])
    return " ".join(to_return)


def process_over_groupby(x, account_id):
    # only agg function on group by result of the below form is supported
    # example: sum(avg(HumanResources.EmployeePayHistory.Rate) over (partition by HumanResources.vEmployeeDepartment.BusinessEntityID))
    # The above sql will not compile, but ok as metric function
    # The legal sql is : select sum(avg_rate) from (select avg(eph.rate) over (partition by v_dep.BusinessEntityID) as avg_rate from ...
    query = """
            MATCH (f{account_id:$account_id, id:$id})
            MATCH (f)-[:SQL]->(a: alias)-[:SQL]->(sec_func:function)
            RETURN coalesce(f.name) as main_name, coalesce(sec_func.expr_str, '') as formula

            """
    formula_data = conn.query_read_only(
        query=query, parameters={"account_id": account_id, "id": x["id"][0]}
    )
    if len(formula_data) > 0:
        formula = formula_data[0]["formula"]
        main_name = formula_data[0]["main_name"]
        if "over" in formula.lower():
            formula = formula.replace("over", ",over")
            formula = formula.replace("order by", "")
            formula = formula.replace("partition by", "")  # somehow save the over type
            formula = f"{main_name}(({formula}))"
        if "group by" in formula.lower():
            formula = formula.replace("group by", ",groupby")
            formula = f"{main_name}(({formula}))"
    else:
        formula = ""
    return formula


def process_sub_sql(x, account_id):
    query = """
                MATCH (f{account_id:$account_id, id:$id})
                CALL apoc.path.subgraphAll(f,{
                    relationshipFilter: ">SQL",
                    labelFilter: "",
                    minLevel: 0})
                YIELD nodes, relationships
                RETURN nodes, [n IN nodes | apoc.map.setKey(properties(n), "label", labels(n)[0])] as nodes_props, 
                [p IN relationships | [coalesce(properties(p).child_idx,0),p]] as edges
            """
    func_subgraph = conn.query_read_only(
        query=query, parameters={"account_id": account_id, "id": x["id"][0]}
    )[0]
    func_nx_graph = nx.DiGraph(name=x["formula"])
    func_nx_graph.add_nodes_from([(y["id"], y) for y in func_subgraph["nodes_props"]])
    func_nx_graph.add_edges_from(
        [
            (y[1][0]["id"], y[1][2]["id"], {"child_idx": y[0]})
            for y in func_subgraph["edges"]
        ]
    )
    text_formula, _ = traverse_graph(
        func_nx_graph,
        start_node=list(func_nx_graph.nodes)[0],
        parent=None,
        text_formula=[],
    )
    return "".join(text_formula)


def get_kpi_recommendation(account_id):
    query = """ match(f:function|operator{account_id:$account_id})
                WHERE ((f:function) or (f:operator and f.name in ['Minus', 'Plus', 'Divide', 'Multiply'])) 
                CALL apoc.path.subgraphNodes(f,{
                relationshipFilter: ">SQL",
                labelFilter: "/column",     
                minLevel: 0})
                YIELD node as colfunc
                WHERE coalesce(colfunc.deleted, false) = false
                
                WITH f, colfunc
                MATCH(colfunc)<-[:attr_of|reaching]-(attr:attribute)<-[:term_of]-(bt:term)
                MATCH(colfunc)<-[:schema]-(tab:table)<-[:schema]-(sch:schema)
                WHERE coalesce(tab.deleted, false) = false and coalesce(sch.deleted, false) = false

                CALL apoc.path.subgraphNodes(f,{
                relationshipFilter: "<SQL",
                labelFilter: "/sql|-table|-function|-operator",     
                minLevel: 0})
                YIELD node as minisqlnode

                CALL apoc.path.subgraphNodes(f,{
                relationshipFilter: "<SQL",
                labelFilter:">sql|-table",
                minLevel: 0})
                YIELD node as sqlnode
                WHERE not sqlnode.is_sub_select
                
                call (f){
                    CALL apoc.path.subgraphNodes(f,{
                    relationshipFilter: "SQL>",
                    labelFilter:">sql|-column|-table"
                    })
                    YIELD node as f_sub_sql
                    WHERE f_sub_sql.is_sub_select
                    RETURN collect(distinct f_sub_sql.id) as f_sub_sql
                
                }
          
                OPTIONAL MATCH (minisqlnode) -[:SQL]->(:command{name:"Select"})-[:SQL]->(wherenode:command{name:"Where"}) 
                WITH case when wherenode is null then true else false end as where_not_exist,f, sqlnode, minisqlnode, wherenode, sch, tab, colfunc, bt, attr , f_sub_sql
   
                CALL apoc.do.case([
                       where_not_exist,

                      'RETURN sqlnode.sql_full_query as main_sql_text, sqlnode.id as main_sql, f.sql_id as sql_id,f.id as id, f.expr_str as formula, 
                       sum(distinct sqlnode.total_counter) as counter, 
                       collect(distinct {col_name: sch.name+"."+tab.name+"."+colfunc.name, bt_attr: bt.name+"."+attr.name}) as terms_attrs,  
                       [] as sub_terms_attrs, "" as conditions, f_sub_sql'
                      ],
                      '
                      CALL apoc.path.subgraphNodes(wherenode,{
                        relationshipFilter: ">SQL",
                        labelFilter:"/column",
                        minLevel:0})
                        YIELD node as minicolnode
                        WHERE coalesce(minicolnode.deleted, false) = false

                      MATCH(minicolnode)<-[:attr_of]-(attr_sub:attribute)<-[:term_of]-(bt_sub:term)
                      MATCH(minicolnode)<-[:schema]-(sub_tab:table)<-[:schema]-(sub_sch:schema)
                      WHERE coalesce(sub_tab.deleted, false) = false and coalesce(sub_sch.deleted, false) = false

                      RETURN sqlnode.sql_full_query as main_sql_text, sqlnode.id as main_sql, f.sql_id as sql_id,
                        f.id as id, f.expr_str as formula, sum(distinct sqlnode.total_counter) as counter, 
                        collect(distinct {col_name: sch.name+"."+tab.name+"."+colfunc.name, bt_attr: bt.name+"."+attr.name}) as terms_attrs,  
                        collect(distinct {col_name: sub_sch.name+"."+sub_tab.name+"."+minicolnode.name, bt_attr: bt_sub.name+"."+attr_sub.name}) as sub_terms_attrs, 
                        wherenode.expr_str as conditions, f_sub_sql
                      ',
                      {where_not_exist:where_not_exist, sqlnode:sqlnode, f:f, sch:sch, tab:tab, colfunc:colfunc, bt:bt, attr:attr, wherenode:wherenode, f_sub_sql:f_sub_sql})
                   yield value 

                RETURN value.main_sql_text as main_sql_text, value.main_sql as main_sql, value.sql_id as sql_id,value.id as id, value.formula as formula, 
                      value.counter as counter, value.terms_attrs as terms_attrs, 
                      value.sub_terms_attrs as sub_terms_attrs, value.conditions as conditions, value.f_sub_sql as f_sub_sql
                
                """
    funcs_counter = pd.DataFrame(
        conn.query_write(query=query, parameters={"account_id": account_id})
    )
    if funcs_counter.size > 0:
        funcs_counter_clean = clean_functions(funcs_counter).reset_index(drop=True)
        terms_attrs_dict = dict(
            ChainMap(
                *[
                    {x["col_name"]: x["bt_attr"]}
                    for y in list(funcs_counter_clean["terms_attrs"])
                    + list(funcs_counter_clean["sub_terms_attrs"])
                    for x in y
                ]
            )
        )

        if funcs_counter_clean.size > 0:
            funcs_counter_clean["formula"] = funcs_counter_clean.apply(
                lambda x: (
                    process_sub_sql(x, account_id)
                    if len(x.f_sub_sql) > 0
                    else x.formula
                ),
                axis=1,
            )
            funcs_counter_clean["formula"] = funcs_counter_clean.apply(
                lambda x: (
                    process_over_groupby(x, account_id)
                    if ("over" in x.formula.lower() or "group by" in x.formula.lower())
                    else x.formula
                ),
                axis=1,
            )
            funcs_counter_clean = funcs_counter_clean.loc[
                funcs_counter_clean.formula != ""
            ]

            funcs_counter_clean["formula_replaced"] = (
                funcs_counter_clean["formula"]
                .apply(lambda x: replace_col_name(x, terms_attrs_dict))
                .copy()
            )
            funcs_counter_clean["conds_replaced"] = (
                funcs_counter_clean["conditions"]
                .apply(lambda x: replace_col_name(x, terms_attrs_dict))
                .copy()
            )

            funcs_counter_clean["formula_to_kpi"] = funcs_counter_clean.apply(
                lambda x: (
                    "(" + x.formula_replaced + " ,FILTER (" + x.conds_replaced + "))"
                    if len(x.conds_replaced) > 0
                    else x.formula_replaced
                ),
                axis=1,
            ).copy()

            # kpi_similar_groups = group_kpis(formulas_trees, funcs_counter_clean)
            # funcs_counter_clean['similar_to_formula'] = funcs_counter_clean['formula'].apply(
            #     lambda x: kpi_similar_groups[x] if x in kpi_similar_groups else [])
            funcs_counter_clean["recommended_kpi"] = funcs_counter_clean[
                "formula_to_kpi"
            ]
            return (
                funcs_counter_clean.drop_duplicates(
                    subset="recommended_kpi", keep="first"
                )
                .dropna(subset="recommended_kpi")
                .reset_index(drop=True)
            )

    return pd.DataFrame()


def add_filter(text_formula, filter_str):
    if len(filter_str) > 0:
        idx = text_formula.index("possible_filter_start")
        text_formula.insert(idx, "(")
        del text_formula[idx + 1]
        text_formula.append(f", FILTER ({filter_str.strip()}))")
    else:
        idx = text_formula.index("possible_filter_start")
        del text_formula[idx]
    return text_formula


def traverse_graph(graph_f, start_node, parent, text_formula=[], start_agg=False):
    neibrs = [
        x[0]
        for x in sorted(
            graph_f[start_node].items(), key=lambda edge: edge[1]["child_idx"]
        )
    ]
    filter_str = ""

    if graph_f.nodes.get(start_node)["label"] in ["function"]:
        text_formula.append("possible_filter_start")
        text_formula.append(graph_f.nodes.get(start_node)["name"])
        text_formula.append("(")

        for neibr in neibrs:
            text_formula, filter = traverse_graph(
                graph_f, neibr, start_node, text_formula, start_agg=True
            )
        text_formula = add_filter(text_formula, filter)  # TODO think about solution
        text_formula.append(")")

    elif graph_f.nodes.get(start_node)["label"] in ["operator", "command"]:
        if graph_f.nodes.get(start_node)["name"] == "Case":
            expr_str = graph_f.nodes.get(start_node)["expr_str"]
            if "select" in expr_str.lower():
                raise "Select in case, currently not implemented"
            text_formula.append(expr_str)

        elif len(neibrs) == 2:
            text_formula.append("possible_filter_start")
            text_formula.append("(")
            text_formula, filter_str = traverse_graph(
                graph_f, neibrs[0], start_node, text_formula
            )
            text_formula = add_filter(text_formula, filter_str)
            text_formula.append(")")
            text_formula.append(
                sql.op_name_to_symbol.get_symbol(graph_f.nodes.get(start_node)["name"])
            )

            text_formula.append("possible_filter_start")
            text_formula.append("(")
            text_formula, filter_str = traverse_graph(
                graph_f, neibrs[1], start_node, text_formula
            )
            text_formula = add_filter(text_formula, filter_str)
            text_formula.append(")")

        else:
            logger.error("Not supportable number of neighbors in Operator")
            raise

    elif graph_f.nodes.get(start_node)["label"] in ["sql"]:
        select_str = graph_f.nodes.get(start_node)["select_str"]
        select_str = remove_alias(select_str)
        text_formula.append(select_str)
        filter_str = graph_f.nodes.get(start_node)["where_str"]
        filter_str = remove_alias(filter_str)

    elif graph_f.nodes.get(start_node)["label"] in [
        "table",
        "column",
        "alias",
        "set_op_column",
    ]:
        text_formula.append(graph_f.nodes.get(start_node)["name"])

    elif graph_f.nodes.get(start_node)["label"] in ["constant"]:
        text_formula.append(graph_f.nodes.get(start_node)["expr_str"])

    else:
        logger.error(f"Not supportable label {graph_f.nodes.get(start_node)['label']}")
        raise

    return text_formula, filter_str
