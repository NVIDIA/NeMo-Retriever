from shared.graph.model.node import Node
from shared.graph.model.reserved_words import Labels, SQLType, SQL, Props, Parser
from typing import Union, Type, Self
import pendulum

PRESERVED_LABELS = [
    Labels.ALIAS,
    Labels.SET_OP_COLUMN,
    Labels.COLUMN,
    Labels.TABLE,
    Labels.TEMP_COLUMN,
    Labels.TEMP_TABLE,
]
PRESERVED_COMMANDS = ["Joins"]


class UnsupportedQueryError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class MissingDataError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class NotValidSyntaxError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class NotSelectSqlTypeError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class NoFKError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class NestedPropertyError(Exception):
    def __init__(self, message):
        self.message = f"{message}\nNeo4j allowed properties: BOOLEAN, DATE, DURATION, FLOAT, INTEGER, LIST, LOCAL DATETIME, LOCAL TIME, POINT, STRING, ZONED DATETIME, and ZONED TIME"
        super().__init__(self.message)


def get_sql_counters(sql_node):
    total_counter = sql_node.get_properties()["total_counter"]
    cnt_per_month = {}
    for prop_key, prop_val in sql_node.get_properties().items():
        if prop_key.startswith("cnt_"):
            cnt_per_month.update({prop_key: prop_val})
    return total_counter, cnt_per_month


# explicit function to flatten a
# nested list
def flatten_list(nestedList):
    # check if list is empty
    if not (bool(nestedList)):
        return nestedList

    # to check instance of list is empty or not
    if isinstance(nestedList[0], list):
        # call function with sublist as argument
        return flatten_list(*nestedList[:1]) + flatten_list(nestedList[1:])

    # call function with sublist as argument
    return nestedList[:1] + flatten_list(nestedList[1:])


def get_tables_recursive(query_tables):
    tables = [t[1] for t in query_tables if isinstance(t[1], Node)]
    tables_from_subqueries = [
        get_tables_recursive(sub_query[1].tables)
        for sub_query in query_tables
        if isinstance(sub_query[1], Query)
    ]

    return list(set(tables + flatten_list(tables_from_subqueries)))


class Query:
    def __init__(
        self,
        schemas,
        id,
        q,
        utterance,
        ltimestamp,
        count,
        is_subselect,
        sql_node=None,
        direct_parent_query=None,
        default_schema=None,
        dialect=None,
        query_tag=None,
    ):
        self.query_tag = query_tag
        self.dialect = dialect
        self.schemas = schemas
        self.id = id
        self.is_subselect = is_subselect
        self.string_query = q.replace('"', '\\"')
        self.utterance = (
            utterance.replace('"', '\\"') if utterance is not None else None
        )
        self.latest_timestamp = ltimestamp
        self.subselects: dict[str, Query] = {}
        self.section_to_subselect: dict[str, list[Query]] = {}
        self.alias_to_table = {}
        self.expressions_aliases = {}
        self.tables = []
        self.projection_nodes: list[Node] = []
        self.wildcarded_tables: list[Node | Query] = []
        self.select_edges = []
        self.from_edges = []
        self.where_edges = []
        self.order_by_edges = []
        self.limit_edges = []
        self.top_edges = []
        self.distinct_edges = []
        self.group_by_edges = []
        self.over_edges = []
        self.to_subselects = []
        self.with_edges = []
        self.name = None
        self.count = count
        self.filtered_edges = []
        self.root_query = (
            direct_parent_query.root_query
            if is_subselect and direct_parent_query
            else self
        )
        self.nodes_counter = 0 if self.root_query == self else None
        self.tables_ids: list[str] = list()
        self.reached_columns_ids: list[str] = list()  ## only for slim parsing

        if sql_node is None:
            month = self.latest_timestamp.month
            year = self.latest_timestamp.year
            props = {
                "name": f"query_{str(id)}",
                "is_sub_select": self.is_subselect,
                f"cnt_{month}_{year}": self.count,
                "total_counter": self.count,
                "sql_full_query": self.string_query,
                "last_query_timestamp": self.latest_timestamp,
                "query_tag": self.query_tag,
            }
            if self.utterance is not None:
                props["description"] = self.utterance
            sql_node = Node(
                name="query_" + str(id), label=Labels.SQL, props=props, existing_id=id
            )
        self.sql_node = sql_node
        if direct_parent_query:
            self.sql_node.props["sql_type"] = direct_parent_query.sql_node.props[
                "sql_type"
            ]
        self.direct_parent_query = direct_parent_query
        self.default_schema = default_schema

        # for insert into, create, update, select into sqls
        self.source_to_target_edges = []

    def get_wildcarded_tables(self):
        return self.wildcarded_tables

    def add_wilcarded_table(self, table_or_query):
        self.wildcarded_tables.append(table_or_query)

    def get_tables_ids(self):
        return list(set(self.tables_ids))

    def get_reached_columns_ids(self):
        return list(set(self.reached_columns_ids))

    def set_reached_columns_ids(self, columns_ids):
        self.reached_columns_ids = columns_ids

    def get_root_query(self):
        return self.root_query

    def add_table_id_to_root_query(self, table_id):
        self.root_query.tables_ids.append(table_id)

    def set_sql_type(self, sql_type):
        self.sql_node.props["sql_type"] = sql_type.lower()
        if sql_type.lower() != SQLType.QUERY:
            self.set_counter_to_zero()

    def set_counter_to_zero(self):
        self.count = 0
        self.sql_node.props["total_counter"] = 0
        month = self.latest_timestamp.month
        year = self.latest_timestamp.year
        self.sql_node.props[f"cnt_{month}_{year}"] = 0

    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def get_sql_node(self):
        return self.sql_node

    def add_projection_node(self, proj_node):
        self.projection_nodes.append(proj_node)

    def get_projection_nodes(self):
        return self.projection_nodes

    def get_id(self):
        return self.id

    def get_string_query(self):
        return self.string_query

    def get_utterance(self):
        return self.utterance

    def get_latest_timestamp(self):
        return self.latest_timestamp

    def get_tables(self):
        """
        :return: self.tables - a list of tuples of the form (table_name, table_node) holding all tables
        that the query addresses. These may be both from the schemas and from the WITH section.
        """
        return self.tables

    def get_table_node_from_schema(self, table_name, schema_name):
        """
        :return: table_node
        """
        table_node = None
        if schema_name:
            schema_name_lower = schema_name.lower()
            if schema_name_lower not in self.schemas:
                raise MissingDataError(
                    f"Schema {schema_name} has not been found in the graph."
                )
            table_node = self.schemas[schema_name_lower].get_table_node(table_name)
        else:
            # first check if the table name is an alias of a with subselect
            if table_name.lower() in self.alias_to_table:
                table_node = self.alias_to_table[table_name.lower()]
            if (
                not table_node
                and self.default_schema
                and self.schemas[self.default_schema.lower()].table_exists(table_name)
            ):
                table_node = self.schemas[self.default_schema.lower()].get_table_node(
                    table_name
                )
            if not table_node:
                for table_tuple in self.tables:
                    if table_tuple[0].lower() == table_name.lower():
                        table_node = table_tuple[1]
            if not table_node:
                for _, schema in self.schemas.items():
                    if schema.table_exists(table_name):
                        if table_node:
                            # when no schema name is given, it means that the table should be part of one schema only.
                            raise Exception(
                                f"Table {table_name} found in more than one schema in "
                                f'"get_table_node_from_schema". '
                            )
                        table_node = schema.get_table_node(table_name)
            if not table_node:
                # the table name is probably an alias of a derived subselect
                if self.direct_parent_query is not None:
                    table_node = self.direct_parent_query.get_table_node_from_schema(
                        table_name, schema_name
                    )

        return table_node

    def find_the_table_node_of_the_column(self, column_name):
        for table_name, table_node in self.tables:
            if self.is_column_in_table(table_node, column_name):
                return table_node
        return None

    def find_column_node(self, column_name):
        """
        :param: column_name: the name of the required column_node
        :return: column_node
        """
        column_node = []
        tables = (
            self.tables
            if len(self.tables) > 0
            else enumerate(self.section_to_subselect[SQL.FROM])
        )
        for table_name, table_node in tables:
            if isinstance(table_node, Query):
                proj_nodes = table_node.get_projection_nodes()
                for proj_n in proj_nodes:
                    if column_name.lower() == proj_n.get_name().lower():
                        c = proj_n
                        column_node.append(c)
            # table can also be a function
            elif table_node.label in [Labels.TABLE, Labels.TEMP_TABLE]:
                schema_name = table_node.get_match_props()["schema_name"]
                schema_name_lower = schema_name.lower()
                if self.schemas[schema_name_lower].is_column_in_table(
                    table_node, column_name
                ):
                    c = self.schemas[schema_name_lower].get_column_node(
                        column_name, table_name
                    )
                    if self.sql_node.props["sql_type"].lower() == SQLType.QUERY:
                        c.get_properties().update(
                            {"last_query_timestamp": self.latest_timestamp}
                        )
                    column_node.append(c)
        if len(column_node) == 0:
            column_node = self.find_column_in_with_clause(column_name)
        if len(column_node) == 0 and column_name in self.expressions_aliases:
            column_node = [self.expressions_aliases[column_name]]
        if len(column_node) == 0:
            for table_name, table_node in tables:
                if isinstance(table_node, Query):
                    if column_name in table_node.expressions_aliases:
                        column_node.append(table_node.expressions_aliases[column_name])
        if len(column_node) == 0:
            column_node = [
                c
                for c in self.projection_nodes
                if c.name.lower() == column_name.lower()
            ]
        if len(column_node) == 0:
            raise MissingDataError(
                f'Unknown column name {column_name} in "find_column_node". '
            )

        return column_node

    def find_column_in_with_clause(self, column_name):
        for subq in self.get_subselects_by_section(SQL.WITH):
            proj_nodes = subq.get_projection_nodes()
            for proj_n in proj_nodes:
                if column_name.lower() == proj_n.get_name().lower():
                    return [proj_n]
            if column_name in subq.expressions_aliases:
                return [subq.expressions_aliases[column_name]]
        return []

    def is_column_in_table(self, table_node, column_name):
        if isinstance(table_node, Query):
            proj_nodes = table_node.get_projection_nodes()
            for proj_n in proj_nodes:
                if column_name.lower() == proj_n.get_name().lower():
                    return True
            if column_name in self.expressions_aliases.keys():
                return True
        else:
            schema_name = table_node.get_match_props()["schema_name"]
            schema_name_lower = schema_name.lower()
            return self.schemas[schema_name_lower].is_column_in_table(
                table_node, column_name
            )

    def get_column_node(self, column_name, table_node, section_type=None):
        column_node = None
        if isinstance(table_node, Query):
            proj_nodes = table_node.get_projection_nodes()
            for proj_n in proj_nodes:
                if column_name.lower() == proj_n.get_name().lower():
                    column_node = proj_n
                    break
                elif proj_n.label == Labels.FUNCTION and proj_n.props[
                    "expr_str"
                ].lower().replace(" ", "") == column_name.lower().replace(" ", ""):
                    column_node = proj_n
                    break
            if (
                column_node is None
                and column_name in table_node.expressions_aliases.keys()
            ):
                column_node = table_node.expressions_aliases[column_name]
        elif isinstance(table_node, Node) and table_node.props.get("is_nested"):
            tables = table_node.props.get("tables")
            cols = []
            for t in tables:
                try:
                    c = self.get_column_node(column_name, t)
                except Exception:
                    # skip this table, continue to the next one
                    continue
                if c:
                    cols.append(c)
            if len(cols) == 1:
                column_node = cols[0]
            elif len(cols) > 1:
                column_node = Node(
                    name=column_name,
                    label=Labels.SET_OP_COLUMN,
                    props={"name": column_name, Props.SQL_ID: str(self.id)},
                )
                column_node.add_property("data_type", cols[0].props["data_type"])
                edge_params = {Props.SQL_ID: str(self.id)}
                for c in cols:
                    self.add_edge((column_node, c, edge_params), section_type)
        else:
            schema_name = table_node.get_match_props()["schema_name"]
            schema_name_lower = schema_name.lower()
            table_name = table_node.get_name()
            if self.schemas[schema_name_lower].is_column_in_table(
                table_node, column_name
            ):
                column_node = self.schemas[schema_name_lower].get_column_node(
                    column_name, table_name
                )

        if column_node is None and column_name in self.expressions_aliases.keys():
            column_node = self.expressions_aliases[column_name]

        if column_node is None:
            raise MissingDataError(
                f'"get_column_node": Unknown column name {column_name} in table {table_node.name}. '
            )
        return column_node

    def add_table_to_query(self, table_node: Union[Node, Type[Self]], table_name: str):
        if not isinstance(table_node, Query):
            self.add_table_id_to_root_query(table_node.id)
        if (table_name, table_node) not in self.tables:
            self.tables.append((table_name, table_node))
        self.increment_nodes_counter()

    def get_source_to_target_edges(self):
        return self.source_to_target_edges

    def set_filtered_edges(self, edges):
        self.filtered_edges = edges

    def get_edges(self):
        if self.filtered_edges:
            return self.filtered_edges + self.source_to_target_edges
        all_edges = []
        all_edges.extend(self.with_edges)
        all_edges.extend(self.select_edges)
        all_edges.extend(self.from_edges)
        all_edges.extend(self.where_edges)
        all_edges.extend(self.order_by_edges)
        all_edges.extend(self.limit_edges)
        all_edges.extend(self.top_edges)
        all_edges.extend(self.distinct_edges)
        all_edges.extend(self.group_by_edges)
        all_edges.extend(self.over_edges)
        all_edges.extend(self.to_subselects)

        all_edges.extend(self.source_to_target_edges)
        return all_edges

    def get_edges_by_section(self, section_type: SQL):
        match section_type:
            case SQL.WITH:
                return self.with_edges
            case SQL.SELECT:
                return self.select_edges
            case SQL.FROM:
                return self.from_edges
            case SQL.WHERE:
                return self.where_edges
            case SQL.ORDER_BY:
                return self.order_by_edges
            case SQL.LIMIT:
                return self.limit_edges
            case SQL.TOP:
                return self.top_edges
            case SQL.DISTINCT:
                return self.distinct_edges
            case SQL.GROUP_BY:
                return self.group_by_edges
            case SQL.OVER:
                return self.over_edges
            case Parser.SUBSELECT:
                return self.to_subselects
        return []

    def get_edges_dict(self) -> dict[SQL, list[tuple[Node, Node, dict, str]]]:
        edges_dict = {}
        edges_dict.update({SQL.WITH: self.with_edges})
        edges_dict.update({SQL.SELECT: self.select_edges})
        edges_dict.update({SQL.FROM: self.from_edges})
        edges_dict.update({SQL.WHERE: self.where_edges})
        edges_dict.update({SQL.ORDER_BY: self.order_by_edges})
        edges_dict.update({SQL.LIMIT: self.limit_edges})
        edges_dict.update({SQL.TOP: self.top_edges})
        edges_dict.update({SQL.DISTINCT: self.distinct_edges})
        edges_dict.update({SQL.GROUP_BY: self.group_by_edges})
        edges_dict.update({SQL.OVER: self.over_edges})
        edges_dict.update({Parser.SUBSELECT: self.to_subselects})
        return edges_dict

    def get_schemas(self):
        return self.schemas

    def get_tables_aliases(self):
        return self.alias_to_table

    def get_expression_aliases(self):
        return self.expressions_aliases

    def add_alias_to_table(self, table_alias, table_node):
        self.alias_to_table.update({table_alias.lower(): table_node})

    def add_expression_alias(self, alias_name, alias_node):
        self.expressions_aliases.update({alias_name: alias_node})

    def add_subselect(self, subselect_obj, id, section_type):
        self.subselects.update({id: subselect_obj})
        if section_type not in self.section_to_subselect.keys():
            self.section_to_subselect.update({section_type: []})
        self.section_to_subselect[section_type].append(subselect_obj)

    def get_tables_of_subselect(self, subselect_id):
        if subselect_id not in self.subselects:
            return []
        return get_tables_recursive(self.subselects[subselect_id].tables)

    def get_subselects_by_section(self, section_type):
        if section_type not in self.section_to_subselect.keys():
            return []
        return self.section_to_subselect[section_type]

    def update_edges_by_type(self, edges, section_type):
        if section_type == SQL.SELECT:
            self.select_edges = edges
        elif section_type == SQL.FROM:
            self.from_edges = edges
        elif section_type == SQL.WHERE:
            self.where_edges = edges
        elif section_type == SQL.ORDER_BY:
            self.order_by_edges = edges
        elif section_type == SQL.LIMIT:
            self.limit_edges = edges
        elif section_type == SQL.TOP:
            self.top_edges = edges
        elif section_type == SQL.DISTINCT:
            self.distinct_edges = edges
        elif section_type == Parser.SUBSELECT:
            self.to_subselects = edges
        elif section_type == SQL.GROUP_BY:
            self.group_by_edges = edges
        elif section_type == SQL.OVER:
            self.over_edges = edges
        elif section_type == SQL.WITH:
            self.with_edges = edges
        else:
            raise Exception(
                "Unknown section type "
                + section_type
                + " when trying to update with the edges: "
                + edges
            )

    def add_edge(self, edge, section_type):
        self.add_edges([edge], section_type)

    def add_edges(self, edges, section_type):
        if section_type == SQL.SELECT:
            self.select_edges.extend(edges)
        elif section_type == SQL.FROM:
            self.from_edges.extend(edges)
        elif section_type == SQL.WHERE:
            self.where_edges.extend(edges)
        elif section_type == SQL.ORDER_BY:
            self.order_by_edges.extend(edges)
        elif section_type == SQL.LIMIT:
            self.limit_edges.extend(edges)
        elif section_type == SQL.TOP:
            self.top_edges.extend(edges)
        elif section_type == SQL.DISTINCT:
            self.distinct_edges.extend(edges)
        elif section_type == Parser.SUBSELECT:
            self.to_subselects.extend(edges)
        elif section_type == SQL.GROUP_BY:
            self.group_by_edges.extend(edges)
        elif section_type == SQL.OVER:
            self.over_edges.extend(edges)
        elif section_type == SQL.WITH:
            self.with_edges.extend(edges)
        else:
            raise Exception(
                "Unknown section type "
                + section_type
                + " when adding the edges: "
                + edges
            )

    def get_sum_of_counters(self):
        if self.is_subselect:
            raise Exception("No counters in a subselect query object.")
        return self.sql_node.get_properties()["total_counter"]

    def get_nodes_counter(self):
        return self.root_query.nodes_counter

    def increment_nodes_counter(self):
        self.root_query.nodes_counter += 1

    def increase_sql_counter(self, query_obj):
        ltimestamp = query_obj.get_latest_timestamp()
        parsed_date_obj = pendulum.parse(str(ltimestamp))
        month = parsed_date_obj.month
        year = parsed_date_obj.year
        counter_prop_key = f"cnt_{month}_{year}"
        counter = query_obj.get_sql_node().get_properties()[counter_prop_key]
        if counter_prop_key in self.sql_node.get_properties().keys():
            self.sql_node.get_properties()[counter_prop_key] += counter
        else:
            self.sql_node.get_properties().update({counter_prop_key: counter})
        self.sql_node.get_properties()["total_counter"] += counter

    def add_sql_strs_to_node(self, select_str, from_str, where_str):
        if select_str.lower().startswith("select"):
            select_str = select_str[len("select") :]
        self.sql_node.add_property("select_str", select_str.strip())

        if from_str.lower().startswith("from"):
            from_str = from_str[len("from") :]
        self.sql_node.add_property("from_str", from_str.strip())

        if where_str.lower().startswith("where"):
            where_str = where_str[len("where") :]
        self.sql_node.add_property("where_str", where_str.strip())

    def get_left_side_of_assignment_operation(self, assignment_operation_node):
        operands = [
            edge[1]
            for edge in self.select_edges
            if (
                edge[0].id == assignment_operation_node.id
                and edge[1].label == Labels.COLUMN
            )
        ]
        if len(operands) == 1:
            return operands[0]
        return None

    def create_current_node(
        self,
        name: str,
        label: Labels,
        props: dict,
        section_type: str,
        edge_params: dict,
        parent: Node,
        is_full_parse: bool = False,
    ) -> Node:
        if get_should_update(label, is_full_parse, name):
            current_node = Node(
                name=name,
                label=label,
                props=props,
                existing_id=props.get("id"),
            )
            self.add_edge((parent, current_node, edge_params), section_type)
        else:
            current_node = parent
        self.increment_nodes_counter()
        return current_node


def is_join(label: Labels, name: str):
    return label == Labels.COMMAND and name == "Joins"


def get_should_update(
    label: Labels,
    is_full_parse: bool,
    name: str = "",
) -> bool:
    return is_full_parse or (label in PRESERVED_LABELS or is_join(label, name))


def handle_node_update(
    label: Labels,
    node: Node,
    is_full_parse: bool,
    payload: dict,
):
    should_update = get_should_update(label, is_full_parse, node.name)
    if not should_update or label in [
        Labels.COLUMN,
        Labels.TEMP_COLUMN,
        Labels.TEMP_TABLE,
        Labels.TABLE,
    ]:
        return
    node.add_properties(payload)
