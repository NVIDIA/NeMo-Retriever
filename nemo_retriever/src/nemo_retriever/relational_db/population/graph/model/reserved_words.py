class Parser:
    ATTIMEZONE = "AtTimeZone"
    TUPLE = "Tuple"
    NESTED_JOIN = "NestedJoin"
    ARRAYAGG = "ArrayAgg"
    ARRAY = "Array"
    ARRAYINDEX = "ArrayIndex"
    INTERVAL = "Interval"
    LIKE = "Like"
    ILIKE = "ILike"
    LISTAGG = "ListAgg"
    EXTRACT = "Extract"
    TYPED_STRING = "TypedString"
    CAST = "Cast"
    CONVERT = "Convert"
    TRIM = "Trim"
    FLOOR = "Floor"
    CEIL = "Ceil"
    OFFSET = "offset"
    FETCH = "fetch"
    BODY = "body"
    PROJECTION = "projection"
    LATERAL_VIEWS = "lateral_views"
    SELECTION = "selection"
    CLUSTER_BY = "cluster_by"
    DISTRIBUTE_BY = "distribute_by"
    SORT_BY = "sort_by"
    BINARY_OP = "BinaryOp"
    UNARY_OP = "UnaryOp"
    ANY_OP = "AnyOp"
    SET_OPERATION = "SetOperation"
    IDENTIFIER = "Identifier"
    COMPOUND_IDENTIFIER = "CompoundIdentifier"
    VALUE = "Value"
    WILDCARD = "Wildcard"
    Q_WILDCARD = "QualifiedWildcard"
    FUNCTION = "Function"
    SUBQUERY = "subquery"
    QUERY = "Query"
    INLIST = "InList"
    ISNULL = "IsNull"
    ISNOTNULL = "IsNotNull"
    ISTRUE = "IsTrue"
    ISFALSE = "IsFalse"
    ISNOTFALSE = "IsNotFalse"
    ISNOTTRUE = "IsNotTrue"
    SUBSELECT = "subselect"
    IN_SUBQUERY = "InSubquery"
    NEGATED = "negated"
    EXISTS = "Exists"
    JSON_ACCESS = "JsonAccess"
    TUPLE = "Tuple"


class SQLType:
    QUERY = "query"
    INSERT = "insert"
    CREATE_TABLE = "createtable"
    UPDATE = "update"
    MERGE = "merge"
    DELETE = "delete"
    VIEW = "view"
    BI = "bi"
    SEMANTIC = "semantic"


class SQL:
    SELECT = "select"
    FROM = "from"
    WITH = "with"
    WHERE = "where"
    ORDER_BY = "order_by"
    LIMIT = "limit"
    TOP = "top"
    DISTINCT = "distinct"
    GROUP_BY = "group_by"
    HAVING = "having"
    BETWEEN = "Between"
    CASE = "Case"
    UNION = "Union"
    EXCEPT = "Except"
    INTERSECT = "Intersect"
    OVER = "over"
    SET_OP_OPERATIONS = [UNION, EXCEPT, INTERSECT]


class Labels:
    COMMAND = "command"
    OPERATOR = "operator"
    FUNCTION = "function"
    CONSTANT = "constant"
    ALIAS = "alias"
    SQL = "sql"
    SET_OP_COLUMN = "set_op_column"
    COLUMN = "column"
    TEMP_COLUMN = "temp_column"
    TABLE = "table"
    TEMP_TABLE = "temp_table"
    SCHEMA = "schema"
    TEMP_SCHEMA = "temp_schema"
    DB = "db"
    BT = "term"
    ATTR = "attribute"
    ZONE = "zone"
    METRIC = "metric"
    METRIC_ITEM = "item"
    METRIC_AGG = "agg"
    METRIC_COND = "cond"
    ANALYSIS = "analysis"
    TAG = "tag"
    QUERY = "query"
    RULE = "rule"
    DOCUMENT = "document"
    CTE = "cte"

    PUBLISHED_DS = "published_datasource"
    EMBEDDED_DS = "embedded_datasource"
    CUSTOM_SQL_TABLE = "CustomSQLTable"
    SHEET = "sheet"
    FIELD = "field"
    WORKBOOK = "workbook"
    DASHBOARD = "dashboard"
    PROJECT = "project"
    SITE = "site"

    TABLEAU_LABELS = [
        PUBLISHED_DS,
        EMBEDDED_DS,
        CUSTOM_SQL_TABLE,
        SHEET,
        FIELD,
        WORKBOOK,
        DASHBOARD,
        PROJECT,
        SITE,
    ]

    QS = "quicksight"
    QS_ANALYSIS = "qs_analysis"
    QS_DS = "qs_dataset"
    QS_DASHBOARD = "qs_dashboard"
    QS_VISUALIZATION = "qs_visualization"
    QS_CUSTOM_SQL_TABLE = "qs_custom_sql_table"
    QS_FOLDER = "qs_folder"

    QUICKSIGHT_LABELS = [
        QS,
        QS_ANALYSIS,
        QS_DS,
        QS_DASHBOARD,
        QS_VISUALIZATION,
        QS_CUSTOM_SQL_TABLE,
        QS_FOLDER,
    ]

    SISENSE = "sisense"
    SISENSE_DATAMODEL = "sisense_datamodel"
    SISENSE_DASHBOARD = "sisense_dashboard"
    SISENSE_WIDGET = "sisense_widget"
    SISENSE_FOLDER = "sisense_folder"
    SISENSE_TABLE = "sisense_table"

    SISENSE_LABELS = [
        SISENSE,
        SISENSE_DATAMODEL,
        SISENSE_DASHBOARD,
        SISENSE_WIDGET,
        SISENSE_FOLDER,
        SISENSE_TABLE,
    ]

    POWERBI_DASHBOARD = "powerbi_dashboard"
    POWERBI_DATAFLOW = "powerbi_dataflow"
    POWERBI_REPORT = "powerbi_report"
    POWERBI_SEMANTIC_MODEL = "powerbi_semantic_model"
    POWERBI_SEMANTIC_TABLE = "powerbi_semantic_table"
    POWERBI_DATAFLOW_TABLE = "powerbi_dataflow_table"
    POWERBI_COPIED_DATAFLOW_TABLE = "powerbi_copied_dataflow_table"
    POWERBI_VISUALIZATION = "powerbi_visualization"
    POWERBI_WORKSPACE = "powerbi_workspace"

    POWERBI_LABELS = [
        POWERBI_DASHBOARD,
        POWERBI_DATAFLOW,
        POWERBI_REPORT,
        POWERBI_SEMANTIC_MODEL,
        POWERBI_SEMANTIC_TABLE,
        POWERBI_VISUALIZATION,
        POWERBI_WORKSPACE,
        POWERBI_DATAFLOW_TABLE,
        POWERBI_COPIED_DATAFLOW_TABLE,
    ]

    POWERBI_TABLES_LABELS = [
        POWERBI_DATAFLOW_TABLE,
        POWERBI_SEMANTIC_TABLE,
        POWERBI_COPIED_DATAFLOW_TABLE,
    ]

    LOOKER = "looker"
    LOOKER_FOLDER = "looker_folder"
    LOOKER_PROJECT = "looker_project"
    LOOKER_VIEW = "looker_view"
    LOOKER_EXPLORE = "looker_explore"
    LOOKER_LOOK = "looker_look"
    LOOKER_DASHBOARD = "looker_dashboard"
    LOOKER_VISUAL = "looker_visual"
    LOOKER_BOARD = "looker_board"

    LOOKER_LABELS = [
        LOOKER,
        LOOKER_FOLDER,
        LOOKER_PROJECT,
        LOOKER_VIEW,
        LOOKER_EXPLORE,
        LOOKER_LOOK,
        LOOKER_DASHBOARD,
        LOOKER_VISUAL,
        LOOKER_BOARD,
    ]

    LOOKER_DISCOVERY_LABELS = [
        LOOKER_VIEW,
        LOOKER_EXPLORE,
        LOOKER_LOOK,
        LOOKER_DASHBOARD,
        LOOKER_VISUAL,
    ]

    LIST_OF_ALL = [
        DB,
        SCHEMA,
        TABLE,
        COLUMN,
        TEMP_SCHEMA,
        TEMP_TABLE,
        TEMP_COLUMN,
        SET_OP_COLUMN,
        SQL,
        COMMAND,
        OPERATOR,
        FUNCTION,
        CONSTANT,
        ALIAS,
        BT,
        ATTR,
        ZONE,
        METRIC,
        METRIC_ITEM,
        METRIC_AGG,
        METRIC_COND,
        ANALYSIS,
        TAG,
        RULE,
        CTE,
        PUBLISHED_DS,
        EMBEDDED_DS,
        CUSTOM_SQL_TABLE,
        SHEET,
        FIELD,
        WORKBOOK,
        DASHBOARD,
        PROJECT,
        SITE,
        QS,
        QS_DS,
        QS_DASHBOARD,
        QS_ANALYSIS,
        QS_VISUALIZATION,
        QS_CUSTOM_SQL_TABLE,
        QS_FOLDER,
        SISENSE,
        SISENSE_DATAMODEL,
        SISENSE_DASHBOARD,
        SISENSE_WIDGET,
        SISENSE_FOLDER,
        SISENSE_TABLE,
        POWERBI_WORKSPACE,
        POWERBI_DATAFLOW,
        POWERBI_SEMANTIC_MODEL,
        POWERBI_REPORT,
        POWERBI_DASHBOARD,
        POWERBI_VISUALIZATION,
        POWERBI_SEMANTIC_TABLE,
        POWERBI_COPIED_DATAFLOW_TABLE,
        POWERBI_DATAFLOW_TABLE,
        LOOKER,
        LOOKER_FOLDER,
        LOOKER_PROJECT,
        LOOKER_VIEW,
        LOOKER_EXPLORE,
        LOOKER_LOOK,
        LOOKER_DASHBOARD,
        LOOKER_BOARD,
        LOOKER_VISUAL,
        DOCUMENT,
    ]

    LIST_OF_SEMANTIC = [
        BT,
        ATTR,
        METRIC,
        ANALYSIS,
    ]

    LIST_OF_VISUALS = [
        SHEET,
        QS_VISUALIZATION,
        SISENSE_WIDGET,
        POWERBI_VISUALIZATION,
        LOOKER_VISUAL,
        LOOKER_LOOK,
    ]
    LIST_OF_DASHBOARDS = [
        DASHBOARD,
        QS_DASHBOARD,
        SISENSE_DASHBOARD,
        POWERBI_REPORT,
        POWERBI_DASHBOARD,
        LOOKER_DASHBOARD,
    ]

    LIST_OF_BI = (
        TABLEAU_LABELS
        + QUICKSIGHT_LABELS
        + SISENSE_LABELS
        + POWERBI_LABELS
        + LOOKER_LABELS
    )

    LIST_FOR_VECTOR_INDEX = LIST_OF_SEMANTIC + [
        DB,
        SCHEMA,
        TABLE,
        COLUMN,
        ZONE,
    ]


## when adding new datasources please make sure that you add the relationships
## and the roots so that the data page keeps working :)
class DatasourcesRelationships:
    DASHBOARD_FIELD = "dashboard_field"
    EMBEDDED_FIELD = "embedded_field"
    PUBLISHED_FIELD = "published_field"
    COPY_PUBLISHED_FIELD = "copied_published_field"
    FIELD = "field_of"
    SHEET_FIELD = "sheet_field"
    WIDGET_FIELD = "widget_field"
    VISUALIZATION_FIELD = "visualization_field"
    SCHEMA = "schema"
    BI = "bi_rel"


visuals_fields_relationships = [
    DatasourcesRelationships.SHEET_FIELD,
    DatasourcesRelationships.VISUALIZATION_FIELD,
    DatasourcesRelationships.WIDGET_FIELD,
]

fields_relationships = visuals_fields_relationships + [
    DatasourcesRelationships.PUBLISHED_FIELD,
    DatasourcesRelationships.COPY_PUBLISHED_FIELD,
    DatasourcesRelationships.DASHBOARD_FIELD,
    DatasourcesRelationships.EMBEDDED_FIELD,
    DatasourcesRelationships.FIELD,
]

data_relationships = fields_relationships + [
    DatasourcesRelationships.SCHEMA,
    DatasourcesRelationships.BI,
]

data_tree_labels = [
    Labels.DB,
    Labels.SCHEMA,
    Labels.TABLE,
] + Labels.LIST_OF_BI

semantic_labels = [
    Labels.BT,
    Labels.ATTR,
    Labels.METRIC,
    Labels.ANALYSIS,
]

labels_to_types = {
    Labels.QS_DS: "dataset",
    Labels.POWERBI_SEMANTIC_MODEL: "semantic_model",
    Labels.POWERBI_SEMANTIC_TABLE: "powerbi_table",
    Labels.POWERBI_DATAFLOW_TABLE: "powerbi_table",
    Labels.POWERBI_COPIED_DATAFLOW_TABLE: "powerbi_table",
    ## visualizations labels
    Labels.SHEET: Labels.SHEET,  ##tableau
    Labels.QS_VISUALIZATION: Labels.SHEET,
    Labels.SISENSE_WIDGET: Labels.SHEET,
    Labels.POWERBI_VISUALIZATION: Labels.SHEET,
    Labels.LOOKER_VISUAL: Labels.SHEET,
    Labels.LOOKER_LOOK: Labels.SHEET,
    ## dashboards labels
    Labels.DASHBOARD: Labels.DASHBOARD,  ##tableau
    Labels.POWERBI_DASHBOARD: Labels.DASHBOARD,
    Labels.QS_DASHBOARD: Labels.DASHBOARD,
    Labels.SISENSE_DASHBOARD: Labels.DASHBOARD,
    Labels.LOOKER_DASHBOARD: Labels.DASHBOARD,
    Labels.POWERBI_DATAFLOW: "dataflow",
    Labels.POWERBI_REPORT: "report",
    Labels.QS_FOLDER: "folder",
    Labels.LOOKER_FOLDER: "looker_folder",
    Labels.LOOKER_BOARD: "looker_board",
    Labels.QS: Labels.PROJECT,
    Labels.SISENSE: Labels.PROJECT,
    Labels.LOOKER: Labels.PROJECT,
    Labels.POWERBI_WORKSPACE: "workspace",
    Labels.TABLE: "base table",
    ## looker labels
    Labels.LOOKER_VIEW: Labels.PUBLISHED_DS,
    Labels.LOOKER_EXPLORE: Labels.PUBLISHED_DS,
}

entities_without_owners = [
    Labels.ATTR,
    Labels.COLUMN,
    Labels.FIELD,
    Labels.POWERBI_VISUALIZATION,
]


class BiConnectors:
    QUICKSIGHT = "quicksight"
    SISENSE = "sisense"
    TABLEAU = "tableau"
    POWERBI = "powerbi"
    LOOKER = "looker"
    ALL = [QUICKSIGHT, SISENSE, TABLEAU, POWERBI, LOOKER]


types_to_labels_by_connection = {
    BiConnectors.LOOKER: {
        Labels.PROJECT: Labels.LOOKER,
        Labels.DASHBOARD: Labels.LOOKER_DASHBOARD,
        Labels.SHEET: [
            Labels.LOOKER_VISUAL,
            Labels.LOOKER_LOOK,
        ],
        Labels.PUBLISHED_DS: [
            Labels.LOOKER_VIEW,
            Labels.LOOKER_EXPLORE,
        ],
        "looker_folder": Labels.LOOKER_FOLDER,
        Labels.LOOKER_BOARD: [Labels.LOOKER_DASHBOARD, Labels.LOOKER_BOARD],
    },
    BiConnectors.QUICKSIGHT: {
        Labels.PROJECT: "quicksight",
        "dataset": Labels.QS_DS,
        Labels.DASHBOARD: Labels.QS_DASHBOARD,
        Labels.SHEET: Labels.QS_VISUALIZATION,
        "folder": Labels.QS_FOLDER,
    },
    BiConnectors.POWERBI: {
        Labels.PROJECT: Labels.POWERBI_WORKSPACE,
        "semantic_model": Labels.POWERBI_SEMANTIC_MODEL,
        Labels.SHEET: Labels.POWERBI_VISUALIZATION,
        "dataflow": Labels.POWERBI_DATAFLOW,
        "report": Labels.POWERBI_REPORT,
        "powerbi_table": [
            Labels.POWERBI_COPIED_DATAFLOW_TABLE,
            Labels.POWERBI_DATAFLOW_TABLE,
            Labels.POWERBI_SEMANTIC_TABLE,
        ],
        Labels.DASHBOARD: Labels.POWERBI_DASHBOARD,
    },
    BiConnectors.SISENSE: {
        Labels.PROJECT: Labels.SISENSE,
        Labels.DASHBOARD: Labels.SISENSE_DASHBOARD,
        Labels.SHEET: Labels.SISENSE_WIDGET,
    },
}

bi_roots = {
    BiConnectors.SISENSE: Labels.SISENSE,
    BiConnectors.TABLEAU: Labels.SITE,
    BiConnectors.QUICKSIGHT: Labels.QS,
    BiConnectors.POWERBI: Labels.POWERBI_WORKSPACE,
    BiConnectors.LOOKER: Labels.LOOKER,
}


def label_to_type(label: str) -> str:
    if label in labels_to_types:
        return labels_to_types[label]
    return label


def type_to_label(type: str, connector_type: str = None):
    if type in ["view", "materialized view", "external table", "base table"]:
        return Labels.TABLE
    if (
        connector_type is not None
        and connector_type in types_to_labels_by_connection
        and type in types_to_labels_by_connection[connector_type]
    ):
        label = types_to_labels_by_connection[connector_type][type]
        if isinstance(label, list):
            return "|".join(label)
        return label
    return type


def type_to_labels(type: str):
    labels = []
    for label, typeValue in labels_to_types.items():
        if typeValue == type:
            if isinstance(label, list):
                labels.extend(label)
            else:
                labels.append(label)
    return labels


def get_number_of_sheets_by_connector(connector: str):
    labels = types_to_labels_by_connection.get(connector, {})
    if isinstance(labels.get("sheet"), list):
        return len(labels.get("sheet", []))
    return 1


class JoinNodes:
    INNER = "Inner"
    LEFT_OUTER = "LeftOuter"
    RIGHT_OUTER = "RightOuter"
    NATURAL = "Natural"

    LIST_OF_ALL_JOINS = [INNER, LEFT_OUTER, RIGHT_OUTER, NATURAL]

    join_str_map = {
        INNER: INNER,
        LEFT_OUTER: "Left Outer",
        RIGHT_OUTER: "Right Outer",
        NATURAL: "natural join",
        "CrossJoin": "Cross Join",
        "FullOuter": "Full Outer",
    }


class Props:
    SQL_ID = "sql_id"
    SOURCE_SQL_ID = "source_sql_id"
    JOIN = "join"
    JOIN_SQL_ID = "join_sql_id"
    UNION = "union"


class Views:
    VIEW = "view"
    NON_BINDING_VIEW = "non_binding_view"
    MATERIALIZED_VIEW = "materialized view"


class MathExpr:
    exps = {
        "divide": "/",
        "minus": "-",
        "plus": "+",
        "eq": "=",
        "multiply": "*",
        "and": "and",
        "or": "or",
        "lt": "<",
        "gt": ">",
        "lteq": "=<",
        "gteq": "=>",
    }


class DataTypes:
    datetime_types = [
        "datetime",
        "date",
        "timestamp",
        "time",
        "year",
        "datetime2",
        "smalldatetime",
        "date",
        "time",
        "datetimeoffset",
        "timestamp",
    ]

    numeric_types = [
        "int",
        "bit",
        "tinyint",
        "bool",
        "boolean",
        "smallint",
        "mediumint",
        "integer",
        "bigint",
        "float",
        "double",
        "double precision",
        "decimal",
        "dec",
        "numeric",
        "smallmoney",
        "money",
        "float",
        "real",
        "number",
    ]

    string_types = [
        "char",
        "varchar",
        "text",
        "nchar",
        "nvarchar",
        "ntext",
        "binary",
        "varbinary",
        "varbinary",
        "image",
        "tinyblob",
        "tinytext",
        "blob",
        "mediumtext",
        "mediumblob",
        "longtext",
        "longblob",
        "enum",
        "set",
        "string",
        "singlequotedstring",
    ]


def get_type_family(type: str):
    if type.lower() in DataTypes.datetime_types:
        return "datetime"
    if type.lower() in DataTypes.numeric_types:
        return "numeric"
    if type.lower() in DataTypes.string_types:
        return "string"
    return ""


def get_types_families(types: list[str]):
    families = [get_type_family(type) for type in types]
    unique_families = list(set(families))
    return unique_families


class SQLFunctions:
    relation_functions = ["GETITEMFORAGGREGATORS", "RESULT_SCAN"]
    editable_time_functions = {"NOW": "NOW()"}
    editable_numeric_functions = {
        "PREVIOUS_MONTH": "MONTH(DATEADD(MONTH, -1, NOW()))",
        "THIS_MONTH": "MONTH(DATEADD(MONTH, 0, CURRENT_TIMESTAMP))",
    }
    dates_parameters = [
        "day",
        "month",
        "year",
        "microsecond",
        "millisecond",
        "second",
        "minute",
        "hour",
        "week",
        "quarter",
        "decade",
        "century",
        "millenium",
    ]
    agg_funcs = ["median", "avg", "max", "min", "sum", "count"]

    all_funcs = {
        "dateadd": [
            DataTypes.numeric_types + DataTypes.datetime_types + dates_parameters,
            DataTypes.datetime_types,
        ],
        "date_trunc": [
            DataTypes.string_types + dates_parameters + DataTypes.datetime_types,
            DataTypes.datetime_types,
        ],
        "from_unixtime": [DataTypes.numeric_types, DataTypes.datetime_types],
        "to_unixtime": [DataTypes.datetime_types, DataTypes.numeric_types],
        "from_iso8601_timestamp": [DataTypes.string_types, DataTypes.datetime_types],
        "to_iso8601": [DataTypes.datetime_types, DataTypes.string_types],
        "datediff": [DataTypes.datetime_types, DataTypes.numeric_types],
        "date_diff": [
            DataTypes.datetime_types + DataTypes.string_types,
            DataTypes.numeric_types,
        ],
        "datename": [DataTypes.datetime_types, DataTypes.numeric_types],
        "datepart": [DataTypes.datetime_types, DataTypes.numeric_types],
        "day": [DataTypes.datetime_types, DataTypes.numeric_types],
        "getdate": [[""], DataTypes.datetime_types],
        "getutcdate": [[""], DataTypes.datetime_types],
        "month": [DataTypes.datetime_types, DataTypes.numeric_types],
        "sysdatetime": [[""], DataTypes.datetime_types],
        "year": [DataTypes.datetime_types, DataTypes.numeric_types],
        "adddate": [DataTypes.datetime_types, DataTypes.datetime_types],
        "addtime": [DataTypes.datetime_types, DataTypes.datetime_types],
        "curdate": [[""], DataTypes.datetime_types],
        "current_date": [[""], DataTypes.datetime_types],
        "current_time": [[""], DataTypes.datetime_types],
        "current_timestamp": [[""], DataTypes.datetime_types],
        "curtime": [[""], DataTypes.datetime_types],
        "date": [DataTypes.datetime_types, DataTypes.datetime_types],
        # 'datediff': [DataTypes.datetime_types, DataTypes.numeric_types],
        "date_add": [DataTypes.datetime_types, DataTypes.datetime_types],
        "date_sub": [DataTypes.datetime_types, DataTypes.datetime_types],
        "dayname": [DataTypes.datetime_types, DataTypes.string_types],
        "dayofmonth": [DataTypes.datetime_types, DataTypes.numeric_types],
        "dayofweek": [DataTypes.datetime_types, DataTypes.numeric_types],
        "dayofyear": [DataTypes.datetime_types, DataTypes.numeric_types],
        "extract": [DataTypes.datetime_types, DataTypes.numeric_types],
        "from_days": [DataTypes.numeric_types, DataTypes.datetime_types],
        "hour": [DataTypes.datetime_types, DataTypes.numeric_types],
        "last_day": [DataTypes.datetime_types, DataTypes.numeric_types],
        "localtime": [[""], DataTypes.datetime_types],
        "localtimestamp": [[""], DataTypes.datetime_types],
        "minute": [DataTypes.datetime_types, DataTypes.numeric_types],
        "monthname": [DataTypes.datetime_types, DataTypes.string_types],
        "now": [[""], DataTypes.datetime_types],
        "period_add": [DataTypes.numeric_types, DataTypes.numeric_types],
        "period_diff": [DataTypes.numeric_types, DataTypes.numeric_types],
        "quarter": [DataTypes.datetime_types, DataTypes.numeric_types],
        "second": [DataTypes.datetime_types, DataTypes.numeric_types],
        "subdate": [DataTypes.datetime_types, DataTypes.datetime_types],
        "subtime": [DataTypes.datetime_types, DataTypes.datetime_types],
        "sysdate": [[""], DataTypes.datetime_types],
        "time": [DataTypes.string_types, DataTypes.datetime_types],
        "timediff": [DataTypes.datetime_types, DataTypes.datetime_types],
        "timestamp": [DataTypes.datetime_types, DataTypes.datetime_types],
        "to_days": [DataTypes.datetime_types, DataTypes.numeric_types],
        "week": [DataTypes.datetime_types, DataTypes.numeric_types],
        "weekday": [DataTypes.datetime_types, DataTypes.numeric_types],
        "weekofyear": [DataTypes.datetime_types, DataTypes.numeric_types],
        "yearweek": [DataTypes.datetime_types, DataTypes.numeric_types],
        # numeric_funcs
        "abs": [DataTypes.numeric_types, DataTypes.numeric_types],
        "acos": [DataTypes.numeric_types, DataTypes.numeric_types],
        "asin": [DataTypes.numeric_types, DataTypes.numeric_types],
        "atan": [DataTypes.numeric_types, DataTypes.numeric_types],
        "atn2": [DataTypes.numeric_types, DataTypes.numeric_types],
        "avg": [
            DataTypes.numeric_types + DataTypes.string_types,
            DataTypes.numeric_types,
        ],
        "average": [
            DataTypes.numeric_types + DataTypes.string_types,
            DataTypes.numeric_types,
        ],
        "ceiling": [DataTypes.numeric_types, DataTypes.numeric_types],
        "count": [
            DataTypes.numeric_types + DataTypes.string_types + DataTypes.datetime_types,
            DataTypes.numeric_types,
        ],
        "cos": [DataTypes.numeric_types, DataTypes.numeric_types],
        "cot": [DataTypes.numeric_types, DataTypes.numeric_types],
        "degrees": [DataTypes.numeric_types, DataTypes.numeric_types],
        "exp": [DataTypes.numeric_types, DataTypes.numeric_types],
        "floor": [DataTypes.numeric_types, DataTypes.numeric_types],
        "log": [DataTypes.numeric_types, DataTypes.numeric_types],
        "log10": [DataTypes.numeric_types, DataTypes.numeric_types],
        "max": [
            DataTypes.numeric_types + DataTypes.datetime_types + DataTypes.string_types,
            DataTypes.numeric_types + DataTypes.datetime_types + DataTypes.string_types,
        ],
        "min": [
            DataTypes.numeric_types + DataTypes.datetime_types + DataTypes.string_types,
            DataTypes.numeric_types + DataTypes.datetime_types + DataTypes.string_types,
        ],
        "pi": [[""], DataTypes.numeric_types],
        "power": [DataTypes.numeric_types, DataTypes.numeric_types],
        "radians": [DataTypes.numeric_types, DataTypes.numeric_types],
        "rand": None,
        "round": [DataTypes.numeric_types, DataTypes.numeric_types],
        "sign": [DataTypes.numeric_types, DataTypes.numeric_types],
        "sin": [DataTypes.numeric_types, DataTypes.numeric_types],
        "sqrt": [DataTypes.numeric_types, DataTypes.numeric_types],
        "square": [DataTypes.numeric_types, DataTypes.numeric_types],
        "sum": [DataTypes.numeric_types, DataTypes.numeric_types],
        "tan": [DataTypes.numeric_types, DataTypes.numeric_types],
        "ceil": [DataTypes.numeric_types, DataTypes.numeric_types],
        "div": [DataTypes.numeric_types, DataTypes.numeric_types],
        "greatest": [DataTypes.numeric_types, DataTypes.numeric_types],
        "least": [DataTypes.numeric_types, DataTypes.numeric_types],
        "ln": [DataTypes.numeric_types, DataTypes.numeric_types],
        "log2": [DataTypes.numeric_types, DataTypes.numeric_types],
        "mod": [DataTypes.numeric_types, DataTypes.numeric_types],
        "pow": [DataTypes.numeric_types, DataTypes.numeric_types],
        "truncate": [DataTypes.numeric_types, DataTypes.numeric_types],
        # string_funcs
        "datalength": [DataTypes.string_types, DataTypes.numeric_types],
        "difference": [DataTypes.string_types, DataTypes.numeric_types],
        "left": [DataTypes.string_types, DataTypes.string_types],  # Sql Server
        "len": [DataTypes.string_types, DataTypes.numeric_types],
        "lower": [DataTypes.string_types, DataTypes.string_types],
        "replace": [DataTypes.string_types, DataTypes.string_types],
        "reverse": [DataTypes.string_types, DataTypes.string_types],
        "right": [DataTypes.string_types, DataTypes.string_types],
        "soundex": [DataTypes.string_types, DataTypes.string_types],
        "space": [DataTypes.numeric_types, DataTypes.string_types],
        "str": [DataTypes.numeric_types, DataTypes.string_types],
        "substring": [
            DataTypes.string_types + DataTypes.numeric_types,
            DataTypes.string_types,
        ],
        "upper": [DataTypes.string_types, DataTypes.string_types],
        "char_length": [DataTypes.string_types, DataTypes.numeric_types],
        "character_length": [DataTypes.string_types, DataTypes.numeric_types],
        "field": [DataTypes.string_types, DataTypes.numeric_types],
        "instr": [DataTypes.string_types, DataTypes.numeric_types],
        "lcase": [DataTypes.string_types, DataTypes.string_types],
        "length": [DataTypes.string_types, DataTypes.numeric_types],
        "ltrim": [DataTypes.string_types, DataTypes.string_types],
        "mid": [DataTypes.string_types, DataTypes.string_types],
        "position": [DataTypes.string_types, DataTypes.numeric_types],
        "strcmp": [DataTypes.string_types, DataTypes.numeric_types],
        "substr": [
            DataTypes.string_types + DataTypes.numeric_types,
            DataTypes.string_types,
        ],  # numeric for indecies from where to do substring, maybe think of better solution
        "ucase": [DataTypes.string_types, DataTypes.string_types],
        "median": [
            DataTypes.numeric_types,
            DataTypes.numeric_types,
        ],  # NO SQL FUNCTION!!!!!!
        "dense_rank": [[""], DataTypes.numeric_types],
        "order by": [
            DataTypes.numeric_types + DataTypes.string_types + DataTypes.datetime_types,
            DataTypes.numeric_types + DataTypes.string_types + DataTypes.datetime_types,
        ],
        "format": [
            DataTypes.numeric_types + DataTypes.datetime_types + DataTypes.string_types,
            DataTypes.numeric_types + DataTypes.datetime_types,
        ],
        "row_number": [[""], DataTypes.numeric_types],
        "bitwise_and": [DataTypes.numeric_types, DataTypes.numeric_types],
        "concat": [DataTypes.string_types, DataTypes.string_types],
    }


class ArgsForSQLFunctions:
    args = [
        "d",
        "dd",
        "date",
        "time",
        "qtr",
        "day",
        "days",
        "dow",
        "datetime",
        "varchar",
        "quarter",
        "quarters",
        "year",
        "years",
        "month",
        "months",
        "dayofyear",
        "week",
        "weeks",
        "weekday",
        "hour",
        "minute",
        "minutes",
        "n",
        "second",
        "millisecond",
        "nanosecond",
        "microsecond",
        "tzoffset",
        "iso_week",
        "century",
        "decade",
        "milliseconds",
        "microseconds",
        "doy",
        "epoch",
        "isodow",
        "isoyear",
        "timezone",
        "timezone_hour",
        "timezone_minute",
        "bigint",
        "int",
        "smallint",
        "tinyint",
        "bit",
        "decimal",
        "numeric",
        "money",
        "smallmoney",
        "float",
        "real",
        "datetime",
        "smalldatetime",
        "char",
        "varchar",
        "text",
        "nchar",
        "nvarchar",
        "ntext",
        "binary",
        "varbinary",
        "h",
        "y",
        "w",
        "yy",
        "mm",
        "sysdate",
    ]


class SQLFunctionsWithConsantArg:
    functions = [
        "datediff",
        "date_part",
        "convert",
        "dateadd",
        "datepart",
        "timediff",
        "timestampdiff",
        "datename",
        "date_trunc",
        "trunc",
        "add_months",
        "to_char",
    ]


class OpsInduced:
    converter = {
        "gt": ["eq", "GtEq"],
        "eq": ["Gt", "GtEq", "Lt", "LtEq"],
        "gteq": ["Gt", "Eq"],
        "lt": ["Eq", "LtEq"],
        "lteq": ["Eq", "Lt"],
    }


unimportant_funcs_for_description = {
    "from_unixtime": [],
    "to_unixtime": [],
    "from_iso8601_timestamp": [],
    "to_iso8601": [],
    "substr": [1, 2],
    "round": [1],
    "format": [1],
}


class UsageDesc:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNUSED = "unused"
