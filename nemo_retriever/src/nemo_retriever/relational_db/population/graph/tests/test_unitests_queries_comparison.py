import os
import pytest
from shared.files.utils import load_tables, load_columns
from shared.graph.parsers import schemas_parser
from shared.graph.services.queries_comparison.compare_queries import (
    compare_query_to_list_of_queries,
    get_slim_graph_from_edges,
)
from shared.graph.parsers.sql.queries_parser import parse_single

ACCOUNT_ID = "test_account"


@pytest.fixture
def data_dir():
    # Get the directory of the current test file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "data")


@pytest.fixture
def tables_df(data_dir):
    tables_path = os.path.join(data_dir, "tables.csv")
    return load_tables(tables_path)


@pytest.fixture
def columns_df(data_dir):
    columns_path = os.path.join(data_dir, "columns.csv")
    return load_columns(columns_path)


@pytest.fixture
def schemas(tables_df, columns_df):
    schemas, _ = schemas_parser.parse_df(
        tables_df, columns_df, ACCOUNT_ID, temp_schema_creation_flag=True
    )
    return schemas


def test_find_identical_queries(schemas):
    main_sql = "SELECT ORDERLINEID, ORDERID FROM SALES.ORDERLINES"
    in_memory_queries = {
        "id1": {"string_query": "SELECT ORDERID, ORDERLINEID FROM SALES.ORDERLINES"},
    }

    query_obj = parse_single(
        q=main_sql,
        schemas=schemas,
        dialects=["snowflake"],
        is_full_parse=True,
    )
    main_graph = get_slim_graph_from_edges(query_obj.get_edges())

    (
        identical_sqls_ids_in_memory,
        subset_ids_from_memory,
    ) = compare_query_to_list_of_queries(
        main_graph=main_graph,
        queries_ids=in_memory_queries.keys(),
        get_parsed_query=lambda sql_query: parse_single(
            q=sql_query,
            schemas=schemas,
            dialects=["snowflake"],
            is_full_parse=True,
            keep_string_vals=False,
        ),
        get_sql_string_by_id=lambda id: in_memory_queries[id]["string_query"],
        is_subgraph=False,
        remove_aliases=False,
    )

    assert len(identical_sqls_ids_in_memory) == 1
    assert identical_sqls_ids_in_memory[0] == "id1"
    assert len(subset_ids_from_memory) == 0


def test_find_identical_queries_with_functions(schemas):
    main_sql = "SELECT COUNT(distinct ORDERLINEID) FROM SALES.ORDERLINES"
    in_memory_queries = {
        "id1": {"string_query": "SELECT COUNT(ORDERLINEID) FROM SALES.ORDERLINES"},
        "id2": {
            "string_query": "\nselect \n count(distinct orderlineid) \n from \n sales.orderlines"
        },
        "id3": {
            "string_query": "SELECT COUNT(ORDERLINEID) as COUNT_ORDERLINEID FROM SALES.ORDERLINES"
        },
        "id4": {"string_query": "SELECT SUM(ORDERLINEID) FROM SALES.ORDERLINES"},
        "id5": {
            "string_query": "SELECT COUNT(COUNT(ORDERLINEID)) FROM SALES.ORDERLINES"
        },
        "id6": {"string_query": "SELECT COUNT(*) FROM SALES.ORDERLINES"},
        "id7": {
            "string_query": "SELECT COUNT(distinct ORDERLINEID) FROM SALES.ORDERLINES"
        },
        "id8": {
            "string_query": "SELECT CAST(COUNT(distinct ORDERLINEID) AS INTEGER) FROM SALES.ORDERLINES"
        },
        "id9": {
            "string_query": "SELECT COUNT(distinct ORDERLINEID) + 2 FROM SALES.ORDERLINES"
        },
        "id10": {
            "string_query": "SELECT * FROM (SELECT COUNT(distinct ORDERLINEID) FROM SALES.ORDERLINES)"
        },
    }

    query_obj = parse_single(
        q=main_sql,
        schemas=schemas,
        dialects=["snowflake"],
        is_full_parse=True,
        keep_string_vals=True,
    )
    main_graph = get_slim_graph_from_edges(query_obj.get_edges())

    (
        identical_sqls_ids_in_memory,
        subset_ids_from_memory,
    ) = compare_query_to_list_of_queries(
        main_graph=main_graph,
        queries_ids=in_memory_queries.keys(),
        get_parsed_query=lambda sql_query: parse_single(
            q=sql_query,
            schemas=schemas,
            dialects=["snowflake"],
            is_full_parse=True,
            keep_string_vals=True,
        ),
        get_sql_string_by_id=lambda id: in_memory_queries[id]["string_query"],
        is_subgraph=True,
        remove_aliases=False,
    )

    assert len(identical_sqls_ids_in_memory) == 3
    assert identical_sqls_ids_in_memory[0] == "id2"
    assert identical_sqls_ids_in_memory[1] == "id7"
    assert identical_sqls_ids_in_memory[2] == "id8"
    assert len(subset_ids_from_memory) == 1
    assert subset_ids_from_memory[0] == "id10"


def test_find_subset_queries(schemas):
    main_sql = "SELECT ORDERLINEID FROM SALES.ORDERLINES"
    in_memory_queries = {
        "id1": {
            "string_query": "SELECT ORDERLINEID FROM SALES.ORDERLINES WHERE ORDERLINEID = 1"
        },
        "id2": {"string_query": "SELECT ORDERLINEID, ORDERID FROM SALES.ORDERLINES "},
        "id3": {
            "string_query": "SELECT * FROM (SELECT ORDERLINEID, ORDERID FROM SALES.ORDERLINES)"
        },
        "id4": {"string_query": "SELECT ORDERLINEID FROM SALES.ORDERLINES"},
        "id5": {"string_query": "SELECT SUM(ORDERLINEID) FROM SALES.ORDERLINES"},
    }

    query_obj = parse_single(
        q=main_sql,
        schemas=schemas,
        dialects=["snowflake"],
        is_full_parse=True,
    )
    main_graph = get_slim_graph_from_edges(query_obj.get_edges())

    (
        identical_sqls_ids_in_memory,
        subset_ids_from_memory,
    ) = compare_query_to_list_of_queries(
        main_graph=main_graph,
        queries_ids=in_memory_queries.keys(),
        get_parsed_query=lambda sql_query: parse_single(
            q=sql_query,
            schemas=schemas,
            dialects=["snowflake"],
            is_full_parse=True,
            keep_string_vals=False,
        ),
        get_sql_string_by_id=lambda id: in_memory_queries[id]["string_query"],
        is_subgraph=True,
        remove_aliases=False,
    )

    assert len(identical_sqls_ids_in_memory) == 1
    assert identical_sqls_ids_in_memory[0] == "id4"
    assert len(subset_ids_from_memory) == 3
    assert subset_ids_from_memory[0] == "id1"
    assert subset_ids_from_memory[1] == "id2"
    assert subset_ids_from_memory[2] == "id3"


def test_find_subset_queries_without_aliases(schemas):
    main_sql = "SELECT ORDERLINEID AS ORDERLINEID_ALIAS FROM SALES.ORDERLINES"
    in_memory_queries = {
        "id1": {
            "string_query": "SELECT ORDERLINEID AS ORDERLINEID_ALIAS2 FROM SALES.ORDERLINES WHERE ORDERLINEID = 1"
        },
        "id2": {"string_query": "SELECT ORDERLINEID, ORDERID FROM SALES.ORDERLINES "},
        "id3": {
            "string_query": "SELECT * FROM (SELECT ORDERLINEID AS ORDERLINEID_ALIAS3, ORDERID FROM SALES.ORDERLINES)"
        },
    }

    query_obj = parse_single(
        q=main_sql,
        schemas=schemas,
        dialects=["snowflake"],
        is_full_parse=True,
    )
    main_graph = get_slim_graph_from_edges(query_obj.get_edges(), remove_aliases=True)

    (
        identical_sqls_ids_in_memory,
        subset_ids_from_memory,
    ) = compare_query_to_list_of_queries(
        main_graph=main_graph,
        queries_ids=in_memory_queries.keys(),
        get_parsed_query=lambda sql_query: parse_single(
            q=sql_query,
            schemas=schemas,
            dialects=["snowflake"],
            is_full_parse=True,
            keep_string_vals=False,
        ),
        get_sql_string_by_id=lambda id: in_memory_queries[id]["string_query"],
        is_subgraph=True,
        remove_aliases=True,
    )

    assert len(identical_sqls_ids_in_memory) == 0
    assert len(subset_ids_from_memory) == 3
    assert subset_ids_from_memory[0] == "id1"
    assert subset_ids_from_memory[1] == "id2"
    assert subset_ids_from_memory[2] == "id3"


def test_find_identical_merge_queries(schemas):
    main_sql = """MERGE INTO PURCHASING.PURCHASEORDERLINES as PRCHS
                    USING(
                    Select DISTINCT ORDERID,
                        COALESCE(QUANTITY,0) as QUANTITY
                        FROM SALES.ORDERLINES
                        WHERE QUANTITY > 0
                        )  as SRC
                    ON PRCHS.PURCHASEORDERID=SRC.ORDERID
                    --!!!!!! BEGIN MERGE STATEMENT !!!!!!
                    WHEN MATCHED THEN UPDATE
                    SET
                    PRCHS.LASTEDITEDWHEN = SYSDATE()
                    WHEN NOT MATCHED THEN INSERT
                    (
                    PURCHASEORDERLINEID
                    ,LASTEDITEDBY
                    ,LASTEDITEDWHEN
                    )
                    VALUES
                    (
                    SRC.ORDERID
                    ,1
                    ,SYSDATE()
                    )"""
    in_memory_queries = {
        "id1": {
            "string_query": """MERGE 
            INTO PURCHASING.PURCHASEORDERLINES as PRCHS
                    USING(
                    Select DISTINCT ORDERID,
                        COALESCE(QUANTITY,0) as QUANTITY
                        FROM SALES.ORDERLINES
                        WHERE QUANTITY > 0
                        )  as SRC
                    ON PRCHS.PURCHASEORDERID=SRC.ORDERID
                    --!BEGIN MERGE STATEMENT !!!
                    WHEN MATCHED THEN UPDATE
                    SET
                    PRCHS.LASTEDITEDWHEN = SYSDATE()
                    WHEN NOT MATCHED THEN INSERT
                    (PURCHASEORDERLINEID,LASTEDITEDBY,LASTEDITEDWHEN)
                    VALUES
                    (SRC.ORDERID,1,SYSDATE())
                    """
        },
        "id2": {
            "string_query": """MERGE 
            INTO PURCHASING.PURCHASEORDERLINES as PRCHS
                    USING(
                    Select DISTINCT ORDERID, QUANTITY
                        FROM SALES.ORDERLINES
                        WHERE QUANTITY > 0
                        )  as SRC
                    ON PRCHS.PURCHASEORDERID=SRC.ORDERID
                    --!BEGIN MERGE STATEMENT !!!
                    WHEN MATCHED THEN UPDATE
                    SET
                    PRCHS.LASTEDITEDWHEN = SYSDATE()
                    WHEN NOT MATCHED THEN INSERT
                    (
                    PURCHASEORDERLINEID
                    ,LASTEDITEDBY
                    ,LASTEDITEDWHEN
                    )
                    VALUES
                    (
                    SRC.ORDERID
                    ,1
                    ,SYSDATE()
                    )
                    """
        },
        "id3": {
            "string_query": """MERGE 
            INTO PURCHASING.PURCHASEORDERLINES as PRCHS
                    USING(
                    Select DISTINCT ORDERID
                        FROM SALES.ORDERLINES
                        WHERE QUANTITY > 0
                        )  as SRC
                    ON PRCHS.PURCHASEORDERID=SRC.ORDERID
                    --!BEGIN MERGE STATEMENT !!!
                    WHEN MATCHED THEN UPDATE
                    SET
                    PRCHS.LASTEDITEDWHEN = SYSDATE()
                    WHEN NOT MATCHED THEN INSERT
                    (
                    PURCHASEORDERLINEID
                    ,LASTEDITEDWHEN
                    )
                    VALUES
                    (
                    SRC.ORDERID
                    ,SYSDATE()
                    )
                    """
        },
    }

    query_obj = parse_single(
        q=main_sql,
        schemas=schemas,
        dialects=["snowflake"],
        is_full_parse=True,
    )
    main_graph = get_slim_graph_from_edges(query_obj.get_edges())

    identical_sqls_ids_in_memory, subset_ids_from_memory = (
        compare_query_to_list_of_queries(
            main_graph=main_graph,
            queries_ids=in_memory_queries.keys(),
            get_parsed_query=lambda sql_query: parse_single(
                q=sql_query,
                schemas=schemas,
                dialects=["snowflake"],
                is_full_parse=True,
                keep_string_vals=False,
            ),
            get_sql_string_by_id=lambda id: in_memory_queries[id]["string_query"],
            is_subgraph=False,
            remove_aliases=False,
        )
    )

    assert len(identical_sqls_ids_in_memory) == 1
    assert identical_sqls_ids_in_memory[0] == "id1"
    assert len(subset_ids_from_memory) == 0


def test_find_identical_update_queries(schemas):
    main_sql = """UPDATE PURCHASING.PURCHASEORDERLINES f
                SET LASTEDITEDWHEN = s.LASTEDITEDWHEN,
                    LASTEDITEDBY = s.LASTEDITEDBY
                FROM SALES.ORDERLINES s
                WHERE f.PURCHASEORDERID = s.ORDERID
                AND f.LASTEDITEDWHEN >= :start_dt::DATE"""
    in_memory_queries = {
        "id1": {
            "string_query": """UPDATE PURCHASING.PURCHASEORDERLINES f
                SET LASTEDITEDWHEN = s.LASTEDITEDWHEN,
                    LASTEDITEDBY = s.LASTEDITEDBY
                FROM SALES.ORDERLINES s
                WHERE f.PURCHASEORDERID = s.ORDERID
                AND f.LASTEDITEDWHEN >= :start_dt::DATE
                    """
        },
    }

    query_obj = parse_single(
        q=main_sql,
        schemas=schemas,
        dialects=["snowflake"],
        is_full_parse=True,
    )
    main_graph = get_slim_graph_from_edges(query_obj.get_edges())

    identical_sqls_ids_in_memory, subset_ids_from_memory = (
        compare_query_to_list_of_queries(
            main_graph=main_graph,
            queries_ids=in_memory_queries.keys(),
            get_parsed_query=lambda sql_query: parse_single(
                q=sql_query,
                schemas=schemas,
                dialects=["snowflake"],
                is_full_parse=True,
                keep_string_vals=False,
            ),
            get_sql_string_by_id=lambda id: in_memory_queries[id]["string_query"],
            is_subgraph=False,
            remove_aliases=False,
        )
    )

    assert len(identical_sqls_ids_in_memory) == 1
    assert identical_sqls_ids_in_memory[0] == "id1"
    assert len(subset_ids_from_memory) == 0
