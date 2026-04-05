"""Unit tests for queries_parser.py

Tests cover:
- pre_process: SQL string normalisation
- dispatch_sqls: routing to the correct sub-parser
- parse_single: dialect fallback, error cases, and happy-path SELECT

All sqloxide calls and sub-parser modules are mocked so the tests run without
the native C extension and without a fully-wired graph stack.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out every heavy module that `queries_parser` and its sub-parsers pull
# in at import time. This avoids needing sqloxide installed or a live graph.
# ---------------------------------------------------------------------------
_BASE = "nemo_retriever.tabular_data.ingestion.graph.parsers.sql"

for _mod_name in [
    "sqloxide",
    f"{_BASE}.sql_select_parser",
    f"{_BASE}.sql_insert_into_parser",
    f"{_BASE}.sql_update_table_parser",
    f"{_BASE}.sql_merge_table_parser",
    f"{_BASE}.sql_create_table_parser",
    f"{_BASE}.sql_view_parser",
]:
    if _mod_name not in sys.modules:
        _m = ModuleType(_mod_name)
        if _mod_name == "sqloxide":
            _m.parse_sql = MagicMock(return_value=[{"Query": {}}])
        else:
            _m.build_query_obj = MagicMock(return_value=True)
        sys.modules[_mod_name] = _m

from nemo_retriever.tabular_data.ingestion.graph.parsers.sql.queries_parser import (  # noqa: E402
    pre_process,
    dispatch_sqls,
    parse_single,
)
from nemo_retriever.tabular_data.ingestion.graph.model.query import (  # noqa: E402
    NotSelectSqlTypeError,
    NotValidSyntaxError,
    UnsupportedQueryError,
)

_PARSER_MODULE = "nemo_retriever.tabular_data.ingestion.graph.parsers.sql"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_query_mock() -> MagicMock:
    """Return a minimal Query-like mock."""
    qm = MagicMock()
    qm.default_schema = "public"
    return qm


def _parsed_select() -> list[dict]:
    return [{"Query": {"body": {"Select": {}}}}]


def _parsed_insert() -> list[dict]:
    return [{"Insert": {"source": {}}}]


def _parsed_update() -> list[dict]:
    return [{"Update": {}}]


def _parsed_merge() -> list[dict]:
    return [{"Merge": {}}]


def _parsed_delete() -> list[dict]:
    return [{"Delete": {}}]


def _parsed_create_table_as_select() -> list[dict]:
    return [{"CreateTable": {"name": [{"value": "t"}], "query": {}, "clone": None, "temporary": False, "transient": False}}]


def _parsed_create_view() -> list[dict]:
    return [{"CreateView": {}}]


# ---------------------------------------------------------------------------
# pre_process
# ---------------------------------------------------------------------------


class TestPreProcess:
    def test_plain_sql_unchanged(self):
        assert pre_process("SELECT 1") == "SELECT 1"

    def test_insert_overwrite_rewritten(self):
        result = pre_process("INSERT OVERWRITE INTO t SELECT 1")
        assert result.startswith("INSERT INTO")
        assert "OVERWRITE" not in result

    def test_merge_into_identifier_rewritten(self):
        result = pre_process("MERGE INTO IDENTIFIER(foo) USING bar ON foo.id = bar.id")
        assert "IDENTIFIER(" not in result
        assert "MERGE INTO (" in result

    def test_u_and_prefix_stripped(self):
        result = pre_process("SELECT U&'hello'")
        assert "U&'" not in result

    def test_escaped_double_quotes_normalised(self):
        # \\" in the raw string is a backslash followed by a double-quote;
        # pre_process should replace \" with "
        result = pre_process('SELECT \\"id\\"')
        assert '\\"' not in result


# ---------------------------------------------------------------------------
# dispatch_sqls – routing (sub-parsers are mocked)
# ---------------------------------------------------------------------------


class TestDispatchSqls:
    def _dispatch(self, parsed_query, sql_text, sql_type, **kwargs):
        query_obj = _make_query_mock()
        dispatch_sqls(
            parsed_query=parsed_query,
            sql_text=sql_text,
            query_obj=query_obj,
            keep_string_values=False,
            schemas={},
            sql_type=sql_type,
            **kwargs,
        )
        return query_obj

    def test_select_routes_to_select_parser(self):
        with patch(f"{_PARSER_MODULE}.sql_select_parser.build_query_obj", return_value=True) as mock_build:
            self._dispatch(_parsed_select(), "SELECT 1", sql_type="query")
            mock_build.assert_called_once()

    def test_semantic_routes_to_select_parser(self):
        with patch(f"{_PARSER_MODULE}.sql_select_parser.build_query_obj", return_value=True) as mock_build:
            self._dispatch(_parsed_select(), "SELECT 1", sql_type="semantic")
            mock_build.assert_called_once()

    def test_insert_with_select_routes_to_insert_parser(self):
        sql = "INSERT INTO t SELECT id FROM s"
        with patch(f"{_PARSER_MODULE}.sql_insert_into_parser.build_query_obj", return_value=True) as mock_build:
            self._dispatch(_parsed_insert(), sql, sql_type="insert")
            mock_build.assert_called_once()

    def test_update_with_from_routes_to_update_parser(self):
        sql = "UPDATE t SET a = 1 FROM s WHERE t.id = s.id"
        with patch(f"{_PARSER_MODULE}.sql_update_table_parser.build_query_obj", return_value=True) as mock_build:
            self._dispatch(_parsed_update(), sql, sql_type="update")
            mock_build.assert_called_once()

    def test_merge_with_select_routes_to_merge_parser(self):
        sql = "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET t.a = s.a"
        with patch(f"{_PARSER_MODULE}.sql_merge_table_parser.build_query_obj", return_value=True) as mock_build:
            self._dispatch(_parsed_merge(), sql, sql_type="merge")
            mock_build.assert_called_once()

    def test_create_table_as_select_routes_to_create_table_parser(self):
        sql = "CREATE TABLE t AS SELECT id FROM s"
        with patch(f"{_PARSER_MODULE}.sql_create_table_parser.build_query_obj", return_value=True) as mock_build:
            self._dispatch(_parsed_create_table_as_select(), sql, sql_type="createtable")
            mock_build.assert_called_once()

    def test_create_view_routes_to_view_parser(self):
        sql = "CREATE VIEW v AS SELECT 1"
        with patch(f"{_PARSER_MODULE}.sql_view_parser.build_query_obj", return_value=True) as mock_build:
            self._dispatch(_parsed_create_view(), sql, sql_type="createview")
            mock_build.assert_called_once()

    def test_delete_raises_unsupported(self):
        with pytest.raises(UnsupportedQueryError):
            self._dispatch(_parsed_delete(), "DELETE FROM t WHERE id = 1", sql_type="delete")

    def test_unknown_type_raises_unsupported(self):
        with pytest.raises(UnsupportedQueryError):
            self._dispatch(_parsed_select(), "SELECT 1", sql_type="unknownxyz")

    def test_allow_only_select_blocks_insert(self):
        with pytest.raises(NotSelectSqlTypeError):
            self._dispatch(
                _parsed_insert(),
                "INSERT INTO t SELECT id FROM s",
                sql_type="insert",
                allow_only_select=True,
            )

    def test_allow_only_select_permits_select(self):
        with patch(f"{_PARSER_MODULE}.sql_select_parser.build_query_obj", return_value=True) as mock_build:
            self._dispatch(_parsed_select(), "SELECT 1", sql_type="query", allow_only_select=True)
            mock_build.assert_called_once()

    def test_insert_without_select_does_not_route_to_insert_parser(self):
        """INSERT without a SELECT sub-query falls through to UnsupportedQueryError."""
        sql = "INSERT INTO t VALUES (1)"
        with pytest.raises(UnsupportedQueryError):
            self._dispatch(_parsed_insert(), sql, sql_type="insert")

    def test_update_without_from_does_not_route_to_update_parser(self):
        """UPDATE without FROM or SELECT falls through to UnsupportedQueryError."""
        sql = "UPDATE t SET a = 1 WHERE id = 1"
        with pytest.raises(UnsupportedQueryError):
            self._dispatch(_parsed_update(), sql, sql_type="update")


# ---------------------------------------------------------------------------
# parse_single
# ---------------------------------------------------------------------------


class TestParseSingle:
    """parse_single wires together pre_process, sqloxide, and dispatch_sqls."""

    _PARSE_SQL = "nemo_retriever.tabular_data.ingestion.graph.parsers.sql.queries_parser.parse_sql"

    def test_returns_query_object(self):
        from nemo_retriever.tabular_data.ingestion.graph.model.query import Query

        with patch(self._PARSE_SQL, return_value=_parsed_select()):
            with patch(f"{_PARSER_MODULE}.sql_select_parser.build_query_obj", return_value=True):
                result = parse_single(q="SELECT 1", schemas={}, dialects=["ansi"])
        assert isinstance(result, Query)

    def test_invalid_sql_all_dialects_raises_not_valid_syntax(self):
        """When every dialect raises, NotValidSyntaxError is propagated."""
        with patch(self._PARSE_SQL, side_effect=Exception("bad sql")):
            with pytest.raises(NotValidSyntaxError):
                parse_single(q="NOT SQL !!!", schemas={}, dialects=["ansi", "duckdb"])

    def test_first_dialect_fails_second_succeeds(self):
        """parse_single retries subsequent dialects on parse failure."""
        call_count = [0]

        def _parse_sql_side_effect(sql, dialect):
            call_count[0] += 1
            if dialect == "ansi":
                raise Exception("ansi failed")
            return _parsed_select()

        with patch(self._PARSE_SQL, side_effect=_parse_sql_side_effect):
            with patch(f"{_PARSER_MODULE}.sql_select_parser.build_query_obj", return_value=True):
                result = parse_single(q="SELECT 1", schemas={}, dialects=["ansi", "duckdb"])

        assert result is not None
        assert call_count[0] == 2  # tried ansi, then duckdb

    def test_default_schema_propagated(self):
        with patch(self._PARSE_SQL, return_value=_parsed_select()):
            with patch(f"{_PARSER_MODULE}.sql_select_parser.build_query_obj", return_value=True):
                result = parse_single(
                    q="SELECT 1",
                    schemas={},
                    dialects=["ansi"],
                    default_schema="my_schema",
                )
        assert result.default_schema == "my_schema"

    def test_sql_type_set_on_query_node(self):
        with patch(self._PARSE_SQL, return_value=_parsed_select()):
            with patch(f"{_PARSER_MODULE}.sql_select_parser.build_query_obj", return_value=True):
                result = parse_single(q="SELECT 1", schemas={}, dialects=["ansi"])
        assert result.sql_node.props["sql_type"] == "query"

    def test_allow_only_select_rejects_insert(self):
        with patch(self._PARSE_SQL, return_value=_parsed_insert()):
            with pytest.raises(NotSelectSqlTypeError):
                parse_single(
                    q="INSERT INTO t SELECT id FROM s",
                    schemas={},
                    dialects=["ansi"],
                    allow_only_select=True,
                )

    def test_allow_only_select_accepts_select(self):
        with patch(self._PARSE_SQL, return_value=_parsed_select()):
            with patch(f"{_PARSER_MODULE}.sql_select_parser.build_query_obj", return_value=True):
                result = parse_single(
                    q="SELECT 1",
                    schemas={},
                    dialects=["ansi"],
                    allow_only_select=True,
                )
        assert result is not None
