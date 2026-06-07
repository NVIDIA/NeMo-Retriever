# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import regex
from datetime import timezone

import pandas as pd

from nemo_retriever.tabular_data.ingestion.model.reserved_words import TableTypes


def flat_list_recursive(nested_list):
    output = []
    for i in nested_list:
        if isinstance(i, list):
            temp = flat_list_recursive(i)
            for j in temp:
                output.append(j)
        else:
            output.append(i)
    return output


def remove_redundant_parentheses(text):
    r = r"s/(\(|^)\K(\((((?2)|[^()])*)\))(?=\)|$)//"
    if r[0] != "s":
        raise SyntaxError('Missing "s"')
    d = r[1]
    r = r.split(d)
    if len(r) != 4:
        raise SyntaxError("Wrong number of delimiters")
    flags = 0
    count = 1
    for f in r[3]:
        if f == "g":
            count = 0
        else:
            flags |= {
                "i": regex.IGNORECASE,
                "m": regex.MULTILINE,
                "s": regex.DOTALL,
                "x": regex.VERBOSE,
            }[f]
    s = r[2]
    r = r[1]
    while 1:
        m = regex.subn(r, s, text, count, flags)
        text = m[0]
        if m[1] == 0:
            break

    return text


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def normalize_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and type a tables DataFrame.

    Accepts rows from either a SQL connector (``information_schema``) or Neo4j
    graph reload (``get_schema_tables``). Connector-specific columns such as
    ``owner`` are dropped; graph-only columns such as ``id`` and ``database``
    are preserved.
    """
    types = {
        "table_schema": "category",
        "table_name": "string",
        "table_type": "category",
        "created": "string",
        "description": "string",
    }
    base_columns = list(types.keys())
    df = df.copy() if df is not None and not df.empty else pd.DataFrame(columns=base_columns)
    if df.empty:
        return df

    for key in base_columns:
        if key not in df.columns:
            df[key] = pd.NA

    df["table_type"] = df["table_type"].fillna(TableTypes.BASE_TABLE)
    df = df.astype(dtype=types)

    if "created" in df:
        df["created"] = pd.to_datetime(df["created"], utc=True, format="mixed")
        df["created"] = df["created"].apply(
            lambda x: x.tz_convert(timezone.utc).replace(microsecond=0) if pd.notna(x) else x
        )

    for extra_col in ("owner",):
        if extra_col in df.columns:
            df = df.drop(columns=[extra_col])

    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and type a columns DataFrame. Expects a DataFrame only."""
    types = {
        "table_schema": "category",
        "table_name": "category",
        "column_name": "string",
        "ordinal_position": "Int16",
        "data_type": "category",
        "is_nullable": "category",
        "description": "string",
    }
    df = df.copy() if df is not None and not df.empty else pd.DataFrame(columns=list(types.keys()))
    if df.empty:
        return df

    for key in types.keys():
        if key not in df.columns:
            df[key] = pd.NA

    df["ordinal_position"] = pd.to_numeric(df["ordinal_position"])
    df = df.astype(dtype=types)

    return df
