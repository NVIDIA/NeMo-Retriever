import regex
from datetime import timezone

import pandas as pd


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
    r = r"s/(\(|^)\K(\((((?2)|[^()])*)\))(?=\)|$)/\3/"
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
    # z = 0

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


def load_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and type a tables DataFrame. Expects a DataFrame only."""
    types = {
        "database": "category",
        "schema": "category",
        "table_name": "string",
        "table_type": "category",
        "row_count": "Int64",
        "size": "Int64",
        "retention_time": "Int8",
        "created": "string",
        "last_altered": "string",
        "comment": "string",
    }
    df = df.copy() if df is not None and not df.empty else pd.DataFrame()
    if df.empty:
        return df

    if "row_count" not in df.columns:
        df["row_count"] = list(range(1, len(df) + 1))
    if "size" not in df.columns:
        df["size"] = 0
    if "retention_time" not in df.columns:
        df["retention_time"] = 0

    for key in types.keys():
        if key not in df.columns:
            df[key] = pd.NA

    df["row_count"] = pd.to_numeric(df["row_count"])
    df["size"] = pd.to_numeric(df["size"])
    df["retention_time"] = pd.to_numeric(df["retention_time"])
    df = df.astype(dtype=types)

    # parse_dates / optional date fields
    if "last_altered" in df:
        df["last_altered"] = pd.to_datetime(df["last_altered"], utc=True, format="mixed")
        df["last_altered"] = df["last_altered"].apply(lambda x: x.tz_convert(timezone.utc).replace(microsecond=0))
    if "created" in df:
        df["created"] = pd.to_datetime(df["created"], utc=True, format="mixed")
        df["created"] = df["created"].apply(lambda x: x.tz_convert(timezone.utc).replace(microsecond=0))

    # Shkolar did it in playtika
    if "owner" in df:
        df = df.drop(columns=["owner"])

    return df


def load_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and type a columns DataFrame. Expects a DataFrame only."""
    types = {
        "database": "category",
        "schema": "category",
        "table_name": "category",
        "column_name": "string",
        "ordinal_position": "Int16",
        "data_type": "category",
        "default": "category",
        "is_nullable": "category",
        "length": "Int64",
        "scale": "Int16",
        "comment": "string",
    }
    df = df.copy() if df is not None and not df.empty else pd.DataFrame()
    if df.empty:
        return df

    for key in types.keys():
        if key not in df.columns:
            df[key] = pd.NA

    df["ordinal_position"] = pd.to_numeric(df["ordinal_position"])
    df["length"] = pd.to_numeric(df["length"])
    df["scale"] = pd.to_numeric(df["scale"])
    df = df.astype(dtype=types)

    return df


def load_fks(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and type a foreign-keys DataFrame. Expects a DataFrame only."""
    types = {
        "created_on": "string",
        "pk_database_name": "string",
        "pk_schema_name": "string",
        "pk_table_name": "string",
        "pk_column_name": "string",
        "fk_database_name": "string",
        "fk_schema_name": "string",
        "fk_table_name": "string",
        "fk_column_name": "string",
        "key_sequence": "string",
        "update_rule": "string",
        "delete_rule": "string",
        "fk_name": "string",
        "pk_name": "string",
        "deferrability": "string",
        "rely": "boolean",
    }
    df = df.copy() if df is not None and not df.empty else pd.DataFrame()
    if df.empty:
        return df
    for key in types.keys():
        if key not in df.columns:
            df[key] = pd.NA
    df = df.astype(dtype=types)
    return df


def load_pks(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and type a primary-keys DataFrame. Expects a DataFrame only."""
    types = {
        "created_on": "string",
        "database_name": "string",
        "schema_name": "string",
        "table_name": "string",
        "column_name": "string",
        "key_sequence": "string",
        "constraint_name": "string",
        "rely": "string",
    }
    df = df.copy() if df is not None and not df.empty else pd.DataFrame()
    if df.empty:
        return df
    for key in types.keys():
        if key not in df.columns:
            df[key] = pd.NA
    df = df.astype(dtype=types)
    return df
