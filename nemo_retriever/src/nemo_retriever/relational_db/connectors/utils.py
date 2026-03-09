def group_by_queries(queries):
    queries = (
        queries.groupby(["database", "schema", "query_text"], dropna=False)["end_time"]
        .agg(["max", "count"])
        .reset_index()
        .rename({"max": "end_time"}, axis=1)
        .sort_values(by="count", ascending=False)
    )
    return queries[["database", "schema", "end_time", "count", "query_text"]]
