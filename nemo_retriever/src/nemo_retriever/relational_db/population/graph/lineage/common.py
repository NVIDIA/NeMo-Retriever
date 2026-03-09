def extract_datasource_fields(list_ds):
    new_list = []
    for ds in list_ds:
        new_dict = dict()
        if "sql_id" in ds and len(ds["sql_id"]) > 0:
            new_dict["sql_id"] = ds["sql_id"]
        new_dict["columns"] = ds["columns"]
        new_dict["tags"] = ds["tags"]
        for key, val in ds["datasource"].items():
            if key != "other_fields":
                new_dict[key] = val
        new_list.append(new_dict)
    return new_list
