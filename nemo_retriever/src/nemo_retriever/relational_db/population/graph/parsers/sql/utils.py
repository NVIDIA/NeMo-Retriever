keep_string_values = False


def get_name_only(names_dict):
    if len(names_dict) == 2:
        name = names_dict[1]["value"]
    elif len(names_dict) == 1:
        name = names_dict[0]["value"]
    else:
        name = names_dict[2]["value"]
    return name


def get_name_and_parent_name(names_dict):
    if len(names_dict) == 2:
        parent_name = names_dict[0]["value"]
        name = names_dict[1]["value"]
    elif len(names_dict) == 1:
        if "." not in names_dict[0]["value"]:
            parent_name = ""
            name = names_dict[0]["value"]
        else:
            parent_name = names_dict[0]["value"].split(".")[0]
            name = names_dict[0]["value"].split(".")[1]
    else:
        parent_name = names_dict[1]["value"]
        name = names_dict[2]["value"]
    return parent_name, name


def restore_name(names_dict):
    if len(names_dict) == 2:
        parent_name = names_dict[0]["value"]
        name = names_dict[1]["value"]
        return f"{parent_name}.{name}"
    elif len(names_dict) == 1:
        parent_name = ""
        name = names_dict[0]["value"]
        return name
    grandparent_name = names_dict[0]["value"]
    parent_name = names_dict[1]["value"]
    name = names_dict[2]["value"]
    return f"{grandparent_name}.{parent_name}.{name}"


def get_key_recursive(search_dict, field):
    """
    Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    - modified from: https://stackoverflow.com/a/20254842
    """
    fields_found = []

    for key, value in search_dict.items():
        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_key_recursive(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_key_recursive(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)
    return fields_found
