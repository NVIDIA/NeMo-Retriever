from shared.graph.services.sql_snippet import save_custom_snippet
from shared.graph.utils import remove_redundant_parentheses
from shared.graph.dal.attribute_dal import (
    save_attr,
    connect_attribute_to_file as dal_connect_attribute_to_file,
)
from shared.graph.dal.usages.semantic.terms_attributes import (
    update_single_attr_usage,
    update_single_term_usage,
)

import logging

logger = logging.getLogger("attribute.py")


def create_attr(
    account: str,
    attr_name: str,
    attr_description: str,
    term_id: str,
    snippet: str,
    schemas: dict,
    dialects: list[str],
    user_id: str = None,
    source_file_name: str | None = None,
    source_file_id: str | None = None,
    use_as_reference: bool = False,
):
    attr_res = save_attr(
        account,
        attr_name,
        attr_description,
        term_id,
        user_id,
        source_file_name,
        source_file_id,
        use_as_reference=use_as_reference,
    )
    attr_id = attr_res["id"]
    attr_snippet = remove_redundant_parentheses(snippet)
    if attr_snippet is not None and attr_snippet != "":
        snippet_id = save_custom_snippet(
            account_id=account,
            sql_snippet=attr_snippet,
            schemas=schemas,
            dialects=dialects,
            attribute_id=attr_id,
            user_id=user_id,
        )

        update_single_attr_usage(account, attr_res["id"])
        update_single_term_usage(account, term_id)

        return {"id": attr_res["id"], "snippet_id": snippet_id}

    return {"id": attr_res["id"]}


def connect_attribute_to_file(
    account_id: str,
    attr_id: str,
    description: str,
    source_file_id: str,
    source_file_name: str,
    use_as_reference: bool = False,
) -> None:
    """
    Connect an existing attribute to a file/document via attr_of_file edge.
    Does not modify any attribute/doc properties.
    """
    dal_connect_attribute_to_file(
        account_id=account_id,
        attr_id=attr_id,
        description=description,
        source_file_id=source_file_id,
        source_file_name=source_file_name,
        use_as_reference=use_as_reference,
    )
