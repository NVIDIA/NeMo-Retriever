from .reserved_words import Labels, DataTypes
from typing import Union


class Snippet:
    def __init__(
        self,
        id: str,
        account_id: str,
        attribute_id: str,
        attribute_name: str,
        term_name: str,
        data_item_id: str,
        data_item_label: Union[Labels.COLUMN, Labels.ALIAS, Labels.SQL],
        data_item_data_type: list[DataTypes],
        props: dict[str, str],
        tables_names_ids: list[dict[str, str]],
        source_table_name_ids: list[dict[str, str]] = None,
    ):
        self.id = id
        self.account_id = account_id
        self.attribute_id = attribute_id
        self.attribute_name = attribute_name
        self.term_name = term_name
        self.data_item_id = data_item_id
        self.data_item_label = data_item_label
        self.data_item_data_type = (
            data_item_data_type
            if isinstance(data_item_data_type, list)
            else [data_item_data_type]
        )
        self.props = props
        self.tables_names_ids = tables_names_ids
        self.tables_ids = [table["id"] for table in tables_names_ids]
        self.source_table_name_ids = source_table_name_ids
