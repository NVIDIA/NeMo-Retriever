from shared.graph.dal.usages.data.queries import calculate_queries_usage
from shared.graph.dal.usages.data.columns import (
    calculate_column_queries_and_usage,
)
from shared.graph.dal.usages.data.tables import (
    calculate_tables_queries_and_usage,
)
import logging

logger = logging.getLogger("usages/calculate_data_layer_usages.py")


def calculate_data_layer_usage(account_id: str):
    logger.info("Starting Data Layer usages calcualtions:")
    calculate_queries_usage(account_id)
    calculate_column_queries_and_usage(account_id)
    calculate_tables_queries_and_usage(account_id)
    logger.info("Completed Data Layer usages calcualtions:")
