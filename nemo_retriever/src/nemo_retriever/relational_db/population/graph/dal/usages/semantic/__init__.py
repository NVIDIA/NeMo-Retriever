from shared.graph.dal.usages.semantic.analysis import calculate_analyses_usage
from shared.graph.dal.usages.semantic.terms_attributes import (
    calculate_attributes_usage,
    calculate_terms_usage,
)

import logging

logger = logging.getLogger("usages/calculate_semantic_layer_usages.py")


def calculate_semantic_layer_usage(account_id: str):
    logger.info("Starting Semantic Layer usages calcualtions:")
    calculate_attributes_usage(account_id)
    calculate_terms_usage(account_id)
    calculate_analyses_usage(account_id)
    logger.info("Completed Semantic Layer usages calcualtions:")
