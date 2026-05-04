# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Union
from typing import Dict
from typing import Optional
from typing import Tuple

import pandas as pd

from nv_ingest_api.internal.schemas.extract.extract_chart_schema import ChartExtractorSchema
from nv_ingest_api.internal.schemas.meta.ingest_job_schema import IngestTaskChartExtraction

logger = logging.getLogger(f"ray.{__name__}")


def extract_chart_data_from_image_internal(
    df_extraction_ledger: pd.DataFrame,
    task_config: Union[IngestTaskChartExtraction, Dict[str, Any]],
    extraction_config: ChartExtractorSchema,
    execution_trace_log: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Pass-through for chart extraction.

    Chart bounding boxes are detected by page-elements-v3, then OCR is run on the
    chart crops and the OCR text is emitted verbatim into the OCR-produced chart
    column upstream. This stage no longer performs additional inference.

    Parameters
    ----------
    df_extraction_ledger : pd.DataFrame
        DataFrame containing the content from which chart data is to be extracted.
    task_config : Dict[str, Any]
        Dictionary containing task properties and configurations.
    extraction_config : Any
        The validated configuration object for chart extraction (unused).
    execution_trace_log : Optional[Dict], optional
        Optional trace information for debugging or logging. Defaults to None.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        A tuple containing the (unchanged) DataFrame and the trace information.
    """
    _ = task_config  # Unused
    _ = extraction_config  # Unused

    if execution_trace_log is None:
        execution_trace_log = {}

    return df_extraction_ledger, {"trace_info": execution_trace_log}
