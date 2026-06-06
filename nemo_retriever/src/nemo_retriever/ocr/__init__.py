# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.ocr.gpu_ocr import OCRActor
from nemo_retriever.ocr.shared import ocr_page_elements

__all__ = [
    "OCRActor",
    "ocr_page_elements",
]
