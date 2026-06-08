# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.common.api.internal.primitives.nim.nim_client import NimClient
from nemo_retriever.common.api.internal.primitives.nim.nim_client import get_nim_client_manager
from nemo_retriever.common.api.internal.primitives.nim.nim_model_interface import ModelInterface

__all__ = ["NimClient", "ModelInterface", "get_nim_client_manager"]
