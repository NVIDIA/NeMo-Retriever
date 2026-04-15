# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from unittest.mock import patch


def test_decorators_import_does_not_start_manager():
    module_name = "nv_ingest_api.internal.primitives.nim.model_interface.decorators"
    sys.modules.pop(module_name, None)

    import multiprocessing as mp

    with patch.object(mp, "Manager", side_effect=AssertionError("Manager called during import")):
        module = importlib.import_module(module_name)

    assert module is not None
