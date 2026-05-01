# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch


def tensor_to_numpy_cpu(t: torch.Tensor) -> np.ndarray:
    """
    Return a host ``numpy`` array with ordinary storage from *t*.

    vmap / functorch and some tensor subclasses can yield tensors that do not
    expose storage to ``.numpy()``; a ``detach().clone()`` materializes first.
    """
    return t.detach().clone().contiguous().cpu().numpy()
