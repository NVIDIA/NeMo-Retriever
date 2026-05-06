# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy resolution of concrete VDB operator classes (avoids import cycles)."""


def get_vdb_op_cls(vdb_op: str):
    """
    Lazily import and return the VDB operation class for the given op string.
    Returns the class if found, else raises ValueError.
    """

    available_vdb_ops = ["lancedb"]

    if vdb_op == "lancedb":
        from nemo_retriever.vdb.lancedb import LanceDB

        return LanceDB

    raise ValueError(f"Invalid vdb_op: {vdb_op}. Available vdb_ops - {available_vdb_ops}.")
