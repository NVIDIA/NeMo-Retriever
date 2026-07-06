# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = ["app", "main"]


def __getattr__(name: str):
    """Load the root application only when callers request its public exports.

    Internal modules such as ``cli.ingest_workflow`` are imported by the
    harness. Eagerly importing ``cli.main`` here makes that dependency loop
    back into a partially initialized harness package.
    """
    if name in __all__:
        from importlib import import_module

        main_module = import_module("nemo_retriever.cli.main")
        return getattr(main_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
