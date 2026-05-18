# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""
Local pipeline stage6: thin proxy for VDB upload CLI.

This module intentionally contains no configuration logic. It re-exports the
`nemo_retriever.vdb.stage` Typer application so arguments provided to:

  `retriever local stage6 ...`

are handled by `nemo_retriever.vdb.stage`.
"""

from nemo_retriever.vdb.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
