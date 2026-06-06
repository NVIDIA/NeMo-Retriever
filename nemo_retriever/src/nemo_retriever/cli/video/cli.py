# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Re-export of the video Typer app for ``retriever video ...``."""

from __future__ import annotations

from nemo_retriever.video.stage import app as app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
