# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the stock NRL harness after installing the experiment-scoped tail."""

from full_pipeline_transport import install

install()

from nemo_retriever.harness.cli import app


if __name__ == "__main__":
    app()
