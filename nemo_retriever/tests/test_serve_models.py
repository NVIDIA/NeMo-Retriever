# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.adapters.cli import serve_models as sm


def test_build_vllm_argv_uses_pooling_runner_and_trust_remote_code() -> None:
    argv = sm.build_vllm_argv("my/embed", "127.0.0.1", 8081)
    # argv[0] is the resolved vllm path; the rest is the spike-confirmed invocation.
    assert argv[1:] == [
        "serve",
        "my/embed",
        "--runner",
        "pooling",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
    ]
