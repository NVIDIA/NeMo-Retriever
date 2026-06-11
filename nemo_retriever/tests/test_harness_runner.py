# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_retriever.harness.runner import _build_registration_payload


def test_runner_registration_payload_includes_validity_metadata():
    payload = _build_registration_payload(
        "ipp2-0545",
        {
            "host": "ipp2-0545",
            "gpu_type": "H100",
            "gpu_count": 1,
            "cpu_count": 64,
            "memory_gb": 512,
        },
        ["h100"],
        heartbeat_interval=30,
        valid_until="2026-06-11T14:30:00+00:00",
        lease_expires_at="2026-06-11T14:40:00+00:00",
        lease_id="lease-1",
        resource_name="ipp2-0545",
        orchestrator_run="run-1",
    )

    assert payload["valid_until"] == "2026-06-11T14:30:00+00:00"
    assert payload["lease_expires_at"] == "2026-06-11T14:40:00+00:00"
    assert payload["lease_id"] == "lease-1"
    assert payload["resource_name"] == "ipp2-0545"
    assert payload["orchestrator_run"] == "run-1"
