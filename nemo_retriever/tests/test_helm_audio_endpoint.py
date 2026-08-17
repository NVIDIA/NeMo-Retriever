# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for manual Helm Parakeet ASR endpoint configuration.

Enabling ``nimOperator.audio`` deploys the Parakeet NIM but does not
auto-wire ``audio_grpc_endpoint`` into the retriever service. Callers
must set ``serviceConfig.nimEndpoints.audioGrpcEndpoint`` explicitly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest import SkipTest, TestCase


_AUDIO_GRPC_ENDPOINT = "audio:50051"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _helm_template(*args: str) -> subprocess.CompletedProcess[str]:
    helm = shutil.which("helm")
    if helm is None:
        raise SkipTest("`helm` binary not available in this environment.")
    return subprocess.run(
        [
            helm,
            "template",
            "retriever",
            str(_repo_root() / "nemo_retriever/helm"),
            "--set",
            "ngcImagePullSecret.create=false",
            "--set",
            "ngcApiSecret.create=false",
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


class HelmAudioEndpointTests(TestCase):
    def test_values_expose_manual_audio_endpoint(self) -> None:
        values = (_repo_root() / "nemo_retriever/helm/values.yaml").read_text(encoding="utf-8")
        self.assertIn('audioGrpcEndpoint: ""', values)
        self.assertIn("Not auto-wired when nimOperator.audio is enabled", values)

    def test_enabling_audio_nim_does_not_auto_wire_service_config(self) -> None:
        rendered = _helm_template(
            "--api-versions",
            "apps.nvidia.com/v1alpha1",
            "--set",
            "nimOperator.audio.enabled=true",
        )
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertIn("audio_grpc_endpoint: null", rendered.stdout)
        self.assertRegex(rendered.stdout, r"(?m)^kind: NIMService$")
        self.assertRegex(rendered.stdout, r"(?m)^  name: audio$")

    def test_explicit_audio_endpoint_renders_into_service_config(self) -> None:
        rendered = _helm_template(
            "--api-versions",
            "apps.nvidia.com/v1alpha1",
            "--set",
            "nimOperator.audio.enabled=true",
            "--set",
            f"serviceConfig.nimEndpoints.audioGrpcEndpoint={_AUDIO_GRPC_ENDPOINT}",
        )
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        self.assertIn(
            f'audio_grpc_endpoint: "{_AUDIO_GRPC_ENDPOINT}"',
            rendered.stdout,
        )
