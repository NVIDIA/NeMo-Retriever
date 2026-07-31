# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest import SkipTest, TestCase, main


def _read_dockerfile() -> str:
    dockerfile = Path(__file__).resolve().parents[2] / "Dockerfile"
    if not dockerfile.is_file():
        raise SkipTest(f"Required file not present in this test environment: {dockerfile}")
    return dockerfile.read_text(encoding="utf-8")


class ContainerMultimediaDependencyTests(TestCase):
    def test_base_image_explicitly_installs_libcairo2(self) -> None:
        dockerfile = _read_dockerfile()

        self.assertIn("      libcairo2 \\\n", dockerfile)

    def test_both_service_targets_install_multimedia_extra(self) -> None:
        dockerfile = _read_dockerfile()

        self.assertIn('uv pip install -e "./nemo_retriever[service,multimedia]"', dockerfile)
        self.assertIn('uv pip install -e "./nemo_retriever[service,local,multimedia]"', dockerfile)


if __name__ == "__main__":
    main()
