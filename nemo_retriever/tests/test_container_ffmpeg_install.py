# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest import TestCase, main


class ContainerFfmpegInstallTests(TestCase):
    def test_dockerfile_uses_default_off_apt_ffmpeg_build_arg(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        dockerfile = (repo_root / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn("ARG INSTALL_FFMPEG=false", dockerfile)
        self.assertIn("--build-arg INSTALL_FFMPEG=true", dockerfile)
        self.assertIn("apt-get install -y --no-install-recommends ffmpeg", dockerfile)
        self.assertNotIn("install_ffmpeg.sh", dockerfile)
        self.assertNotIn("ffmpeg.org/releases", dockerfile)


if __name__ == "__main__":
    main()
