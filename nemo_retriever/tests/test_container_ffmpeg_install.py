# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest import SkipTest, TestCase, main


def _read_required_dockerfile(path: Path) -> str:
    if not path.is_file():
        raise SkipTest(f"Dockerfile not present in this test environment: {path}")
    return path.read_text(encoding="utf-8")


class ContainerFfmpegInstallTests(TestCase):
    def test_dockerfile_policy_test_skips_when_repo_root_not_available(self) -> None:
        missing_dockerfile = Path("/tmp/nemo-retriever-missing-root/Dockerfile")

        with self.assertRaises(SkipTest):
            _read_required_dockerfile(missing_dockerfile)

    def test_dockerfile_uses_default_off_apt_ffmpeg_build_arg(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        dockerfile = _read_required_dockerfile(repo_root / "Dockerfile")

        self.assertIn("ARG INSTALL_FFMPEG=false", dockerfile)
        self.assertIn("--build-arg INSTALL_FFMPEG=true", dockerfile)
        self.assertIn("apt-get install -y --no-install-recommends ffmpeg", dockerfile)
        self.assertNotIn("install_ffmpeg.sh", dockerfile)
        self.assertNotIn("ffmpeg.org/releases", dockerfile)


if __name__ == "__main__":
    main()
