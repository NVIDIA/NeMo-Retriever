# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch


def _load_media_interface():
    module_path = Path(__file__).resolve().parents[1] / "src" / "nemo_retriever" / "audio" / "media_interface.py"
    spec = importlib.util.spec_from_file_location("media_interface_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MediaDependencyAvailabilityTests(TestCase):
    def test_split_dependency_checks_report_each_missing_binary(self) -> None:
        media_interface = _load_media_interface()

        def fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name == "ffmpeg" else None

        with (
            patch.object(media_interface, "ffmpeg", SimpleNamespace()),
            patch.object(media_interface.shutil, "which", side_effect=fake_which),
        ):
            self.assertTrue(media_interface.is_ffmpeg_python_available())
            self.assertTrue(media_interface.is_ffmpeg_cli_available())
            self.assertFalse(media_interface.is_ffprobe_cli_available())
            self.assertEqual(media_interface.missing_media_dependencies(), ["ffprobe"])
            self.assertFalse(media_interface.is_media_available())

    def test_dependency_error_message_points_to_manual_and_container_installs(self) -> None:
        media_interface = _load_media_interface()

        with (
            patch.object(media_interface, "ffmpeg", None),
            patch.object(media_interface.shutil, "which", return_value=None),
        ):
            message = media_interface.media_dependency_error_message("VideoFrameActor")

        self.assertIn("VideoFrameActor requires media dependencies", message)
        self.assertIn("ffmpeg-python", message)
        self.assertIn("ffmpeg", message)
        self.assertIn("ffprobe", message)
        self.assertIn("apt-get update && apt-get install -y --no-install-recommends ffmpeg", message)
        self.assertIn("--build-arg INSTALL_FFMPEG=true", message)


if __name__ == "__main__":
    main()
