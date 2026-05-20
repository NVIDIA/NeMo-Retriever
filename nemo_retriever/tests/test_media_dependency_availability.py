# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import importlib.util
import tempfile
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
    def test_optional_ffmpeg_import_only_swallows_missing_package(self) -> None:
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ffmpeg":
                raise TypeError("broken ffmpeg import")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(TypeError, "broken ffmpeg import"):
                _load_media_interface()

    def test_probe_media_handles_os_error_when_ffmpeg_python_missing(self) -> None:
        media_interface = _load_media_interface()

        with patch.object(media_interface, "ffmpeg", None):
            result = media_interface.MediaInterface().probe_media(
                Path("/tmp/does-not-exist-for-nemo-retriever-tests.mp4"),
                split_interval=10,
                split_type=media_interface.SplitType.SIZE,
            )

        self.assertEqual(result, (None, None, None))

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
        self.assertIn("INSTALL_FFMPEG=true", message)
        self.assertIn("service.installFfmpeg=true", message)
        self.assertNotIn("--build-arg INSTALL_FFMPEG=true", message)

    def test_dependency_error_message_is_coherent_when_nothing_is_missing(self) -> None:
        media_interface = _load_media_interface()

        with (
            patch.object(media_interface, "ffmpeg", SimpleNamespace()),
            patch.object(media_interface.shutil, "which", return_value="/usr/bin/tool"),
        ):
            message = media_interface.media_dependency_error_message("VideoFrameActor")

        self.assertEqual(message, "VideoFrameActor media dependencies are available.")
        self.assertEqual(message, message.rstrip())

    def test_unknown_dependency_names_are_reported_missing(self) -> None:
        media_interface = _load_media_interface()

        with (
            patch.object(media_interface, "ffmpeg", SimpleNamespace()),
            patch.object(media_interface.shutil, "which", return_value="/usr/bin/tool"),
        ):
            self.assertEqual(media_interface.missing_media_dependencies(("future-codec",)), ["future-codec"])
            message = media_interface.media_dependency_error_message("Media processing", required=("future-codec",))

        self.assertIn("missing: future-codec", message)

    def test_run_ffmpeg_dependency_error_wraps_internal_label(self) -> None:
        media_interface = _load_media_interface()

        for ffmpeg_module in (None, SimpleNamespace()):
            with self.subTest(ffmpeg_module=ffmpeg_module):
                with (
                    patch.object(media_interface, "ffmpeg", ffmpeg_module),
                    patch.object(media_interface.shutil, "which", return_value=None),
                ):
                    with self.assertRaises(RuntimeError) as error:
                        media_interface._run_ffmpeg(object(), label="split", input_path="/tmp/input.mp4")

                message = str(error.exception)
                self.assertIn("FFmpeg operation 'split' requires media dependencies", message)
                self.assertNotIn("split requires media dependencies", message)

    def test_get_audio_from_video_does_not_require_ffprobe(self) -> None:
        media_interface = _load_media_interface()

        class FakeFFmpegStream:
            def output(self, *_args, **_kwargs):
                return self

            def overwrite_output(self):
                return self

        stream = FakeFFmpegStream()
        fake_ffmpeg = SimpleNamespace(input=lambda _path: stream, Error=Exception)

        def fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name == "ffmpeg" else None

        with (
            patch.object(media_interface, "ffmpeg", fake_ffmpeg),
            patch.object(media_interface.shutil, "which", side_effect=fake_which),
            patch.object(media_interface, "_run_ffmpeg") as run_ffmpeg,
        ):
            result = media_interface._get_audio_from_video("/tmp/input.mp4", "/tmp/output.mp3")

        self.assertEqual(result, Path("/tmp/output.mp3"))
        run_ffmpeg.assert_called_once_with(stream, label="extract_audio", input_path="/tmp/input.mp4")

    def test_extract_frames_does_not_require_ffprobe(self) -> None:
        media_interface = _load_media_interface()

        class FakeFFmpegStream:
            def output(self, *_args, **_kwargs):
                return self

            def overwrite_output(self):
                return self

        stream = FakeFFmpegStream()
        fake_ffmpeg = SimpleNamespace(input=lambda _path: stream, Error=Exception)

        def fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name == "ffmpeg" else None

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(media_interface, "ffmpeg", fake_ffmpeg),
                patch.object(media_interface.shutil, "which", side_effect=fake_which),
                patch.object(media_interface, "_run_ffmpeg") as run_ffmpeg,
            ):
                result = media_interface.MediaInterface().extract_frames("/tmp/input.mp4", tmpdir)

        self.assertEqual(result, [])
        run_ffmpeg.assert_called_once_with(stream, label="extract_frames", input_path="/tmp/input.mp4")

    def test_video_frame_loader_does_not_require_ffprobe(self) -> None:
        from nemo_retriever.audio import media_interface
        from nemo_retriever.params import VideoFrameParams
        from nemo_retriever.video import frame_actor

        def fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name == "ffmpeg" else None

        row = {
            "path": "/tmp/input.mp4",
            "source_path": "/tmp/input.mp4",
            "image_b64": "AA==",
            "page_number": 0,
            "metadata": {},
            "bytes": b"",
            "_content_type": "video_frame",
        }

        with (
            patch.object(media_interface, "ffmpeg", SimpleNamespace()),
            patch.object(media_interface.shutil, "which", side_effect=fake_which),
            patch.object(frame_actor, "_extract_one", return_value=[row]) as extract_one,
        ):
            actor = frame_actor.VideoFrameActor(VideoFrameParams())
            df = frame_actor.video_path_to_frames_df("/tmp/input.mp4", VideoFrameParams())

        self.assertTrue(actor._params.enabled)
        self.assertEqual(len(df), 1)
        extract_one.assert_called_once()

    def test_video_split_frame_only_does_not_require_ffprobe(self) -> None:
        from nemo_retriever.audio import media_interface
        from nemo_retriever.params import AudioChunkParams, VideoFrameParams
        from nemo_retriever.video.split import VideoSplitActor

        def fake_which(name: str) -> str | None:
            return f"/usr/bin/{name}" if name == "ffmpeg" else None

        with (
            patch.object(media_interface, "ffmpeg", SimpleNamespace()),
            patch.object(media_interface.shutil, "which", side_effect=fake_which),
        ):
            actor = VideoSplitActor(
                audio_chunk_params=AudioChunkParams(enabled=False),
                video_frame_params=VideoFrameParams(enabled=True),
            )

        self.assertFalse(actor._audio_chunk_params.enabled)
        self.assertTrue(actor._video_frame_params.enabled)


if __name__ == "__main__":
    main()
