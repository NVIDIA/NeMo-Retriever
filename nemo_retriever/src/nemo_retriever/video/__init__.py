# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Video pipeline: frame extraction (VideoFrameActor), full-frame OCR
(VideoFrameOCRActor), and audio+visual fusion (AudioVisualFuser).

Audio-from-video and ASR are still handled by the existing
:mod:`nemo_retriever.audio` actors; this module adds the frame-OCR and
fusion branches that run alongside them when ``input_type=video``.
"""

from __future__ import annotations

from nemo_retriever.params import AudioVisualFuseParams
from nemo_retriever.params import VideoFrameParams
from nemo_retriever.params import VideoFrameTextDedupParams
from nemo_retriever.params import VideoOCRParams
from nemo_retriever.video.audio_visual_fuser import AudioVisualFuser
from nemo_retriever.video.frame_actor import VideoFrameActor
from nemo_retriever.video.frame_actor import dedup_video_frames
from nemo_retriever.video.frame_actor import video_path_to_frames_df
from nemo_retriever.video.ocr_actor import VideoFrameOCRActor
from nemo_retriever.video.ocr_actor import VideoFrameOCRCPUActor
from nemo_retriever.video.ocr_actor import VideoFrameOCRGPUActor
from nemo_retriever.video.split import VideoSplitActor
from nemo_retriever.video.text_dedup import VideoFrameTextDedup

from .cli import app

__all__ = [
    "app",
    "AudioVisualFuser",
    "AudioVisualFuseParams",
    "dedup_video_frames",
    "VideoFrameActor",
    "VideoFrameOCRActor",
    "VideoFrameOCRCPUActor",
    "VideoFrameOCRGPUActor",
    "VideoFrameParams",
    "VideoFrameTextDedup",
    "VideoFrameTextDedupParams",
    "VideoOCRParams",
    "VideoSplitActor",
    "video_path_to_frames_df",
]
