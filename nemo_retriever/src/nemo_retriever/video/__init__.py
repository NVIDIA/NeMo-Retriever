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

from nemo_retriever.common.params import AudioVisualFuseParams
from nemo_retriever.common.params import VideoFrameParams
from nemo_retriever.common.params import VideoFrameTextDedupParams
from nemo_retriever.operators.extract.video.audio_visual_fuser import AudioVisualFuser
from nemo_retriever.operators.extract.video.frame_actor import VideoFrameActor
from nemo_retriever.operators.extract.video.frame_actor import dedup_video_frames
from nemo_retriever.operators.extract.video.frame_actor import video_path_to_frames_df
from nemo_retriever.operators.extract.video.ocr_actor import VideoFrameOCRActor
from nemo_retriever.operators.extract.video.ocr_actor import VideoFrameOCRCPUActor
from nemo_retriever.operators.extract.video.ocr_actor import VideoFrameOCRGPUActor
from nemo_retriever.operators.extract.video.split import VideoSplitActor
from nemo_retriever.operators.extract.video.split import video_asr_audio_chunk_params
from nemo_retriever.operators.extract.video.text_dedup import VideoFrameTextDedup

from nemo_retriever.cli.video.cli import app

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
    "VideoSplitActor",
    "video_asr_audio_chunk_params",
    "video_path_to_frames_df",
]
