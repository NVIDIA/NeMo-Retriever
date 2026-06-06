# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical content-type discriminator strings for video-pipeline rows.

Rows flowing through the video graph carry a ``_content_type`` value (both
as a top-level column and inside ``metadata``) so each stage can filter to
the rows it owns and pass everything else through. Keeping these literals
in one module avoids typos across the actors that read and write them.
"""

from __future__ import annotations

# Per-utterance ASR text from the audio track.
AUDIO = "audio"

# Single OCR'd video frame.
VIDEO_FRAME = "video_frame"

# Fused row: an audio utterance joined with concurrent frame OCR text.
AUDIO_VISUAL = "audio_visual"
