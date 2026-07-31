#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initialize Nemotron OCR v2 from Hugging Face (nvidia/nemotron-ocr-v2).

Requires the ``nemotron-ocr`` package (>=2.0.0), PyTorch with CUDA, and Python 3.11–3.13.
Install: https://huggingface.co/nvidia/nemotron-ocr-v2
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and initialize NemotronOCRV2 from Hugging Face.",
    )
    parser.add_argument(
        "--lang",
        default="multi",
        help='Hub bundle selector: "multi" (default), "en", or "v1"/"legacy".',
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Local checkpoint directory with detector.pth, recognizer.pth, etc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

    if args.model_dir:
        ocr = NemotronOCRV2(model_dir=args.model_dir)
    else:
        ocr = NemotronOCRV2(lang=args.lang)

    print(f"Initialized NemotronOCRV2: {ocr!r}")


if __name__ == "__main__":
    main()
