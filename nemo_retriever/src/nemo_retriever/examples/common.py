# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers used by multiple example pipeline scripts."""

from __future__ import annotations

from typing import Optional

from nemo_retriever.vdb.lancedb_read import estimate_processed_pages

__all__ = ["estimate_processed_pages", "print_pages_per_second"]


def print_pages_per_second(
    processed_pages: Optional[int],
    ingest_elapsed_s: float,
    *,
    label: str = "ingest only",
) -> None:
    """Print a throughput summary line."""
    if ingest_elapsed_s <= 0:
        print("Pages/sec: unavailable (ingest elapsed time was non-positive).")
        return
    if processed_pages is None:
        print(f"Pages/sec: unavailable (could not estimate processed pages). Ingest time: {ingest_elapsed_s:.2f}s")
        return

    pps = processed_pages / ingest_elapsed_s
    print(f"Pages processed: {processed_pages}")
    print(f"Pages/sec ({label}): {pps:.2f}")
