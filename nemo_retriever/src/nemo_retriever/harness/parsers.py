# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

# Legacy patterns (inprocess_pipeline / fused_pipeline)
DONE_RE = re.compile(r"\[done\]\s+(?P<files>\d+)\s+files,\s+(?P<pages>\d+)\s+pages\s+in\s+(?P<secs>[0-9.]+)s")
INGEST_ROWS_RE = re.compile(
    r"Ingestion complete\.\s+(?P<rows>\d+)\s+rows\s+proces+ed\s+in\s+(?P<secs>[0-9.]+)\s+seconds\.\s+"
    r"(?P<pps>[0-9.]+)\s+PPS"
)
# Matches both detection_summary.py ("Pages/sec (ingest only; excludes Ray startup and recall): N")
# and common.py ("Pages/sec (ingest only): N") formats
PAGES_PER_SEC_RE = re.compile(r"Pages/sec \([^)]+\):\s*(?P<val>[0-9.]+)")

# batch_pipeline print_run_summary() patterns
TOTAL_PAGES_RE = re.compile(r"Total pages processed:\s*(?P<pages>\d+)")
PAGES_PROCESSED_RE = re.compile(r"Pages processed:\s*(?P<pages>\d+)")
INGEST_ONLY_TIME_RE = re.compile(r"Ingestion only time:\s*(?P<secs>[0-9.]+)s\b")
INGEST_ONLY_PPS_RE = re.compile(r"Ingestion only PPS:\s*(?P<val>[0-9.]+)\b")
TOTAL_PPS_RE = re.compile(
    r"Total - Processed:\s*(?P<pages>\d+)\s+pages\s+in\s+(?P<secs>[0-9.]+)s.*@\s*(?P<pps>[0-9.]+)\s+PPS"
)

TOTAL_FILES_RE = re.compile(r"Total files processed:\s*(?P<files>\d+)")
METRIC_RE = re.compile(r"(?P<metric>[A-Za-z_]+@\d+):\s*(?P<val>[0-9.]+)\s*$")


_TAIL_BUFFER_LINES = 50


@dataclass
class StreamMetrics:
    """Metrics extracted from subprocess stdout via regex.

    The primary metric source is the structured JSON written by batch_pipeline
    to --metrics-output-file.  StreamMetrics is a fallback for legacy pipelines
    that don't write that file, and always provides the tail buffer for error
    reporting regardless.
    """

    files: int | None = None
    pages: int | None = None
    ingest_secs: float | None = None
    pages_per_sec_ingest: float | None = None
    rows_processed: int | None = None
    rows_per_sec_ingest: float | None = None
    recall_metrics: dict[str, float] = field(default_factory=dict)
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    _tail: deque[str] = field(default_factory=lambda: deque(maxlen=_TAIL_BUFFER_LINES))
    _metric_block: str | None = None

    @property
    def tail_lines(self) -> list[str]:
        """Return the last N lines of subprocess output (ANSI-stripped)."""
        return list(self._tail)

    def consume(self, line: str) -> None:
        stripped = line.rstrip()
        if stripped:
            self._tail.append(stripped)

        # Legacy: [done] N files, N pages in Ns
        done_match = DONE_RE.search(line)
        if done_match:
            self.files = int(done_match.group("files"))
            self.pages = int(done_match.group("pages"))
            self.ingest_secs = float(done_match.group("secs"))

        # Legacy: Ingestion complete. N rows processed in Ns. N PPS
        ingest_rows_match = INGEST_ROWS_RE.search(line)
        if ingest_rows_match:
            self.rows_processed = int(ingest_rows_match.group("rows"))
            self.ingest_secs = float(ingest_rows_match.group("secs"))
            self.rows_per_sec_ingest = float(ingest_rows_match.group("pps"))

        ingest_only_time_match = INGEST_ONLY_TIME_RE.search(line)
        if ingest_only_time_match:
            self.ingest_secs = float(ingest_only_time_match.group("secs"))

        ingest_only_pps_match = INGEST_ONLY_PPS_RE.search(line)
        if ingest_only_pps_match:
            self.pages_per_sec_ingest = float(ingest_only_pps_match.group("val"))

        pps_match = PAGES_PER_SEC_RE.search(line)
        if pps_match:
            self.pages_per_sec_ingest = float(pps_match.group("val"))

        total_files_match = TOTAL_FILES_RE.search(line)
        if total_files_match and self.files is None:
            self.files = int(total_files_match.group("files"))

        total_pages_match = TOTAL_PAGES_RE.search(line)
        if total_pages_match and self.pages is None:
            self.pages = int(total_pages_match.group("pages"))

        pages_processed_match = PAGES_PROCESSED_RE.search(line)
        if pages_processed_match and self.pages is None:
            self.pages = int(pages_processed_match.group("pages"))

        total_match = TOTAL_PPS_RE.search(line)
        if total_match:
            if self.pages is None:
                self.pages = int(total_match.group("pages"))

        # Metric block detection: "Recall metrics:" or "BEIR metrics:" headers
        # start a block where subsequent indented metric lines are captured.
        normalized_line = line.strip().lower()
        if "recall metrics" in normalized_line:
            self._metric_block = "recall"
            return

        if "beir metrics" in normalized_line:
            self._metric_block = "beir"
            return

        if self._metric_block is not None:
            metric_match = METRIC_RE.search(line)
            if metric_match:
                metric = metric_match.group("metric").lower()
                value = float(metric_match.group("val"))
                if self._metric_block == "recall":
                    self.recall_metrics[metric] = value
                else:
                    self.evaluation_metrics[metric] = value
                return

            if line.strip() and not line.startswith(" "):
                self._metric_block = None


def parse_stream_text(stdout_text: str) -> StreamMetrics:
    metrics = StreamMetrics()
    for line in stdout_text.splitlines():
        metrics.consume(line)
    return metrics
