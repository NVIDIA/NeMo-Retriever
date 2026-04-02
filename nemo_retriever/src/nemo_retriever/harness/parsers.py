# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass, field

DONE_RE = re.compile(r"\[done\]\s+(?P<files>\d+)\s+files,\s+(?P<pages>\d+)\s+pages\s+in\s+(?P<secs>[0-9.]+)s")
INGEST_ROWS_RE = re.compile(
    r"Ingestion complete\.\s+(?P<rows>\d+)\s+rows\s+proces+ed\s+in\s+(?P<secs>[0-9.]+)\s+seconds\.\s+"
    r"(?P<pps>[0-9.]+)\s+PPS"
)
INGEST_ONLY_TIME_RE = re.compile(r"Ingestion only time:\s*(?P<secs>[0-9.]+)s\b")
INGEST_ONLY_PPS_RE = re.compile(r"Ingestion only PPS:\s*(?P<val>[0-9.]+)\b")
PAGES_PER_SEC_RE = re.compile(r"Pages/sec \(ingest only; excludes Ray startup and recall\):\s*(?P<val>[0-9.]+)")
METRIC_RE = re.compile(r"(?P<metric>[A-Za-z_]+@\d+):\s*(?P<val>[0-9.]+)\s*$")


@dataclass
class StreamMetrics:
    files: int | None = None
    pages: int | None = None
    ingest_secs: float | None = None
    pages_per_sec_ingest: float | None = None
    rows_processed: int | None = None
    rows_per_sec_ingest: float | None = None
    recall_metrics: dict[str, float] = field(default_factory=dict)
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    _metric_block: str | None = None

    def consume(self, line: str) -> None:
        done_match = DONE_RE.search(line)
        if done_match:
            self.files = int(done_match.group("files"))
            self.pages = int(done_match.group("pages"))
            self.ingest_secs = float(done_match.group("secs"))

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
