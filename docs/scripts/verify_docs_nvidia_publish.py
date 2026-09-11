#!/usr/bin/env python3
"""Verify docs.nvidia.com serves freshly published HTML at canonical URLs.

Checks canonical URLs without cache-busting query strings. Retries until the
response includes required marker text or the retry budget expires.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


@dataclass(frozen=True)
class ProbeResult:
    url: str
    status: int | None
    last_modified: str | None
    cache_status: str | None
    marker_found: bool
    error: str | None = None


def _fetch(url: str, marker: str, timeout_s: float) -> ProbeResult:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "nrl-docs-nvidia-publish-verify/1.0",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
            headers = response.headers
            return ProbeResult(
                url=url,
                status=response.status,
                last_modified=headers.get("Last-Modified"),
                cache_status=headers.get("Akamai-Cache-Status"),
                marker_found=marker in body,
            )
    except urllib.error.HTTPError as exc:
        return ProbeResult(
            url=url,
            status=exc.code,
            last_modified=exc.headers.get("Last-Modified") if exc.headers else None,
            cache_status=exc.headers.get("Akamai-Cache-Status") if exc.headers else None,
            marker_found=False,
            error=str(exc),
        )
    except urllib.error.URLError as exc:
        return ProbeResult(
            url=url,
            status=None,
            last_modified=None,
            cache_status=None,
            marker_found=False,
            error=str(exc),
        )


def _parse_last_modified(value: str | None):
    if not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None


def _format_result(result: ProbeResult) -> str:
    parts = [
        f"url={result.url}",
        f"status={result.status}",
        f"last_modified={result.last_modified}",
        f"akamai_cache_status={result.cache_status}",
        f"marker_found={result.marker_found}",
    ]
    if result.error:
        parts.append(f"error={result.error}")
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        action="append",
        required=True,
        help="Canonical docs.nvidia.com URL to verify (repeatable).",
    )
    parser.add_argument(
        "--marker",
        required=True,
        help="Substring that must appear in the response body.",
    )
    parser.add_argument(
        "--min-last-modified-utc",
        default="",
        help="Optional ISO-8601 UTC timestamp. Fail if Last-Modified is older.",
    )
    parser.add_argument(
        "--retry-seconds",
        type=int,
        default=600,
        help="Total retry budget before failing (default: 600).",
    )
    parser.add_argument(
        "--retry-interval-seconds",
        type=int,
        default=30,
        help="Seconds between attempts (default: 30).",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=30.0,
        help="Per-request timeout (default: 30).",
    )
    args = parser.parse_args()

    min_last_modified = None
    if args.min_last_modified_utc:
        min_last_modified = datetime.fromisoformat(args.min_last_modified_utc.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )

    deadline = time.monotonic() + args.retry_seconds
    attempt = 0
    last_results: list[ProbeResult] = []

    while True:
        attempt += 1
        last_results = [_fetch(url, args.marker, args.request_timeout_seconds) for url in args.url]
        all_markers = all(result.marker_found for result in last_results)
        all_fresh = True
        if min_last_modified is not None:
            for result in last_results:
                modified = _parse_last_modified(result.last_modified)
                if modified is None:
                    all_fresh = False
                    break
                if modified.tzinfo is None:
                    modified = modified.replace(tzinfo=timezone.utc)
                if modified < min_last_modified:
                    all_fresh = False
                    break

        if all_markers and all_fresh:
            print(f"docs.nvidia.com verification passed after {attempt} attempt(s).")
            for result in last_results:
                print(_format_result(result))
            return 0

        if time.monotonic() >= deadline:
            print("docs.nvidia.com verification failed.", file=sys.stderr)
            for result in last_results:
                print(_format_result(result), file=sys.stderr)
            return 1

        print(
            f"Attempt {attempt} not ready; retrying in {args.retry_interval_seconds}s...",
            file=sys.stderr,
        )
        for result in last_results:
            print(_format_result(result), file=sys.stderr)
        time.sleep(args.retry_interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
