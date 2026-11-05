#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan idempotent HF publications and report superseded PyPI nightlies."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILES = (
    ".github/workflows/huggingface-nightly.yml",
    "ci/scripts/nightly_build_publish.py",
    "ci/scripts/hf_publication.py",
)


def project_data(project: str) -> dict:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{project}/json", headers={"User-Agent": "nrl-hf-publication/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return {"releases": {}}
        raise


def resolve_source(url: str, revision: str = "") -> str:
    if revision:
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("Source revision must be a full Git SHA")
        return revision.lower()
    output = subprocess.check_output(["git", "ls-remote", "--exit-code", url, "HEAD"], text=True)
    return resolve_source(url, output.split()[0])


def build_identity(source_url: str, revision: str, settings: dict, root: Path = ROOT) -> str:
    """Exclude run IDs, upload credentials, and unrelated NRL commits."""
    payload = {
        "source_url": source_url,
        "source_revision": revision,
        "settings": settings,
        "publisher_files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SETTINGS_FILES},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def wheel_matches(file: dict, identity: str, python_tag: str = "", arch: str = "") -> bool:
    try:
        _, _, build, tags = parse_wheel_filename(file["filename"])
    except InvalidWheelFilename:
        return False
    if build != (1, identity):
        return False
    return not python_tag or any(
        tag.interpreter == python_tag
        and (tag.abi == python_tag if python_tag.startswith("cp") else tag.abi == "none")
        and (tag.platform == "any" if arch == "any" else tag.platform.endswith(f"_{arch}"))
        for tag in tags
    )


def publication_plan(
    data: dict,
    *,
    identity: str,
    base_version: str,
    suffix: str,
    python_tag: str,
    arch: str,
    require_sdist: bool,
    release_version: str = "",
    force_build: bool = False,
) -> dict:
    candidates = []
    for version, files in data["releases"].items():
        try:
            parsed = Version(version)
        except InvalidVersion:
            continue
        if release_version:
            eligible = parsed == Version(release_version)
        else:
            eligible = parsed.is_devrelease and parsed.base_version == Version(base_version).base_version
        if eligible and any(wheel_matches(file, identity) for file in files):
            candidates.append((parsed, version, files))
    if candidates:
        _, version, files = max(candidates)
    else:
        version = release_version or f"{base_version}.dev{suffix}"
        files = []
    complete = any(wheel_matches(file, identity, python_tag, arch) for file in files)
    if require_sdist:
        complete = complete and any(file["filename"].endswith(".tar.gz") for file in files)
    parsed = Version(version)
    return {
        "build_id": identity,
        "version": version,
        "nightly_suffix": str(parsed.dev) if parsed.dev is not None else suffix,
        "should_build": "true" if force_build or not complete else "false",
    }


def upload_time(file: dict) -> dt.datetime:
    return dt.datetime.fromisoformat(file["upload_time_iso_8601"].replace("Z", "+00:00"))


def retention_report(data: dict, project: str) -> dict:
    """Only propose dev releases whose every file predates the latest final release."""
    stable = []
    total = 0
    for version, files in data["releases"].items():
        total += sum(file["size"] for file in files)
        try:
            parsed = Version(version)
        except InvalidVersion:
            continue
        if files and not parsed.is_prerelease and not parsed.local:
            if not all(file.get("yanked") for file in files):
                stable.append((parsed, version, files))
    result = {"project": project, "total_bytes": total, "candidates": [], "reclaimable_bytes": 0}
    if not stable:
        return result
    _, version, files = max(stable)
    cutoff = min(upload_time(file) for file in files)
    result.update(stable_version=version, stable_uploaded=cutoff.isoformat())
    for version, files in data["releases"].items():
        try:
            parsed = Version(version)
        except InvalidVersion:
            continue
        if parsed.is_devrelease and files and max(upload_time(file) for file in files) < cutoff:
            size = sum(file["size"] for file in files)
            result["candidates"].append(
                {
                    "version": version,
                    "bytes": size,
                    "files": len(files),
                    "manage_url": f"https://pypi.org/manage/project/{project}/release/{version}/",
                }
            )
            result["reclaimable_bytes"] += size
    result["candidates"].sort(key=lambda row: Version(row["version"]))
    result["remaining_bytes"] = total - result["reclaimable_bytes"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("resolve-source", help="Resolve one source SHA for the complete OCR matrix")
    source.add_argument("--repo-url", required=True)
    source.add_argument("--source-revision", default="")
    source.add_argument("--github-output", type=Path, required=True)
    plan = commands.add_parser("plan", help="Reuse published build identities and resume incomplete releases")
    plan.add_argument("--project", required=True)
    plan.add_argument("--repo-url", required=True)
    plan.add_argument("--source-revision", default="")
    plan.add_argument("--base-version", required=True)
    plan.add_argument("--nightly-suffix", required=True)
    plan.add_argument("--profile", default="")
    plan.add_argument("--release-version", default="")
    plan.add_argument("--python-tag", required=True)
    plan.add_argument("--arch", required=True)
    plan.add_argument("--require-sdist", action="store_true")
    plan.add_argument(
        "--force-build", action="store_true", help="Build qualification artifacts even if already published"
    )
    plan.add_argument("--github-output", type=Path, required=True)
    report = commands.add_parser("retention-report", help="List obsolete nightlies; never deletes releases")
    report.add_argument("--project", default="nemotron-ocr")
    report.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "retention-report":
        rendered = json.dumps(retention_report(project_data(args.project), args.project), indent=2) + "\n"
        if args.output:
            args.output.write_text(rendered)
        print(rendered, end="")
        return
    revision = resolve_source(args.repo_url, args.source_revision)
    outputs = {"source_revision": revision}
    if args.command == "plan":
        identity = build_identity(
            args.repo_url,
            revision,
            {
                "project": args.project,
                "base_version": args.base_version,
                "profile": args.profile,
                "release_version": args.release_version,
            },
        )
        outputs.update(
            publication_plan(
                project_data(args.project),
                identity=identity,
                base_version=args.base_version,
                suffix=args.nightly_suffix,
                python_tag=args.python_tag,
                arch=args.arch,
                require_sdist=args.require_sdist,
                release_version=args.release_version,
                force_build=args.force_build,
            )
        )
    with args.github_output.open("a") as output:
        for key, value in outputs.items():
            output.write(f"{key}={value}\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
