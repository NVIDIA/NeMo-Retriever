# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "ci/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = load_script("hf_publication")
IDENTITY = "a" * 64


def file(version="2.0.2.dev1", python="cp312", arch="x86_64", identity=IDENTITY, **overrides):
    return {
        "filename": f"nemotron_ocr-{version}-1{identity}-{python}-{python}-manylinux_2_34_{arch}.whl",
        "size": 100,
        "upload_time_iso_8601": "2026-07-01T00:00:00Z",
        "yanked": False,
        **overrides,
    }


def plan(data, **overrides):
    return publisher.publication_plan(
        data,
        **{
            "identity": IDENTITY,
            "base_version": "2.0.2",
            "suffix": "999",
            "python_tag": "cp312",
            "arch": "x86_64",
            "require_sdist": False,
            **overrides,
        },
    )


def test_partial_publication_reuses_version_across_matrix_retries():
    data = {"releases": {"2.0.2.dev1": [file()]}}
    assert plan(data)["should_build"] == "false"
    missing = plan(data, python_tag="cp313", arch="aarch64")
    assert missing["should_build"] == "true"
    assert missing["version"] == "2.0.2.dev1"
    assert missing["nightly_suffix"] == "1"
    assert plan(data, force_build=True)["should_build"] == "true"


@pytest.mark.parametrize("python_tag", ["cp311", "cp312", "cp313"])
@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_complete_ocr_publication_skips_every_matrix_leg(python_tag, arch):
    files = [
        file(python=python, arch=platform)
        for python in ("cp311", "cp312", "cp313")
        for platform in ("x86_64", "aarch64")
    ]
    files.append(file(filename="nemotron_ocr-2.0.2.dev1.tar.gz"))
    assert (
        plan(
            {"releases": {"2.0.2.dev1": files}},
            python_tag=python_tag,
            arch=arch,
            require_sdist=python_tag == "cp312" and arch == "x86_64",
        )["should_build"]
        == "false"
    )


def test_pure_python_publication_also_skips_duplicates():
    files = [
        file(filename=f"example-2.0.2.dev1-1{IDENTITY}-py3-none-any.whl"),
        file(filename="example-2.0.2.dev1.tar.gz"),
    ]
    assert (
        plan({"releases": {"2.0.2.dev1": files}}, python_tag="py3", arch="any", require_sdist=True)["should_build"]
        == "false"
    )


def test_sdist_missing_is_rebuilt_only_by_its_responsible_leg():
    data = {"releases": {"2.0.2.dev1": [file()]}}
    assert plan(data, require_sdist=True)["should_build"] == "true"
    data["releases"]["2.0.2.dev1"].append(file(filename="nemotron_ocr-2.0.2.dev1.tar.gz"))
    assert plan(data, require_sdist=True)["should_build"] == "false"


def test_identity_and_version_train_both_must_match():
    data = {"releases": {"2.0.2.dev1": [file(identity="b" * 64)], "2.1.0.dev1": [file("2.1.0.dev1")]}}
    assert plan(data)["version"] == "2.0.2.dev999"
    assert plan(data)["should_build"] == "true"


def test_yanked_identity_is_not_republished_under_a_fresh_version():
    result = plan({"releases": {"2.0.2.dev1": [file(yanked=True)]}})
    assert result["should_build"] == "false"
    assert result["version"] == "2.0.2.dev1"


def test_stable_promotion_is_not_suppressed_by_matching_nightly():
    data = {"releases": {"2.0.2.dev1": [file()]}}
    assert plan(data, release_version="2.0.3")["version"] == "2.0.3"
    assert plan(data, release_version="2.0.3")["should_build"] == "true"


def test_legacy_wheels_without_identity_do_not_prove_a_match():
    data = {"releases": {"2.0.2.dev1": [file(filename="nemotron_ocr-2.0.2.dev1-cp312-cp312-linux_x86_64.whl")]}}
    assert plan(data)["should_build"] == "true"


def test_identity_tracks_source_and_publishing_files_but_not_unrelated_commits(tmp_path):
    for name in publisher.SETTINGS_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)

    def identity(revision="a" * 40, settings=None):
        return publisher.build_identity(
            "https://huggingface.co/nvidia/example", revision, settings or {"profile": "211"}, tmp_path
        )

    original = identity()
    (tmp_path / "unrelated.py").write_text("different unrelated source commit")
    assert identity() == original
    assert identity(revision="b" * 40) != original
    assert identity(settings={"profile": "213"}) != original
    for name in publisher.SETTINGS_FILES:
        path = tmp_path / name
        path.write_text(path.read_text() + " changed")
        assert identity() != original
        path.write_text(name)


def test_retention_uses_upload_dates_instead_of_dev_version_order():
    data = {
        "releases": {
            "2.0.2": [file("2.0.2", upload_time_iso_8601="2026-07-20T00:00:00Z")],
            "2.0.2.dev1": [file()],
            "2.0.2.dev2": [file(upload_time_iso_8601="2026-07-21T00:00:00Z")],
            "2.0.2.dev3": [file(), file(upload_time_iso_8601="2026-07-20T01:00:00Z")],
            "2.0.3rc1": [file()],
            "2.1.0": [file(yanked=True, upload_time_iso_8601="2026-08-01T00:00:00Z")],
        }
    }
    report = publisher.retention_report(data, "nemotron-ocr")
    assert report["stable_version"] == "2.0.2"
    assert [row["version"] for row in report["candidates"]] == ["2.0.2.dev1"]
    assert report["reclaimable_bytes"] == 100
    assert report["remaining_bytes"] == report["total_bytes"] - 100


def test_retention_without_a_stable_release_proposes_no_deletions():
    assert publisher.retention_report({"releases": {"2.0.2.dev1": [file()]}}, "example")["candidates"] == []


@pytest.mark.parametrize("status", [401, 429, 500])
def test_pypi_errors_fail_closed(monkeypatch, status):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError("https://pypi.org", status, "test", {}, None)

    monkeypatch.setattr(publisher.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.HTTPError):
        publisher.project_data("example")


def test_real_wheel_build_tag_preserves_contents_and_record(tmp_path, monkeypatch):
    from wheel.wheelfile import WheelFile

    builder = load_script("nightly_build_publish")
    wheel = tmp_path / "example-1.0.0-py3-none-any.whl"
    with WheelFile(wheel, "w") as archive:
        archive.writestr("example/__init__.py", "DATA = 42\n")
        archive.writestr("example-1.0.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: example\nVersion: 1.0.0\n")
        archive.writestr("example-1.0.0.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n")
    monkeypatch.setattr(builder, "_ensure_venv", lambda *args, **kwargs: Path(sys.executable))
    monkeypatch.setattr(builder, "_pip_install", lambda *args, **kwargs: None)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    builder._tag_publication_identity(tmp_path, IDENTITY)
    tagged = next(tmp_path.glob("*.whl"))
    assert publisher.wheel_matches({"filename": tagged.name}, IDENTITY, "py3", "any")
    with WheelFile(tagged) as archive:
        for name in archive.namelist():
            archive.read(name)  # WheelFile verifies RECORD hashes during reads.
        assert archive.read("example/__init__.py") == b"DATA = 42\n"
        assert f"Build: 1{IDENTITY}" in archive.read("example-1.0.0.dist-info/WHEEL").decode()
    assert not wheel.exists()
