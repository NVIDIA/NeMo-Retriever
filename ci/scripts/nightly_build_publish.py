#!/usr/bin/env python3
"""
Nightly builder/publisher for Hugging Face-hosted NVIDIA OSS repos.

Behavior:
- Clones a HF git repo with Git LFS smudge disabled (so large weights are not downloaded).
- Patches a PEP 440 dev version into pyproject.toml or setup.cfg (default suffix: UTC
  YYYYMMDD; override with NIGHTLY_DATE_SUFFIX or NIGHTLY_DATE_YYYYMMDD, e.g. from CI).
- Builds sdist + wheel via `python -m build`.
- Optional ``--auditwheel-repair`` rewrites ``linux_*`` wheels to ``manylinux_*`` for PyPI.
- Optionally uploads to (Test)PyPI via twine.

This is intentionally best-effort across heterogeneous upstream projects.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _nightly_suffix() -> str:
    """Return the numeric part after ``.dev`` in the nightly PEP 440 version.

    Default is UTC ``YYYYMMDD`` so parallel matrix legs (e.g. x86_64 + aarch64) publish
    under the same version. Override with ``NIGHTLY_DATE_SUFFIX`` or
    ``NIGHTLY_DATE_YYYYMMDD`` (e.g. set once per workflow run in CI).
    """
    forced = os.environ.get("NIGHTLY_DATE_SUFFIX") or os.environ.get("NIGHTLY_DATE_YYYYMMDD")
    if forced:
        return forced
    return _dt.datetime.now(_dt.UTC).strftime("%Y%m%d")


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv(venv_dir: Path, *, system_site_packages: bool) -> Path:
    """
    Ensure a local venv exists and return its python path.

    We intentionally do all pip/build/twine operations inside this venv to avoid
    system-Python restrictions (e.g. PEP 668 externally-managed environments).
    """
    marker = venv_dir / ".orch-system-site-packages"
    py = _venv_python(venv_dir)
    if py.exists():
        # If caller changes system_site_packages setting, recreate the venv to ensure
        # the correct site-packages visibility.
        has_marker = marker.exists()
        if has_marker == system_site_packages:
            return py
        shutil.rmtree(venv_dir)

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "venv"]
    if system_site_packages:
        cmd.append("--system-site-packages")
    cmd.append(str(venv_dir))
    _run(cmd)
    py = _venv_python(venv_dir)
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    if system_site_packages:
        marker.write_text("1", encoding="utf-8")
    return py


def _pep440_nightly(base_version: str, suffix: str) -> str:
    """
    Convert a base version to a nightly dev version.
    Examples:
      1.2.3 -> 1.2.3.dev20260127
      1.2.3+local -> 1.2.3.dev20260127
    """
    base = base_version.split("+", 1)[0].strip()
    base = re.sub(r"\.dev\d+$", "", base)
    return f"{base}.dev{suffix}"


def _patch_pyproject_version(repo_dir: Path) -> bool:
    pyproject = repo_dir / "pyproject.toml"
    if not pyproject.exists():
        return False

    text = _read_text(pyproject)
    # Match: version = "x.y.z" under [project] (best-effort, not a full TOML parser)
    m = re.search(r'(?ms)^\[project\]\s.*?^\s*version\s*=\s*"([^"]+)"\s*$', text)
    if not m:
        return False

    old_version = m.group(1)
    new_version = _pep440_nightly(old_version, _nightly_suffix())
    if new_version == old_version:
        return False

    text2 = text[: m.start(1)] + new_version + text[m.end(1) :]
    _write_text(pyproject, text2)
    print(f"Patched pyproject.toml version: {old_version} -> {new_version}")
    return True


def _patch_hatch_build_force_platform_wheel(project_dir: Path) -> bool:
    """
    Hatchling may emit py3-none-any for extension builds unless the hook sets
    build_data[\"pure_python\"] = False and build_data[\"infer_tag\"] = True so the
    wheel tag matches the current interpreter/platform. Patch upstream hatch_build.py
    when we recognize the Nemotron OCR layout.
    """
    path = project_dir / "hatch_build.py"
    if not path.is_file():
        return False
    text = _read_text(path)
    has_pp = 'build_data["pure_python"]' in text or "build_data['pure_python']" in text
    has_it = 'build_data["infer_tag"]' in text or "build_data['infer_tag']" in text
    if has_pp and has_it:
        return False
    if "CustomBuildHook" not in text or "def initialize" not in text:
        return False
    needle = "def initialize(self, version: str, build_data: dict) -> None:"
    idx = text.find(needle)
    if idx < 0:
        return False
    body_start = idx + len(needle)
    insert_at = body_start
    while insert_at < len(text) and text[insert_at] in " \t":
        insert_at += 1
    if insert_at < len(text) and text[insert_at] == "\n":
        insert_at += 1

    block = '        build_data["pure_python"] = False\n' '        build_data["infer_tag"] = True\n'

    if not has_pp and not has_it:
        patched = text[:insert_at] + block + text[insert_at:]
        _write_text(path, patched)
        print("Patched hatch_build.py: pure_python=False, infer_tag=True")
        return True

    if has_pp and not has_it:
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        inserted = False
        for line in lines:
            out.append(line)
            if inserted:
                continue
            if ('build_data["pure_python"]' in line or "build_data['pure_python']" in line) and "False" in line:
                out.append('        build_data["infer_tag"] = True\n')
                inserted = True
        if not inserted:
            return False
        _write_text(path, "".join(out))
        print('Patched hatch_build.py: build_data["infer_tag"] = True')
        return True

    # has_it and not has_pp: insert pure_python line before first infer_tag line
    lines = text.splitlines(keepends=True)
    out = []
    inserted = False
    for line in lines:
        if not inserted and ('build_data["infer_tag"]' in line or "build_data['infer_tag']" in line):
            out.append('        build_data["pure_python"] = False\n')
            inserted = True
        out.append(line)
    if not inserted:
        return False
    _write_text(path, "".join(out))
    print('Patched hatch_build.py: build_data["pure_python"] = False')
    return True


def _patch_setup_cfg_version(repo_dir: Path) -> bool:
    setup_cfg = repo_dir / "setup.cfg"
    if not setup_cfg.exists():
        return False

    text = _read_text(setup_cfg)
    # Match in [metadata]: version = x.y.z
    m = re.search(r"(?ms)^\[metadata\]\s.*?^\s*version\s*=\s*([^\s#]+)\s*$", text)
    if not m:
        return False

    old_version = m.group(1).strip().strip('"').strip("'")
    new_version = _pep440_nightly(old_version, _nightly_suffix())
    if new_version == old_version:
        return False

    text2 = text[: m.start(1)] + new_version + text[m.end(1) :]
    _write_text(setup_cfg, text2)
    print(f"Patched setup.cfg version: {old_version} -> {new_version}")
    return True


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _clone_repo(repo_url: str, dest: Path) -> None:
    env = os.environ.copy()
    # Do not download large LFS objects.
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    _run(["git", "clone", "--depth", "1", repo_url, str(dest)], env=env)


def _looks_like_python_project(project_dir: Path) -> bool:
    return (project_dir / "pyproject.toml").exists() or (project_dir / "setup.cfg").exists()


def _auto_project_subdir(repo_dir: Path, repo_id: str) -> str:
    """
    Best-effort detection for repos that keep the Python project in a nested
    same-named directory (e.g. repo 'llama-nemotron-embed-1b-v2' containing
    'llama-nemotron-embed-1b-v2/pyproject.toml').
    """
    if _looks_like_python_project(repo_dir):
        return ""
    candidate = repo_dir / repo_id
    if candidate.is_dir() and _looks_like_python_project(candidate):
        return repo_id
    return ""


def _apply_build_env_overrides(env: dict[str, str], build_env: list[str]) -> dict[str, str]:
    """
    Apply KEY=VALUE overrides to the environment dict.
    """
    for item in build_env:
        if "=" not in item:
            raise ValueError(f"--build-env must be KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"--build-env must have a key, got: {item!r}")
        env[k] = v
    return env


def _ensure_tmpdir(env: dict[str, str]) -> dict[str, str]:
    """
    Ensure temp files (including `python -m build` isolated envs) land on a
    writable filesystem with enough space.

    In some CI/container setups, `/tmp` can be a small tmpfs, which breaks builds
    for projects that need large build dependencies (e.g. PyTorch wheels).
    """
    tmpdir = env.get("TMPDIR") or os.environ.get("TMPDIR")
    if not tmpdir:
        tmpdir = str(Path.cwd() / ".orch-tmp")
        env["TMPDIR"] = tmpdir
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    return env


def _pip_install(py: Path, packages: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    if not packages:
        return
    _run([str(py), "-m", "pip", "install", "--upgrade", *packages], cwd=cwd, env=env)


def _build(
    project_dir: Path,
    out_dir: Path,
    *,
    build_env: list[str],
    no_isolation: bool,
    venv_system_site_packages: bool,
    venv_pip_install: list[str],
) -> None:
    venv_dir = Path(os.environ.get("ORCH_VENV_DIR", ".venv-build"))
    py = _ensure_venv(venv_dir, system_site_packages=venv_system_site_packages)

    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env = _apply_build_env_overrides(env, build_env)
    env = _ensure_tmpdir(env)

    # Clean build outputs
    for p in (project_dir / "dist", project_dir / "build"):
        if p.exists():
            shutil.rmtree(p)

    _pip_install(py, ["build"], cwd=project_dir, env=env)
    _pip_install(py, venv_pip_install, cwd=project_dir, env=env)

    cmd = [str(py), "-m", "build", "--sdist", "--wheel"]
    if no_isolation:
        cmd.append("--no-isolation")
    _run(cmd, cwd=project_dir, env=env)

    dist_dir = project_dir / "dist"
    if not dist_dir.exists():
        raise RuntimeError("Build succeeded but dist/ not found")
    out_dir.mkdir(parents=True, exist_ok=True)
    for artifact in dist_dir.iterdir():
        shutil.copy2(artifact, out_dir / artifact.name)


def _auditwheel_repair_dist_dir(dist_dir: Path, *, exclude_libs: list[str] | None = None) -> None:
    """
    Rewrite linux_* wheels to manylinux_* so TestPyPI/PyPI accept the upload.
    Requires ``patchelf`` on PATH (e.g. apt install patchelf).

    *exclude_libs* is a list of shared library basenames (e.g. ``libtorch_cpu.so``)
    that auditwheel should NOT bundle.  This is needed for wheels that link against
    PyTorch: the torch libs are a runtime dependency, not something to vendor.
    """
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        return

    venv_dir = Path(os.environ.get("ORCH_VENV_DIR", ".venv-build"))
    py = _ensure_venv(venv_dir, system_site_packages=False)
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env = _ensure_tmpdir(env)
    _pip_install(py, ["auditwheel"], cwd=dist_dir.parent, env=env)

    repair_out = dist_dir / ".auditwheel-out"
    _ensure_clean_dir(repair_out)
    exclude_flags: list[str] = []
    for lib in exclude_libs or []:
        exclude_flags += ["--exclude", lib]
    cmd = [
        str(py),
        "-m",
        "auditwheel",
        "repair",
        *[str(w) for w in wheels],
        "-w",
        str(repair_out),
        *exclude_flags,
    ]
    _run(cmd, env=env)

    repaired = sorted(repair_out.glob("*.whl"))
    if not repaired:
        raise RuntimeError("auditwheel repair produced no wheels")

    for w in wheels:
        w.unlink()

    for rw in repaired:
        dest = dist_dir / rw.name
        shutil.move(str(rw), dest)
        print(f"auditwheel: {dest.name}")

    shutil.rmtree(repair_out)


def _twine_upload(
    dist_dir: Path,
    repository_url: str,
    token: str,
    *,
    skip_existing: bool,
    verbose: bool,
) -> None:
    venv_dir = Path(os.environ.get("ORCH_VENV_DIR", ".venv-build"))
    # Twine doesn't need system site packages; keep it off by default.
    py = _ensure_venv(venv_dir, system_site_packages=False)

    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = token
    env = _ensure_tmpdir(env)
    _run([str(py), "-m", "pip", "install", "--upgrade", "twine"], env=env)
    cmd = [
        str(py),
        "-m",
        "twine",
        "upload",
        "--non-interactive",
        "--repository-url",
        repository_url,
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if verbose:
        cmd.append("--verbose")
    cmd.append(str(dist_dir / "*"))
    _run(cmd, env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="Short identifier for logs/paths")
    ap.add_argument("--repo-url", required=True, help="HF git repo URL, e.g. https://huggingface.co/nvidia/xxx")
    ap.add_argument("--work-dir", default=".work", help="Workspace temp dir")
    ap.add_argument("--dist-dir", default="dist-out", help="Where to collect artifacts")
    ap.add_argument(
        "--project-subdir",
        default="",
        help="If the Python project lives in a subdirectory, build from there (e.g. 'nemotron-ocr')",
    )
    ap.add_argument(
        "--build-env",
        action="append",
        default=[],
        help="Extra env vars for the build backend, as KEY=VALUE (repeatable)",
    )
    ap.add_argument(
        "--build-no-isolation",
        action="store_true",
        help="Pass --no-isolation to `python -m build` (useful to reuse preinstalled deps in CI images)",
    )
    ap.add_argument(
        "--venv-dir",
        default=".venv-build",
        help="Local venv used for build/twine (default: .venv-build)",
    )
    ap.add_argument(
        "--venv-system-site-packages",
        action="store_true",
        help="Create the build venv with --system-site-packages (to reuse system-installed packages like torch)",
    )
    ap.add_argument(
        "--venv-pip-install",
        action="append",
        default=[],
        help="Extra packages to pip install into the build venv before building (repeatable)",
    )
    ap.add_argument("--upload", action="store_true", help="Upload built dists via twine")
    ap.add_argument("--repository-url", default="https://test.pypi.org/legacy/", help="Twine repository URL")
    ap.add_argument("--token-env", default="TEST_PYPI_API_TOKEN", help="Env var containing API token")
    ap.add_argument("--skip-existing", action="store_true", help="Pass --skip-existing to twine")
    ap.add_argument(
        "--twine-verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --verbose to twine upload (default: true; use --no-twine-verbose to silence)",
    )
    ap.add_argument(
        "--hatch-force-platform-wheel",
        action="store_true",
        help="Patch hatch_build.py so hatchling emits a platform-specific wheel (not py3-none-any)",
    )
    ap.add_argument(
        "--auditwheel-repair",
        action="store_true",
        help="Run auditwheel repair on built wheels (manylinux tag; needed for PyPI/TestPyPI)",
    )
    ap.add_argument(
        "--auditwheel-exclude",
        action="append",
        default=[],
        help="Shared library to exclude from auditwheel bundling (repeatable). "
        "Use for runtime deps like libtorch_cpu.so that should not be vendored.",
    )
    args = ap.parse_args()

    root = Path.cwd()
    work_root = root / args.work_dir
    dist_root = root / args.dist_dir
    os.environ["ORCH_VENV_DIR"] = str(root / args.venv_dir)

    repo_dir = work_root / args.repo_id
    _ensure_clean_dir(work_root)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print(f"=== Cloning {args.repo_url} -> {repo_dir} ===")
    _clone_repo(args.repo_url, repo_dir)

    print("=== Attempting nightly version patch ===")
    if not args.project_subdir:
        detected = _auto_project_subdir(repo_dir, args.repo_id)
        if detected:
            args.project_subdir = detected
            print(f"Auto-detected project subdir: {args.project_subdir}")
    project_dir = repo_dir / args.project_subdir if args.project_subdir else repo_dir
    patched = _patch_pyproject_version(project_dir) or _patch_setup_cfg_version(project_dir)
    if not patched:
        print("No static version field found to patch (continuing).")

    if args.hatch_force_platform_wheel:
        if not _patch_hatch_build_force_platform_wheel(project_dir):
            print("hatch-force-platform-wheel: no applicable hatch_build.py patch applied")

    print("=== Building sdist + wheel ===")
    out_dir = dist_root / args.repo_id
    _ensure_clean_dir(out_dir)
    _build(
        project_dir,
        out_dir,
        build_env=args.build_env,
        no_isolation=args.build_no_isolation,
        venv_system_site_packages=args.venv_system_site_packages,
        venv_pip_install=args.venv_pip_install,
    )
    print(f"Artifacts in: {out_dir}")

    if args.auditwheel_repair:
        print("=== auditwheel repair ===")
        _auditwheel_repair_dist_dir(out_dir, exclude_libs=args.auditwheel_exclude)

    if args.upload:
        token = os.environ.get(args.token_env)
        if not token:
            raise RuntimeError(f"Missing required env var: {args.token_env}")
        print(f"=== Uploading to {args.repository_url} ===")
        _twine_upload(
            out_dir,
            args.repository_url,
            token,
            skip_existing=args.skip_existing,
            verbose=args.twine_verbose,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
