from __future__ import annotations

import glob
from collections.abc import Iterable
from os import PathLike, fspath
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlparse

INPUT_TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "pdf": ("*.pdf",),
    "txt": ("*.txt",),
    "html": ("*.html",),
    "doc": ("*.docx", "*.pptx"),
    "image": ("*.jpg", "*.jpeg", "*.png", "*.tiff", "*.bmp"),
    "audio": ("*.mp3", "*.wav", "*.m4a"),
    "video": ("*.mp4", "*.mov", "*.mkv"),
}

InputPath = str | PathLike[str]


def _has_uri_scheme(path: str) -> bool:
    return bool(urlparse(path).scheme)


def _is_explicit_glob_path(input_path: InputPath) -> bool:
    return glob.has_magic(fspath(input_path))


def _raise_input_path_not_found(input_path: object, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    raise FileNotFoundError(f"Input path does not exist: {input_path}") from cause


def expand_input_file_patterns(input_paths: InputPath | Iterable[InputPath]) -> list[str]:
    """Expand local path/glob inputs and reject missing local literal paths.

    Empty explicit glob matches are allowed so callers can intentionally
    describe optional file sets. URI inputs are forwarded to lower-level
    readers, which can resolve remote filesystems.
    """
    paths = [input_paths] if isinstance(input_paths, (str, PathLike)) else list(input_paths)

    expanded: list[str] = []
    for input_path in paths:
        raw_path = fspath(input_path)
        if _has_uri_scheme(raw_path):
            expanded.append(raw_path)
            continue

        local_path = Path(raw_path).expanduser()
        if local_path.is_dir():
            raise IsADirectoryError(f"Input path is a directory, not a file or glob pattern: {local_path}")

        pattern = str(local_path)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            expanded.extend(sorted(matches))
        elif _is_explicit_glob_path(pattern):
            expanded.append(pattern)
        elif not Path(pattern).exists():
            _raise_input_path_not_found(pattern)
        else:
            expanded.append(pattern)

    return expanded


def normalize_read_file_not_found(input_paths: list[str], cause: FileNotFoundError) -> NoReturn:
    """Normalize a lower-level file reader error into the input path contract.

    Parameters
    ----------
    input_paths
        Expanded paths or patterns attempted by the file reader.
    cause
        The lower-level ``FileNotFoundError`` raised by the reader.

    Raises
    ------
    FileNotFoundError
        Always raised with a product-level message and ``cause`` chained.
    """
    if len(input_paths) == 1:
        _raise_input_path_not_found(input_paths[0], cause)

    paths = ", ".join(input_paths) if input_paths else "<none>"
    raise FileNotFoundError(
        f"One or more input paths do not exist. Attempted input paths: {paths}. Reader error: {cause}"
    ) from cause


def resolve_input_patterns(input_path: Path, input_type: str) -> list[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if not path.is_dir():
        raise FileNotFoundError(f"Path does not exist: {path}")

    patterns = INPUT_TYPE_PATTERNS.get(input_type, INPUT_TYPE_PATTERNS["pdf"])
    return [str(path / "**" / pattern) for pattern in patterns]


def resolve_input_files(input_path: Path, input_type: str) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return [path]
    if not path.exists():
        return []

    files: list[Path] = []
    for pattern in INPUT_TYPE_PATTERNS.get(input_type, INPUT_TYPE_PATTERNS["pdf"]):
        files.extend(match for match in path.rglob(pattern) if match.is_file())
    return sorted(set(files))
