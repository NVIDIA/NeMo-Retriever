from __future__ import annotations

import glob
from collections.abc import Iterable
from os import PathLike, fspath
from pathlib import Path
from typing import NoReturn

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


def _is_explicit_glob_path(input_path: InputPath) -> bool:
    return glob.has_magic(fspath(input_path))


def raise_input_path_not_found(input_path: object, cause: BaseException | None = None) -> NoReturn:
    """Raise a consistent missing-input-path error.

    Parameters
    ----------
    input_path
        Path, pattern, or list of paths attempted by the caller or file reader.
    cause
        Optional lower-level exception to preserve as the chained cause.

    Raises
    ------
    FileNotFoundError
        Always raised with a product-level missing-input-path message.
    """
    message = f"Input path does not exist: {input_path}"

    if cause is None:
        raise FileNotFoundError(message)
    raise FileNotFoundError(f"{message}. Reader error: {cause}") from cause


def expand_input_file_patterns(input_paths: InputPath | Iterable[InputPath]) -> list[str]:
    """Expand local path/glob inputs and reject missing or directory local literal paths.

    Empty explicit glob matches are allowed so callers can intentionally
    describe optional file sets.
    """
    paths = [input_paths] if isinstance(input_paths, (str, PathLike)) else list(input_paths)

    expanded: list[str] = []
    for input_path in paths:
        raw_path = fspath(input_path)
        pattern = str(Path(raw_path).expanduser())
        matches = [match for match in glob.glob(pattern, recursive=True) if Path(match).is_file()]
        if matches:
            expanded.extend(sorted(matches))
        elif _is_explicit_glob_path(pattern):
            expanded.append(pattern)
        elif not Path(pattern).exists():
            raise_input_path_not_found(pattern)
        elif Path(pattern).is_dir():
            raise IsADirectoryError(
                f"Input path is a directory: {pattern}. "
                "Pass a file path or a glob pattern such as '<dir>/**/*.pdf' or '<dir>/**/*' "
                "to select files inside the directory."
            )
        else:
            expanded.append(pattern)

    return expanded


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
