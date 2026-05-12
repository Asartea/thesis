import re
from pathlib import Path
from typing import Iterator

DAY_PATTERN = re.compile(
    r"^(day[\s_-]?\d{1,2}|d\d{1,2}|\d{1,2})(?:[\s_-].*)?$",
    re.IGNORECASE,
)


def is_day_folder(path: Path) -> bool:
    """Check if the folder name matches the pattern for a day folder."""
    return DAY_PATTERN.fullmatch(path.parent.name) is not None


def is_day_file(path: Path) -> bool:
    """Check if the filename matches the pattern for a day solution file."""
    return path.suffix == ".py" and bool(DAY_PATTERN.fullmatch(path.stem))


def iter_python_files(repo_path: Path, max_depth: int = 4) -> Iterator[Path]:

    def _iter_files(current_path: Path, current_depth: int = 0) -> Iterator[Path]:
        if current_depth >= max_depth:
            return
        try:
            for item_path in current_path.iterdir():
                if item_path.is_file() and item_path.suffix == ".py":
                    yield item_path
                elif item_path.is_dir():
                    yield from _iter_files(item_path, current_depth + 1)
        except (PermissionError, OSError):
            pass

    yield from _iter_files(repo_path)


def find_solution_files(repo_path: Path) -> list[Path]:
    results: list[Path] = []
    for file_path in iter_python_files(repo_path):
        if is_day_file(file_path):
            results.append(file_path)
            continue

        if is_day_folder(file_path):
            results.append(file_path)

    return results
