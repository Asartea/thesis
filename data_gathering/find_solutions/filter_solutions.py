from pathlib import Path
import re

YEAR_PATTERN = re.compile(
    r"(20\d{2}|aoc[\s_-]?\d{2,4}|adventofcode[\s_-]?\d{2,4})", re.IGNORECASE
)


def detect_year(path: Path) -> str | None:
    """
    Lightweight year detector used ONLY for filtering.

    Accepts:
    - AdventOfCode2021 / aoc-2021 / 2021
    - AdventOfCode24 / aoc24 / 2024
    """

    def extract_year_from_match(m: re.Match[str]) -> str:
        val = m.group(0).lower()
        digits = re.search(r"\d{2,4}", val).group(0)
        if len(digits) == 2:
            return "20" + digits
        return digits

    for part in path.parts:
        m = YEAR_PATTERN.search(part)
        if not m:
            continue
        return extract_year_from_match(m)


def is_valid_solution(path: Path, target_years: list[str]) -> bool:
    """
    Returns True if the file belongs to one of the allowed AoC years.
    """

    if path.suffix != ".py":
        return False

    year = detect_year(path)

    return year in set(target_years) if year else False


def filter_solutions(solution_files: list[Path], target_years: list[str]) -> list[Path]:
    """
    Filter solution files to only include those from target years.
    """
    return [f for f in solution_files if is_valid_solution(f, target_years)]
