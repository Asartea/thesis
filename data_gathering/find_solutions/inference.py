import re
from pathlib import Path

DAY_PATTERN = re.compile(r"\b(?:day|d)[\s_-]?(\d{1,2})\b", re.IGNORECASE)
BARE_DAY_PATTERN = re.compile(r"^(\d{1,2})(?:\D|$)")
DAY_WITH_SUFFIX = re.compile(r"^day[_-]?(\d{1,2})(?:[_-][ab])?$", re.IGNORECASE)

YEAR_PATTERN = re.compile(
    r"(?:20\d{2}|aoc[\s_-]?\d{2}|adventofcode\d{2,4})", re.IGNORECASE
)
TWO_DIGIT_YEAR = re.compile(r"\b(\d{2})\b")


def infer_day(path: Path | str) -> str | None:
    """
    Infer AoC day from path (searches most specific segments first).
    """
    p = Path(path)

    for part in reversed(p.parts):
        m = DAY_PATTERN.search(part)
        if m:
            return m.group(1)

    stem = p.stem
    m = BARE_DAY_PATTERN.match(stem)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 25:
            return f"{day:02d}"

    m = DAY_WITH_SUFFIX.match(stem)
    if m:
        day = int(m.group(1))
        if 1 <= day <= 25:
            return f"{day:02d}"

    parent = p.parent
    if BARE_DAY_PATTERN.match(parent.stem):
        day = int(BARE_DAY_PATTERN.match(parent.stem).group(1))
        if 1 <= day <= 25:
            return f"{day:02d}"

    return None


def infer_year(path: Path | str) -> str | None:
    p = Path(path)

    for part in p.parts:
        m = YEAR_PATTERN.search(part)
        if m:
            val = m.group(0).lower()

            # Case: adventofcode24 or adventofcode2024
            if val.startswith("adventofcode"):
                digits = re.search(r"\d{2,4}", val).group(0)
                if len(digits) == 2:
                    return "20" + digits
                return digits

            # Case: aoc24 → 2024
            if "aoc" in val:
                digits = re.search(r"\d{2,4}", val).group(0)
                if len(digits) == 2:
                    return "20" + digits
                return digits

            # Case: full year 2024
            if val.isdigit():
                return val

    return None
