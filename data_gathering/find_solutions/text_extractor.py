from pathlib import Path
from typing import Optional

from find_solutions.inference import infer_year, infer_day
from models.models import HumanSample


def extract_text_from_solution_file(file_path: Path) -> str:
    """
    Read the raw contents of a solution file.

    Returns an empty string if the file cannot be read.
    """
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        print(f"[WARN] missing file {file_path}: {e}")
        return ""


def clean_code(text: str) -> str:
    """
    Normalize Python source code for downstream processing.

    Operations:
    - standardize line endings
    - remove trailing whitespace
    - strip leading/trailing blank space
    """
    text = text.replace("\r\n", "\n")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def build_sample(repo_path: Path, file_path: Path) -> Optional[HumanSample]:
    """
    Construct a structured dataset sample from a solution file.

    Pipeline:
    1. Read file contents
    2. Clean source code
    3. Infer AoC metadata (year/day)
    4. Package into a structured dictionary

    Returns:
        Sample dict if valid solution file, otherwise None.
    """
    code = extract_text_from_solution_file(file_path)

    if not code.strip():
        return None

    cleaned_code = clean_code(code)

    year = infer_year(file_path)
    day = infer_day(file_path)

    if year is None:
        print(f"[WARN] could not infer year from {file_path}")
    if day is None:
        print(f"[WARN] could not infer day from {file_path}")
    if year is None or day is None:
        return None

    return {
        "code": cleaned_code,
        "label": "human",
        "year": year,
        "day": day,
        "language": "python",
        "repo": repo_path.name,
        "path": str(file_path),
    }


def build_samples(repo_path: Path, solution_files: list[Path]) -> list[HumanSample]:
    """
    Convert a list of solution file paths into structured dataset samples.

    Filters out invalid or unparseable files.
    """
    samples: list[HumanSample] = []

    for file_path in solution_files:
        sample = build_sample(repo_path, file_path)
        if sample:
            samples.append(sample)

    return samples
