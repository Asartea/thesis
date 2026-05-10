from pathlib import Path
import json

from generation.config import MODEL
from generation.models import Job, LLMSample


def load_completed_samples(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed_ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line)
                job_id = f"{sample['model']}-{sample['year']}-{sample['day']}-{sample['code_variant']}-{sample['style_variant']}"
                completed_ids.add(job_id)
            except (json.JSONDecodeError, KeyError):
                continue

    return completed_ids


def create_sample(result: tuple[Job, str]) -> LLMSample:
    job, code = result
    return LLMSample(
        model=MODEL,
        prompt=job.prompt,
        code_variant=job.code_variant,
        style_variant=job.style_variant,
        code=code,
        label="machine",
        year=str(job.year),
        day=str(job.day),
        language="python",
    )


def write_all_samples(path: Path, samples: list[LLMSample]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for sample in samples:
            json.dump(sample, f)
            f.write("\n")


def read_file(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None
