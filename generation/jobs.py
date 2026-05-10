import random
from pathlib import Path

from generation.batch import run_batch, validate_batch
from generation.config import (
    CODE_VARIANTS,
    DAYS,
    INPUT_DIR,
    STYLE_VARIANTS,
    TEST_MODE,
    TEST_SAMPLE_SIZE,
    YEARS,
)
from generation.prompts import build_prompt
from generation.storage import read_file
from generation.models import Job


def retry_failed_jobs(failed: list[Job], batch_size: int) -> list[tuple[Job, str]]:
    if not failed:
        return []

    return run_batch(failed, batch_size, seed=random.randint(0, 2**31 - 1))


def build_jobs() -> list[Job]:
    jobs: list[Job] = []

    for year in YEARS:
        for day in DAYS:
            part1_file = Path(INPUT_DIR) / str(year) / str(day) / "part1.txt"
            part2_file = Path(INPUT_DIR) / str(year) / str(day) / "part2.txt"
            print(
                f"Building job for {year} day {day} with inputs {part1_file} and {part2_file}"
            )
            part1 = read_file(part1_file)
            part2 = read_file(part2_file)
            problem = "\n\n".join(filter(None, [part1, part2]))
            if not problem:
                print(
                    f"Warning: No problem statement found for {year} day {day}, skipping."
                )
                continue

            for code_variant in CODE_VARIANTS:
                for style_variant in STYLE_VARIANTS:
                    prompt = build_prompt(problem, code_variant, style_variant)

                    jobs.append(
                        Job(
                            year=year,
                            day=day,
                            code_variant=code_variant,
                            style_variant=style_variant,
                            prompt=prompt,
                        )
                    )
    if TEST_MODE:
        return random.sample(jobs, min(TEST_SAMPLE_SIZE, len(jobs)))
    return jobs


def run_jobs(jobs: list[Job], max_retries: int = 3, batch_size: int = 8):
    results = run_batch(jobs, batch_size)

    valid, failed = validate_batch(results)
    all_valid = valid

    for i in range(max_retries):
        if not failed:
            break

        print(f"Retry {i + 1}: {len(failed)} failed")

        retry_results = retry_failed_jobs(failed, batch_size)
        valid_retry, failed_retry = validate_batch(retry_results)

        failed = failed_retry

        all_valid.extend(valid_retry)

    return all_valid
