import sys

from data_generation.config import SYSTEM_PROMPT
from data_generation.generator import generate_batch, tokenizer
from data_generation.models import Job
from data_generation.prompts import render_chat_prompt
from validation.validation import CodeValidationError, validate_code


def generate_prompt_lens(prompts: list[str]) -> list[int]:
    return [
        len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts
    ]


def run_batch(
    jobs: list[Job], batch_size: int, seed: int | None = None, max_new_tokens: int = 512
) -> list[tuple[Job, str]]:
    results: list[tuple[Job, str]] = []

    rendered_prompts = [
        render_chat_prompt(tokenizer, SYSTEM_PROMPT, job.prompt) for job in jobs
    ]
    prompt_lengths = generate_prompt_lens(rendered_prompts)

    sorted_jobs = sorted(
        zip(
            jobs,
            rendered_prompts,
            prompt_lengths,
            strict=True,
        ),
        key=lambda x: x[2],
    )

    for i in range(0, len(sorted_jobs), batch_size):
        batch = sorted_jobs[i : i + batch_size]
        prompts = [item[1] for item in batch]

        codes = generate_batch(prompts, max_new_tokens, seed)
        for item, code in zip(batch, codes, strict=True):
            job = item[0]
            results.append((job, code))

    return results


def validate_batch(results: list[tuple[Job, str]]):
    valid: list[tuple[Job, str]] = []
    failed: list[Job] = []

    for job, text in results:
        try:
            code = validate_code(text)
            valid.append((job, code))
        except CodeValidationError as e:
            print(f"Validation failed for {job.id}: {e}", file=sys.stderr)
            print(f"Code:\n{text}\n", file=sys.stderr, flush=True)
            failed.append(job)

    return valid, failed
