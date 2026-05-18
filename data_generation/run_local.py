from argparse import ArgumentParser, Namespace

from pathlib import Path

from data_generation.files import (
    create_sample,
    load_completed_samples,
    write_all_samples,
)
from data_generation.jobs import build_jobs
from data_generation.local.generator import (
    GenerationModel,
    GenerationTokenizer,
)
from data_generation.local.jobs import run_jobs
from data_generation.prompts import (
    CompProgrammingConfig,
    NormalConfig,
)


def run_local_model(
    years: list[int],
    days: list[int],
    model: str,
    *,
    comp_programming: bool = False,
    test_mode: bool = False,
    test_sample_size: int = 10,
    max_new_tokens: int = 512,
    batch_size: int = 4,
):
    output_path = Path("data_generation") / "data" / model / "samples.jsonl"
    data_dir = Path("data_generation") / "data" / "aoc-problems"
    completed = load_completed_samples(output_path, model)

    prompt_config = CompProgrammingConfig() if comp_programming else NormalConfig()

    print(data_dir)

    jobs = build_jobs(
        years,
        days,
        data_dir,
        model,
        completed,
        comp_programming=comp_programming,
        test_mode=test_mode,
        test_sample_size=test_sample_size,
    )
    print(f"Pending jobs: {len(jobs)}")

    generation_tokenizer = GenerationTokenizer(model)
    system_prompt = prompt_config.system_prompt
    generation_model = GenerationModel(
        model,
        generation_tokenizer,
        system_prompt,
    )

    final = run_jobs(
        jobs,
        model=generation_model,
        tokenizer=generation_tokenizer,
        system_prompt=system_prompt,
        max_retries=3,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    write_all_samples(
        output_path,
        [create_sample(result) for result in final],
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run local generation pipeline")

    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--days", type=int, nargs="+", required=True)

    parser.add_argument("--comp-programming", action="store_true")
    parser.add_argument("--test-mode", action="store_true")

    parser.add_argument("--test-sample-size", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_local_model(
        years=args.years,
        days=args.days,
        model=args.model,
        comp_programming=args.comp_programming,
        test_mode=args.test_mode,
        test_sample_size=args.test_sample_size,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
