from data_generation.config import (
    BATCH_SIZE,
    MAX_NEW_TOKENS,
    OUTPUT_PATH,
)
from data_generation.jobs import build_jobs, run_jobs
from data_generation.storage import (
    create_sample,
    load_completed_samples,
    write_all_samples,
)


def main() -> None:
    completed = load_completed_samples(OUTPUT_PATH)

    jobs = [job for job in build_jobs() if job.id not in completed]

    print(f"Pending jobs: {len(jobs)}")

    final = run_jobs(
        jobs,
        max_retries=3,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    write_all_samples(
        OUTPUT_PATH,
        [create_sample(result) for result in final],
    )


if __name__ == "__main__":
    main()
