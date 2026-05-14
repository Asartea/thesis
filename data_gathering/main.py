import os
from pathlib import Path

from data_gathering.find_solutions.filter_solutions import filter_solutions
from data_gathering.find_solutions.find_solutions_files import find_solution_files
from data_gathering.find_solutions.text_extractor import HumanSample, build_samples
from data_gathering.github.find_repos import get_possible_repos_for_user
from data_gathering.github.find_solutions import clone_repo
from data_gathering.leaderboard.parser import extract_top_users
from data_gathering.leaderboard.scraper import scrape_data

from validation.validation import CodeValidationError, validate_code
from utils import write_to_jsonl


def write_tsv(top_users: dict[str, str], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("user_id\tgithub_url\n")
        for user_id, github_url in top_users.items():
            f.write(f"{user_id}\t{github_url}\n")


def scrape_and_process_data(
    target_years: list[str], days: list[str], base_url: str, base_dir: Path
) -> dict[str, str]:
    """Scrape, parse, and process Advent of Code leaderboard data."""
    for year in target_years:
        if not os.path.exists(base_dir / year / "html" / "global_leaderboard.html"):
            scrape_data(year, days, base_url, base_dir)
    top_users = extract_top_users(target_years, days, base_dir)
    return top_users


def find_github_repos_and_clone(
    top_users: dict[str, str], base_dir: Path, target_years: list[str]
) -> None:
    """Find GitHub repos for users in the leaderboard and clone them."""

    for _, github_url in top_users.items():
        github_user_name = github_url.split("/")[-1]
        github_repos = get_possible_repos_for_user(
            github_user_name, target_years, threshold=3.0, timeout=0.1
        )
        for repo_name, repo_url in github_repos:
            target_path = base_dir / "repos" / repo_name
            clone_repo(repo_url, target_path)


def find_solution_paths_in_repos_and_extract_solutions(base_dir: Path) -> list[Path]:
    """Find solution files in the cloned GitHub repositories."""
    repo_dir = base_dir / "repos"

    all_solution_files: list[Path] = []

    for user_path in repo_dir.iterdir():
        if not user_path.is_dir():
            continue
        for repo_path in user_path.iterdir():
            if not repo_path.is_dir():
                continue

            repo_solutions = find_solution_files(repo_path)
            filtered_solutions = filter_solutions(
                repo_solutions, target_years=["2021", "2024"]
            )
            if filtered_solutions:
                print(
                    f"Found {len(filtered_solutions)} solution files in {user_path.name}/{repo_path.name}."
                )
                all_solution_files.extend(filtered_solutions)

    return all_solution_files


def validate_samples(samples: list[HumanSample]) -> list[HumanSample]:
    """Validate the extracted code samples."""

    valid_samples: list[HumanSample] = []
    for sample in samples:
        code = sample["code"]

        try:
            validate_code(code)
            valid_samples.append(sample)
        except CodeValidationError as e:
            print(f"Validation error in sample from {sample['path']}: {e}")
    return valid_samples


def validate_and_save_samples(samples: list[HumanSample], output_file: Path) -> None:
    """Validate the extracted code samples and save valid ones to a JSONL file."""
    valid_samples = validate_samples(samples)
    write_to_jsonl(valid_samples, output_file)


def execute(
    target_years: list[str], days: list[str], base_url: str, base_dir: Path
) -> None:
    """Execute the full data gathering pipeline."""

    top_users = scrape_and_process_data(target_years, days, base_url, base_dir)
    write_tsv(top_users, base_dir / "top_users.tsv")
    find_github_repos_and_clone(top_users, base_dir, target_years)
    all_solution_files = find_solution_paths_in_repos_and_extract_solutions(base_dir)
    samples = build_samples(all_solution_files)
    validate_and_save_samples(samples, Path(base_dir) / "extracted_solutions.jsonl")


def main() -> None:
    """Execute the full data gathering pipeline."""
    target_years = ["2021", "2024"]
    days = [str(i) for i in range(1, 26)]
    base_url = "https://adventofcode.com"
    base_dir = Path("data_gathering/") / "data"
    all_solution_files = find_solution_paths_in_repos_and_extract_solutions(base_dir)
    samples = build_samples(all_solution_files)
    validate_and_save_samples(samples, Path(base_dir) / "extracted_solutions.jsonl")


if __name__ == "__main__":
    main()
