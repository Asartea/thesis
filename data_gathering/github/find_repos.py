import re
from time import sleep

import dotenv
import requests

from models.models import GitHubRepo


def filter_repo(repo: GitHubRepo, target_years: list[str], threshold: float) -> bool:
    """
    Filter repositories based on relevance to Advent of Code.

    Scoring criteria:
    - Python language required
    - Topics "advent-of-code" or "aoc": +3.0
    - Keywords in name/description: +1.5
    - Target year mentioned: +2.0 per year
    - Other years mentioned: -0.5 per year

    Args:
        repo: Repository dictionary from GitHub API
        target_years: Years to prioritize (e.g., ["2021", "2024"])
        threshold: Minimum score required to pass filter

    Returns:
        True if repo score >= threshold, False otherwise
    """
    # language check
    language = repo.get("language")
    if not language or language.lower() != "python":
        return False
    keywords = ["advent of code", "aoc", "adventofcode"]

    topics = [t.lower() for t in repo.get("topics", [])]
    description = (repo.get("description") or "").lower()
    name = (repo.get("name") or "").lower()
    text = f"{name} {description}"

    score = 0.0

    if "advent-of-code" in topics or "aoc" in topics:
        score += 3.0

    if any(k in text for k in keywords):
        score += 1.5

    YEAR_PATTERNS = {str(y): re.compile(rf"\b{y}\b") for y in range(2015, 2025)}

    target_hits = sum(1 for y in target_years if YEAR_PATTERNS[y].search(text))

    other_hits = sum(
        1
        for y in range(2015, 2025)
        if str(y) not in target_years and YEAR_PATTERNS[str(y)].search(text)
    )

    score += 2.0 * target_hits

    if other_hits == 1:
        score -= 0.3
    elif other_hits > 1:
        score -= 1.0

    return score >= threshold


def get_possible_repos_for_user(
    user: str, target_years: list[str], threshold: float, timeout: float = 5
) -> list[tuple[str, str]]:
    """
    Fetch and filter repositories for a GitHub user.

    Args:
        user: GitHub username
        target_years: List of years to match (e.g., ["2021", "2024"])
        threshold: Minimum score for a repo to be included
        timeout: Sleep timeout between requests (seconds)

    Returns:
        List of tuples (repo_full_name, clone_url) for matching repos
    """
    url = f"https://api.github.com/users/{user}/repos"
    headers: dict[str, str] = {}
    token = dotenv.get_key(".env", "GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    repos: list[GitHubRepo] = []
    page = 1
    while True:
        params = {"per_page": 100, "page": page}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code != 200:
                print(
                    f"Failed to fetch repos for user {user}. Status code: {response.status_code}"
                )
                return []

            page_repos = response.json()
            if not page_repos:
                break
            repos.extend(page_repos)
            page += 1
            sleep(timeout)
        except requests.RequestException as e:
            print(f"Error fetching repos for user {user}: {e}")
            return []

    return [
        (repo["full_name"], repo["clone_url"])
        for repo in repos
        if filter_repo(repo, target_years, threshold)
    ]
