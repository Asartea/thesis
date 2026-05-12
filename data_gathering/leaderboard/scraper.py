"""
Module for scraping the Advent of Code leaderboards.
"""

import time
from pathlib import Path

import requests


def get_year_leaderboard(year: str, base_url: str) -> str:
    """Fetch the global leaderboard for a given year.

    Args:
        year: The year of the leaderboard to fetch.
        base_url: The base URL of the Advent of Code website.

    Returns:
        The HTML content of the global leaderboard page.

    Raises:
        requests.RequestException: If the request fails or returns a non-200 status code.
    """
    url = f"{base_url}/{year}/leaderboard"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def get_day_leaderboard(year: str, day: str, base_url: str) -> str:
    """Fetch the leaderboard for a specific day of a given year.

    Args:
        year: The year of the leaderboard to fetch.
        day: The day of the leaderboard to fetch.
        base_url: The base URL of the Advent of Code website.

    Returns:
        The HTML content of the day leaderboard page.

    Raises:
        requests.RequestException: If the request fails or returns a non-200 status code.
    """
    url = f"{base_url}/{year}/leaderboard/day/{day}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def scrape_data(
    year: str, days: list[str], base_url: str, base_dir: str, interval: int = 5
) -> None:
    """Scrape both global and day-specific leaderboards for a given year.

    Saves the HTML content in base_dir/{year}/html/[global|day_{day}_leaderboard].html

    Args:
        year: The year to scrape.
        days: List of days to scrape.
        base_url: The base URL of the Advent of Code website.
        base_dir: The directory where the scraped HTML files will be saved.
        interval: Time in seconds to wait between requests to avoid rate limiting.
    """
    # Create directory structure if it doesn't exist
    html_dir = Path(base_dir) / year / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    # Fetch and save global leaderboard
    try:
        leaderboard_html = get_year_leaderboard(year, base_url)
        with open(html_dir / "global_leaderboard.html", "w", encoding="utf-8") as f:
            f.write(leaderboard_html)
        print(f"Saved global leaderboard for year {year}")
    except requests.RequestException as e:
        print(f"Error fetching global leaderboard for year {year}: {e}")

    time.sleep(interval)

    # Fetch and save daily leaderboards
    for d in days:
        try:
            day_leaderboard_html = get_day_leaderboard(year, d, base_url)
            with open(
                html_dir / f"day_{d}_leaderboard.html",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(day_leaderboard_html)
            print(f"Saved leaderboard for year {year} day {d}")
        except requests.RequestException as e:
            print(f"Error fetching leaderboard for year {year} day {d}: {e}")

        time.sleep(interval)
