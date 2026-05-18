"""
File: AoC_scraper.py
Author: Jasper Kleine
Date: 02-03-2026

Description:
Scrape Advent of Code day pages and save:
Input, Output, Assignment text as .txt and .html and assignment metadata as json.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from aocd.models import Puzzle  # type: ignore
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv

env = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env)

BASE_URL = "https://adventofcode.com"
EVENTS_URL = "https://adventofcode.com/2025/events"
YEAR_PATH_RE = re.compile(r"^/\d{4}$")
DAY_PATH_RE = re.compile(r"^/(\d{4})/day/(\d+)$")
SESSION = os.getenv("AOC_SESSION")
SOLVER_URL = "https://advent.fly.dev/solve/"


def fetch_html(url: str) -> str:
    """Fetch HTML content from a URL.
    Args:
        url (str): URL to fetch.
    Raises:
        requests.HTTPError: If the response status is not OK.
    Returns:
        str: Response HTML as text.
    """
    response = requests.get(
        url,
        headers={
            "User-Agent": "AoC scraper (contact: local)",
        },
        cookies={"session": SESSION} if SESSION else None,
    )
    response.raise_for_status()
    return response.text


def _links_outside_layout(soup: BeautifulSoup) -> Iterable[str]:
    """Yield anchor hrefs after removing layout elements.
    Args:
        soup (BeautifulSoup): Parsed HTML document.
    Returns:
        Iterable[str]: Anchor href values outside layout containers.
    """
    for container in soup.select("header, nav, .sidebar"):
        container.decompose()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href")
        if isinstance(href, str):
            yield href


def extract_year_links(events_html: str) -> list[str]:
    """Extract year links from an events page HTML.
    Args:
        events_html (str): Events page HTML.
    Returns:
        list[str]: Unique year URLs sorted ascending.
    """
    soup = BeautifulSoup(events_html, "html.parser")
    links = []
    events_path = urlparse(EVENTS_URL).path
    events_year = events_path.strip("/").split("/")[0] if events_path else ""
    for href in _links_outside_layout(soup):
        if href == "/" and events_year.isdigit():
            links.append(BASE_URL + f"/{events_year}")
            continue
        if YEAR_PATH_RE.match(href):
            links.append(BASE_URL + href)

    return sorted(set(links))


def extract_day_links(year_html: str) -> list[str]:
    """Extract day links from a year page HTML.
    Args:
        year_html (str): Year page HTML.
    Returns:
        list[str]: Unique day URLs sorted by day number.
    """
    soup = BeautifulSoup(year_html, "html.parser")
    links = []
    for anchor in soup.select('a[aria-label^="Day"][href]'):
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        match = DAY_PATH_RE.match(href)
        if match:
            links.append(urljoin(BASE_URL, href))
    unique_links = set(links)
    return sorted(
        unique_links,
        key=_day_sort_key,
    )


def _day_sort_key(link: str) -> int:
    """Sort key for day URLs based on day number.
    Args:
        link (str): Day URL.
    Returns:
        int: Day number for sorting, or 0 if unparseable.
    """
    match = DAY_PATH_RE.match(urlparse(link).path or "")
    if not match:
        return 0
    return int(match.group(2))


def strip_code_in_main_paragraphs(soup: BeautifulSoup) -> Tag | None:
    """Strip code tags from top-level paragraphs inside <main>.
    Args:
        soup (BeautifulSoup): Parsed HTML document.
    Returns:
        Tag | None: The cleaned <main> tag, or None if missing.
    """
    main_tag = soup.find("main")
    if not main_tag:
        return None
    main_tag.extract()
    body = soup.body
    if body is not None:
        body.clear()
        body.append(main_tag)
    else:
        soup.clear()
        soup.append(main_tag)
    for paragraph in main_tag.find_all("p", recursive=False):
        for code_tag in paragraph.find_all("code"):
            code_tag.decompose()
    return main_tag


def extract_day_descriptions(soup: BeautifulSoup | Tag) -> list[Tag]:
    """Select day description blocks from a soup or tag.
    Args:
        soup (BeautifulSoup | Tag): Parsed document or parent tag.
    Returns:
        list[Tag]: Day description tags.
    """
    return list(soup.select(".day-desc"))


def render_text(node: Tag) -> str:
    """Render simplified text from an HTML node.
    Args:
        node (Tag): Day description tag to render.
    Returns:
        str: Plain text rendering.
    """
    lines: list[str] = []

    def add_block(text: str) -> None:
        """Append a non-empty text block.
        Args:
            text (str): Text to append.
        Returns:
            None: None.
        """
        if text:
            lines.append(text)

    def handle_list(tag: Tag, ordered: bool) -> None:
        """Render a list into text lines.
        Args:
            tag (Tag): List tag to render.
            ordered (bool): Whether the list is ordered.
        Returns:
            None: None.
        """
        for idx, item in enumerate(tag.find_all("li", recursive=False), start=1):
            prefix = f"{idx}. " if ordered else "- "
            text = item.get_text(" ", strip=True)
            add_block(prefix + text)

    def walk(tag: Tag) -> None:
        """Walk a tag tree and add text blocks.
        Args:
            tag (Tag): Tag to traverse.
        Returns:
            None: None.
        """
        for child in tag.children:
            if not isinstance(child, Tag):
                continue
            name = child.name
            if name in {"h1", "h2", "h3", "h4", "h5", "h6", "p"}:
                add_block(child.get_text(" ", strip=True))
            elif name == "pre":
                lines.append("")
                pre_text = child.get_text()
                lines.extend(pre_text.rstrip("\n").splitlines())
                lines.append("")
            elif name == "ul":
                handle_list(child, ordered=False)
            elif name == "ol":
                handle_list(child, ordered=True)
            else:
                walk(child)

    walk(node)
    return "\n".join(lines).strip() + "\n"


def save_day_files(
    output_root: str,
    year: str,
    day: str,
    full_html: str,
    part1_html: str | None,
    part2_html: str | None,
    part1_text: str | None,
    part2_text: str | None,
) -> None:
    """Persist scraped HTML and text for a day.
    Args:
        output_root (str): Output root directory.
        year (str): Puzzle year.
        day (str): Puzzle day.
        full_html (str): Full page HTML content.
        part1_html (str | None): Part 1 HTML, if available.
        part2_html (str | None): Part 2 HTML, if available.
        part1_text (str | None): Part 1 text, if available.
        part2_text (str | None): Part 2 text, if available.
    Returns:
        None: None.
    """
    directory = Path(output_root) / year / day
    directory.mkdir(parents=True, exist_ok=True)

    (directory / "full.html").write_text(full_html, encoding="utf-8")
    if part1_html is not None:
        (directory / "part1.html").write_text(part1_html, encoding="utf-8")
    if part2_html is not None:
        (directory / "part2.html").write_text(part2_html, encoding="utf-8")
    if part1_text is not None:
        (directory / "part1.txt").write_text(part1_text, encoding="utf-8")
    if part2_text is not None:
        (directory / "part2.txt").write_text(part2_text, encoding="utf-8")


def save_solution_text(
    output_root: str,
    year: int,
    day: int,
    name: str,
    data: str,
) -> None:
    """Persist solver output for a specific day.
    Args:
        output_root (str): Output root directory.
        year (int): Puzzle year.
        day (int): Puzzle day.
        name (str): Output filename stem.
        data (str): Text to write.
    Returns:
        None: None.
    """
    output_dir = Path(output_root) / str(year) / f"{day}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.txt").write_text(data, encoding="utf-8")


def store_meta_json(
    puzzle: "Puzzle",
    output_root: str,
    input_data: str | None = None,
    year: int = 2015,
    day: int = 1,
) -> None:
    """Store puzzle metadata and answers to JSON.
    Args:
        puzzle (Puzzle): Puzzle instance.
        output_root (str): Output root directory.
        input_data (str | None): Puzzle input data, if available.
        year (int): Puzzle year.
        day (int): Puzzle day.
    Returns:
        None: None.
    """
    meta = {
        "date": f"{year}, puzzle {day:02d}",
        "title": puzzle.title,
        "url": puzzle.url,
        "input": input_data,
        "part1": {
            "examples": [
                {
                    "input": example.input_data,
                    "answer": example.answer_a,
                }
                for example in puzzle.examples
                if example.answer_a is not None
            ],
            "answer": puzzle.answer_a,
        },
        "part2": {
            "examples": [
                {
                    "input": example.input_data,
                    "answer": example.answer_b,
                }
                for example in puzzle.examples
                if example.answer_b is not None
            ],
            "answer": puzzle.answer_b,
        },
    }
    output_dir = Path(output_root) / str(year) / f"{day}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def fetch_solution(input_data: str, year: int, day: int, part: int) -> str:
    """Fetch a solution from the solver service.
    Args:
        input_data (str): Puzzle input data.
        year (int): Puzzle year.
        day (int): Puzzle day.
        part (int): Puzzle part number.
    Returns:
        str: Solver response text, or "0" on error.
    """
    try:
        full_url = f"{SOLVER_URL}{year}/{day}/{part}"
        headers = {"accept": "text/plain", "content-type": "text/plain"}
        response = requests.post(full_url, headers=headers, data=input_data)
        response.raise_for_status()
        return response.text
    except Exception:
        return "0"


def solve_and_submit(
    output_root: str,
    puzzle: "Puzzle",
    input_data: str,
    year_value: int,
    day_value: int,
    stdout_sink: io.StringIO,
    delay: float,
    has_part2: bool,
) -> bool:
    """Solve parts, submit answers, and persist outputs.
    Args:
        output_root (str): Output root directory.
        puzzle (Puzzle): Puzzle instance.
        input_data (str): Puzzle input data.
        year_value (int): Puzzle year.
        day_value (int): Puzzle day.
        stdout_sink (io.StringIO): Sink for captured stdout.
        delay (float): Seconds between submissions.
        has_part2 (bool): Whether part 2 is available.
    Returns:
        bool: True if a rescrape is needed, else False.
    """
    solution1 = fetch_solution(input_data, year_value, day_value, part=1)
    with contextlib.redirect_stdout(stdout_sink):
        puzzle.answer_a = solution1
    save_solution_text(output_root, year_value, day_value, "output1", solution1)

    last_day = 12 if year_value >= 2025 else 25
    if day_value == last_day:
        store_meta_json(puzzle, output_root, input_data, year_value, day_value)
        return False

    if not has_part2:
        return True

    time.sleep(delay)
    solution2 = fetch_solution(input_data, year_value, day_value, part=2)
    with contextlib.redirect_stdout(stdout_sink):
        puzzle.answer_b = solution2
    save_solution_text(output_root, year_value, day_value, "output2", solution2)
    store_meta_json(puzzle, output_root, input_data, year_value, day_value)
    return False


def scrape(
    events_url: str,
    output_root: str,
    delay: float,
    year: str | None = None,
    day: str | None = None,
) -> dict[str, list[str]]:
    """Scrape AoC pages and write outputs to disk.
    Args:
        events_url (str): Events page URL.
        output_root (str): Output root directory.
        delay (float): Seconds between requests.
        year (str | None): Optional year filter.
        day (str | None): Optional day filter.
    Raises:
        EnvironmentError: Missing AOC_SESSION env var.
    Returns:
        dict[str, list[str]]: Mapping of year URL to day URLs.
    """
    if not SESSION:
        raise EnvironmentError("AOC_SESSION is missing. Add it to .env.")

    stdout_sink = io.StringIO()
    if year:
        year_links = [urljoin(BASE_URL, f"/{year}")]
    else:
        events_html = fetch_html(events_url)
        year_links = extract_year_links(events_html)
    year_to_days: dict[str, list[str]] = {}

    for year_link in year_links:
        if day:
            day_links = [urljoin(year_link.rstrip("/") + "/", f"day/{day}")]
        else:
            year_html = fetch_html(year_link)
            day_links = extract_day_links(year_html)
        year_to_days[year_link] = day_links

        for day_link in day_links:
            print(f"Scraping: {day_link}")
            day_html = fetch_html(day_link)
            soup = BeautifulSoup(day_html, "html.parser")
            full = strip_code_in_main_paragraphs(soup)
            day_descriptions = extract_day_descriptions(full if full else soup)
            part1_html = str(day_descriptions[0]) if len(day_descriptions) > 0 else None
            part2_html = str(day_descriptions[1]) if len(day_descriptions) > 1 else None
            part1_text = (
                render_text(day_descriptions[0]) if len(day_descriptions) > 0 else None
            )
            part2_text = (
                render_text(day_descriptions[1]) if len(day_descriptions) > 1 else None
            )

            match = DAY_PATH_RE.match(urlparse(day_link).path or "")
            if not match:
                print(f"Warning: {day_link}.")
                continue
            year_value, day_value = match.groups()
            save_day_files(
                output_root,
                year_value,
                day_value,
                str(full) if full else str(soup),
                part1_html,
                part2_html,
                part1_text,
                part2_text,
            )

            time.sleep(delay)
            puzzle = Puzzle(year=int(year_value), day=int(day_value))
            input_data = puzzle.input_data
            save_solution_text(
                output_root, int(year_value), int(day_value), "input", input_data
            )
            time.sleep(delay)
            needs_rescrape = solve_and_submit(
                output_root,
                puzzle,
                input_data,
                int(year_value),
                int(day_value),
                stdout_sink,
                delay,
                has_part2=part2_html is not None,
            )
            if needs_rescrape:
                time.sleep(delay)
                day_html = fetch_html(day_link)
                soup = BeautifulSoup(day_html, "html.parser")
                full = strip_code_in_main_paragraphs(soup)
                day_descriptions = extract_day_descriptions(full if full else soup)
                part1_html = (
                    str(day_descriptions[0]) if len(day_descriptions) > 0 else None
                )
                part2_html = (
                    str(day_descriptions[1]) if len(day_descriptions) > 1 else None
                )
                part1_text = (
                    render_text(day_descriptions[0])
                    if len(day_descriptions) > 0
                    else None
                )
                part2_text = (
                    render_text(day_descriptions[1])
                    if len(day_descriptions) > 1
                    else None
                )
                save_day_files(
                    output_root,
                    year_value,
                    day_value,
                    str(full) if full else str(soup),
                    part1_html,
                    part2_html,
                    part1_text,
                    part2_text,
                )
                if part2_html is not None:
                    time.sleep(delay)
                    _ = solve_and_submit(
                        output_root,
                        puzzle,
                        input_data,
                        int(year_value),
                        int(day_value),
                        stdout_sink,
                        delay,
                        has_part2=True,
                    )

            if delay:
                time.sleep(delay)

        if delay:
            time.sleep(delay)

    return year_to_days


def main() -> int:
    """CLI entry point.
    Args:
        None: None.
    Raises:
        SystemExit: Raised by argparse on errors.
    Returns:
        int: Exit code.
    """
    parser = argparse.ArgumentParser(description="Scrape AoC day pages.")
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument(
        "--delay", type=float, default=5, help="Seconds between requests."
    )
    parser.add_argument("--year", "-y", help="Scrape a specific year (YYYY).")
    parser.add_argument("--day", "-d", help="Scrape a single day within the year.")
    args = parser.parse_args()

    if args.day and not args.year:
        parser.error("--day requires --year")

    scrape(
        EVENTS_URL,
        args.output_root,
        args.delay,
        year=args.year,
        day=args.day,
    )

    return 0


if __name__ == "__main__":
    main()
