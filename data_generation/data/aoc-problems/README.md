# Advent of Code Dataset + Scraper

This folder contains the dataset produced by the AoC scraper and the script
used to build it.

## What is Advent of Code?
Advent of Code (AoC) is an annual programming event with 25 daily puzzles
released each December. Each day has two parts. Part 2 is unlocked after
submitting a correct answer for part 1. Puzzle inputs are personalized per
user session, so you must use your own session token to access your input and
submit answers.

## Dataset layout
Each day is stored in:

    <output_root>/<year>/<day>/

Files per day:

- full.html     Main HTML for the assignment (removed headers etc)
- part1.html    HTML for the first .day-desc block
- part2.html    HTML for the second .day-desc block
- part1.txt     Plain-text version of part 1
- part2.txt     Plain-text version of part 2
- input.txt     Your personalized input
- output1.txt   Answer for part 1 given input.txt
- output2.txt   Answer for part 2 given input.txt
- meta.json     Metadata (title, url, input, example answers, outputs)

## Scraper script
The scraper lives in the folder: `dataset`.
This is also the default root of the scraper.

It does two things for each requested day:

1) Scrapes the AoC day page and writes the HTML/text files.
2) Fetches your puzzle input and submits solutions via a solver API, then writes input/output/meta files. If part 2 is locked, it submits part 1, re-scrapes, and then continues once part 2 becomes visible.

## WARNING

Do <strong>NOT</strong> run the script using your main Advent of Code account session cookie.
The script submits answers to Advent of Code and will earn you stars.
That means it can mark puzzles as solved even if you have not solved them yourself.
Use a throwaway account if you want to avoid affecting your main progress.

### Usage
Basic example (single day):
```bash
python AoC_scraper.py --year 2025 --day 1
```

Scrape a whole year:
```bash
python AoC_scraper.py --year 2025
```

Scrape all available years from the events page:
```bash
python AoC_scraper.py
```

### Flags
- --year / -y   Scrape a specific year (YYYY)
- --day / -d    Scrape a single day within the year
- --events-url  Override the events page (default: `https://adventofcode.com/2025/events`)
- --output-root Output directory (default: the script folder)
- --delay       Seconds between requests, `aocd` will ratelimit you if you go to fast (default: 5)

## Environment variables
The script expects a .env file in this same folder. Required variables:

- AOC_SESSION
  The session cookie value from your AoC login. This is used both for scraping and for submitting answers.

## Dependencies
Install these in your environment (venv recommended):

- requests
- beautifulsoup4
- python-dotenv
- advent-of-code-data (aocd)

Example install:

```bash
pip install -r requirements.txt
```

or using uv:
```bash
uv sync
```

## Notes
- AoC rate limits aggressive scraping/submission. Keep delays conservative.
- Part 2 content is only available after a correct part 1 submission.
- The solver API used by the scraper is:

    https://advent.fly.dev/solve/

  This script submits your input to that service and then submits the
  resulting answers to AoC via aocd.
