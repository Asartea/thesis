from pathlib import Path

import bs4


def parse_leaderboard_html(html: str, top_users: dict[str, str]) -> None:
    soup = bs4.BeautifulSoup(html, "html.parser")

    for entry in soup.select(".leaderboard-entry"):
        github_link = entry.select_one("a[href*='github.com']")
        if not github_link:
            continue

        user_id = str(entry.get("data-user-id"))
        if not user_id:
            continue

        github_url = str(github_link.get("href", "")).rstrip("/")

        top_users[user_id] = github_url


def extract_top_users(
    years: list[str], days: list[str], base_dir: str
) -> dict[str, str]:
    top_users: dict[str, str] = {}

    for y in years:
        global_path = Path(base_dir) / y / "html" / "global_leaderboard.html"

        if global_path.exists():
            html = global_path.read_text(encoding="utf-8")
            parse_leaderboard_html(html, top_users)

        for d in days:
            day_path = Path(base_dir) / y / "html" / f"day_{d}_leaderboard.html"

            if day_path.exists():
                html = day_path.read_text(encoding="utf-8")
                parse_leaderboard_html(html, top_users)

    return top_users
