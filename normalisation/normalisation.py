def normalise_day(day: str) -> str:
    """Normalise day strings to a consistent format."""
    day = day.strip().lower()
    if day.startswith("day"):
        day = day[3:]
    elif day.startswith("d"):
        day = day[1:]

    return day.zfill(2)


def normalise_year(year: str) -> str:
    """Normalise year strings to a consistent format."""
    year = year.strip()
    if len(year) == 2:
        year = "20" + year
    return year
