import sys
from collections import defaultdict
from pathlib import Path


def list_unique_repo_files(root: str):
    root = Path(root).resolve()
    seen = set()

    for path in root.rglob("*.py"):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue

        parts = rel.parts

        # Expect: user/repo/<rest...>
        if len(parts) < 3:
            continue

        key = Path(*parts[2:])  # strip user/repo

        if key not in seen:
            seen.add(key)

    return seen


def main():
    """
    List all unique Python files in the given directory by their path inside the repository.
    """
    directory = sys.argv[1] if len(sys.argv) > 1 else "data/repos"
    paths = list_unique_repo_files(directory)

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
