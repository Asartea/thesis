import subprocess
from pathlib import Path


def clone_repo(repo: str, target_dir: Path) -> bool:
    if Path(target_dir).exists():
        print(f"Skipping (already exists): {target_dir}")
        return True

    Path(target_dir).parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo, target_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )

        if result.returncode != 0:
            print(f"Failed to clone {repo}: {result.returncode}")
            print(result.stderr)
            return False

        return True

    except subprocess.TimeoutExpired:
        return False
