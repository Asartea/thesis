import json
from pathlib import Path

from find_solutions.text_extractor import HumanSample


def save_jsonl(samples: list[HumanSample], output_file: Path):
    with open(output_file, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
