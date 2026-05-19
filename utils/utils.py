import json
from pathlib import Path
from typing import Any
from models.models import Samples


def load_samples(path: str | Path) -> Samples:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_labels(samples: Samples) -> list[int]:
    return [0 if s["label"] == "machine" else 1 for s in samples]


def write_to_jsonl(output: list[Any], output_path: str | Path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            json.dump(item, f)
            f.write("\n")


def load_jsonl(path: str | Path) -> list[Any]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
