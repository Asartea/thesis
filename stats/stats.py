import json
from collections import Counter
from pathlib import Path

from models.models import Samples


def load_samples(path: Path) -> Samples:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def stats(samples: Samples) -> None:
    print(f"Total samples: {len(samples)}")
    print(f"Unique years: {len(set(sample['year'] for sample in samples))}")
    print(f"Unique days: {len(set(sample['day'] for sample in samples))}")
    print(f"Unique languages: {len(set(sample['language'] for sample in samples))}")
    print(f"Unique labels: {len(set(sample['label'] for sample in samples))}")

    model_counts = Counter(sample["model"] for sample in samples if "model" in sample)
    print("Model distribution:")
    for model, count in model_counts.most_common():
        print(f"  {model}: {count}")

    # print all days, and also print if they are human or if not to which model they belong
    day_counts = Counter(sample["day"] for sample in samples)
    print("Day distribution:")
    for day, count in day_counts.most_common():
        print(f"  Day {day}: {count}")
        for sample in samples:
            if sample["day"] == day:
                if "model" in sample:
                    print(f"    Model: {sample['model']}")
                else:
                    print("    Human")


def main():
    samples = load_samples(Path("samples.jsonl"))
    stats(samples)


if __name__ == "__main__":
    main()
