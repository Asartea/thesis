from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from models.models import Samples
from utils import load_samples


def plot_distribution(
    title: str,
    x_label: str,
    y_label: str,
    counter: Counter[str],
    file_name: str = "distribution.png",
):
    categories = sorted(counter.keys())
    values = [counter[c] for c in categories]

    plt.figure(figsize=(10, 7))
    x = range(len(categories))

    plt.bar(x, values)
    plt.xticks(x, categories, rotation=45)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(file_name)


def plot_grouped_distribution(
    title: str,
    x_label: str,
    y_label: str,
    counter: dict[str, Counter[bool]],
    file_name: str = "grouped_distribution.png",
):
    models = sorted(counter.keys())

    # Ensure consistent ordering of categories (booleans or strings both work)
    all_categories = sorted(
        {cat for model_data in counter.values() for cat in model_data.keys()}
    )

    x = range(len(models))
    width = 0.8 / len(all_categories)  # dynamic bar width

    plt.figure(figsize=(10, 7))

    for i, category in enumerate(all_categories):
        values = [counter[model].get(category, 0) for model in models]

        offset = [p + i * width for p in x]
        plt.bar(offset, values, width=width, label=str(category))

    center_offset = (len(all_categories) - 1) * width / 2
    plt.xticks([p + center_offset for p in x], models, rotation=45)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(title="Category")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_grouped_line_plot(
    title: str,
    x_label: str,
    y_label: str,
    legend_title: str,
    counter: dict[str, Counter[bool]],
    file_name: str = "grouped_lineplot.png",
):
    models = sorted(counter.keys())

    # Consistent category ordering
    all_categories = sorted(
        {cat for model_data in counter.values() for cat in model_data.keys()}
    )

    x = list(range(len(models)))

    plt.figure(figsize=(10, 7))

    for category in all_categories:
        values = [counter[model].get(category, 0) for model in models]

        plt.plot(
            x,
            values,
            marker="o",
            linewidth=2,
            label=str(category),
        )

    plt.xticks(x, models, rotation=45)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.grid(alpha=0.3)
    plt.ylim(bottom=0)
    plt.legend(title=legend_title)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def year_distribution(samples: Samples):
    distributions = defaultdict(lambda: defaultdict(Counter))

    for sample in samples:
        year = sample["year"]
        day = sample["day"].zfill(2)

        if sample["label"] == "human":
            category = "human"
        else:
            category = sample["model"]

        distributions[year][category][day] += 1

    return distributions


def class_distribution(samples: Samples):
    distribution = Counter()

    for sample in samples:
        if sample["label"] == "human":
            distribution["human"] += 1
        else:
            distribution[sample["model"]] += 1

    return distribution


def competitive_programming_distribution(samples: Samples):
    distribution = defaultdict(lambda: Counter())

    for sample in samples:
        if sample["label"] == "human":
            continue

        model = sample["model"]
        used_competitive_programming = sample["use_comp_programming"]

        distribution[model][used_competitive_programming] += 1

    return distribution


def class_per_day_distribution(samples: Samples):
    distribution = defaultdict(lambda: defaultdict(Counter))

    for sample in samples:
        year = sample["year"]
        day = sample["day"].zfill(2)

        if sample["label"] == "human":
            category = "human"
        else:
            category = sample["model"]

        distribution[year][day][category] += 1

    return distribution


def heatmap_from_year_data(year_data):
    # convert: day -> source -> count
    transformed = defaultdict(lambda: Counter())

    for day, sources in year_data.items():
        for source, count in sources.items():
            transformed[source][day] = count

    return transformed


def main():
    samples = load_samples(Path("data/samples.jsonl"))
    # year_distributions = year_distribution(samples)
    class_distributions = class_distribution(samples)
    competitive_programming_distributions = competitive_programming_distribution(
        samples
    )
    class_per_day_distributions = class_per_day_distribution(samples)

    plot_distribution(
        title="Class Distribution",
        x_label="Class",
        y_label="Count",
        counter=class_distributions,
        file_name="class_distribution.png",
    )

    plot_grouped_distribution(
        title="Competitive Programming Samples by Model",
        x_label="Model",
        y_label="Count",
        counter=competitive_programming_distributions,
        file_name="competitive_programming_distribution.png",
    )

    plot_grouped_line_plot(
        title="Source Distribution per Day for 2021",
        x_label="Day",
        y_label="Count",
        legend_title="Source",
        counter=class_per_day_distributions["2021"],
        file_name="source_distribution_2021_lineplot.png",
    )

    plot_grouped_distribution(
        title="Source Distribution per Day for 2021",
        x_label="Model",
        y_label="Count",
        counter=class_per_day_distributions["2021"],
        file_name="source_distribution_2021.png",
    )

    plot_grouped_line_plot(
        title="Source Distribution per Day for 2024",
        x_label="Day",
        y_label="Count",
        legend_title="Source",
        counter=class_per_day_distributions["2024"],
        file_name="source_distribution_2024_lineplot.png",
    )

    plot_grouped_distribution(
        title="Source Distribution per Day for 2024",
        x_label="Model",
        y_label="Count",
        counter=class_per_day_distributions["2024"],
        file_name="source_distribution_2024.png",
    )

    print(class_per_day_distributions["2024"])


if __name__ == "__main__":
    main()
