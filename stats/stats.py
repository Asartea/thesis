from collections import Counter


def github_user_distribution():
    distribution = Counter()

    with open(
        "data_gathering/data/deduplicated_leaderboard_data.tsv", "r", encoding="utf-8"
    ) as f:
        for line in f:
            year = line.split("\t")[0]
            distribution[year] += 1

    return distribution


def main():
    print(github_user_distribution())
