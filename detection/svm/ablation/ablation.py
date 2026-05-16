from typing import TypedDict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from detection.svm.classifier import (
    build_features,
    build_group_id,
    evaluate_model,
    get_label,
    train_model,
)

from models.models import Samples
from utils import load_samples


class AblationConfig(TypedDict):
    name: str
    use_tfidf: bool
    use_ast: bool
    use_token: bool


ABLATIONS: list[AblationConfig] = [
    {
        "name": "tfidf_only",
        "use_tfidf": True,
        "use_ast": False,
        "use_token": False,
    },
    {
        "name": "ast_only",
        "use_tfidf": False,
        "use_ast": True,
        "use_token": False,
    },
    {
        "name": "token_only",
        "use_tfidf": False,
        "use_ast": False,
        "use_token": True,
    },
    {
        "name": "tfidf_ast",
        "use_tfidf": True,
        "use_ast": True,
        "use_token": False,
    },
    {
        "name": "tfidf_token",
        "use_tfidf": True,
        "use_ast": False,
        "use_token": True,
    },
    {
        "name": "ast_token",
        "use_tfidf": False,
        "use_ast": True,
        "use_token": True,
    },
    {
        "name": "full_model",
        "use_tfidf": True,
        "use_ast": True,
        "use_token": True,
    },
]


def run_ablation_cv(
    samples: Samples,
    *,
    group_mode: str = "problem",
    n_splits: int = 5,
):

    labels = np.array([get_label(s) for s in samples])
    groups = np.array([build_group_id(s, mode=group_mode) for s in samples])

    gkf = GroupKFold(n_splits=n_splits)

    results: list[dict[str, str | float]] = []

    for ablation in ABLATIONS:
        fold_accs: list[float] = []
        fold_f1s: list[float] = []
        cm_sum = np.zeros((2, 2), dtype=int)
        print("\n" + "=" * 80)
        print(f"ABLATION: {ablation['name']} | GROUP: {group_mode}")
        print("=" * 80)

        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(samples, labels, groups)
        ):

            train_samples: Samples = [samples[i] for i in train_idx]
            test_samples: Samples = [samples[i] for i in test_idx]

            y_train = labels[train_idx]
            y_test = labels[test_idx]

            tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
            )

            X_train, X_test = build_features(
                train_samples,
                test_samples,
                tfidf,
                use_tfidf=ablation["use_tfidf"],
                use_ast=ablation["use_ast"],
                use_token=ablation["use_token"],
            )

            model = train_model(X_train, y_train)

            metrics = evaluate_model(model, X_test, y_test)

            acc = metrics["accuracy"]
            f1 = metrics["f1_macro"]
            cm = metrics["confusion_matrix"]
            fold_accs.append(acc)
            cm_sum += cm
            fold_f1s.append(f1)

            print(f"Fold {fold}: acc={acc:.4f} | f1={f1:.4f}")
            print("Confusion matrix:")

            print(cm)
        print("\nFinal aggregated confusion matrix:")
        print(cm_sum)

        results.append(
            {
                "name": ablation["name"],
                "group_mode": group_mode,
                "accuracy_mean": float(np.mean(fold_accs)),
                "accuracy_std": float(np.std(fold_accs)),
                "f1_mean": float(np.mean(fold_f1s)),
                "f1_std": float(np.std(fold_f1s)),
            }
        )

    return results


def print_ablation_results(results: list[dict[str, str | float]]):
    print("\n" + "#" * 80)
    print("ABALATION SUMMARY (GROUPED CV)")
    print("#" * 80)

    for r in results:
        print(
            f"{r['name']:15s} | "
            f"{r['group_mode']:10s} | "
            f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}"
        )


def main():
    samples = load_samples("data/normal/samples.jsonl")

    ablation_results_problem = run_ablation_cv(samples, group_mode="problem")
    print_ablation_results(ablation_results_problem)
    ablation_results_author = run_ablation_cv(samples, group_mode="author_like")
    print_ablation_results(ablation_results_author)

    ablation_results_strict = run_ablation_cv(samples, group_mode="strict")
    print_ablation_results(ablation_results_strict)


if __name__ == "__main__":
    main()
