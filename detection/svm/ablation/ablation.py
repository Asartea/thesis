from typing import TypedDict
from pathlib import Path
import json
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
    {"name": "tfidf_only", "use_tfidf": True, "use_ast": False, "use_token": False},
    {"name": "ast_only", "use_tfidf": False, "use_ast": True, "use_token": False},
    {"name": "token_only", "use_tfidf": False, "use_ast": False, "use_token": True},
    {"name": "tfidf_ast", "use_tfidf": True, "use_ast": True, "use_token": False},
    {"name": "tfidf_token", "use_tfidf": True, "use_ast": False, "use_token": True},
    {"name": "ast_token", "use_tfidf": False, "use_ast": True, "use_token": True},
    {"name": "full_model", "use_tfidf": True, "use_ast": True, "use_token": True},
]


def run_ablation_cv(
    samples: Samples,
    *,
    group_mode: str,
    n_splits: int,
    log_path: Path,
):
    labels = np.array([get_label(s) for s in samples])
    groups = np.array([build_group_id(s, mode=group_mode) for s in samples])

    gkf = GroupKFold(n_splits=n_splits)

    log = {"group_mode": group_mode, "n_splits": n_splits, "ablation_runs": []}

    for ablation in ABLATIONS:
        fold_accs, fold_f1s = [], []
        cm_sum = np.zeros((2, 2), dtype=int)

        ablation_log = {"ablation": ablation["name"], "folds": []}

        for fold, (train_idx, test_idx) in enumerate(
            gkf.split(samples, labels, groups)
        ):
            train_samples = [samples[i] for i in train_idx]
            test_samples = [samples[i] for i in test_idx]

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

            acc = float(metrics["accuracy"])
            f1 = float(metrics["f1_macro"])
            cm = metrics["confusion_matrix"]

            fold_accs.append(acc)
            fold_f1s.append(f1)
            cm_sum += cm

            ablation_log["folds"].append(
                {
                    "fold": fold,
                    "accuracy": acc,
                    "f1_macro": f1,
                    "confusion_matrix": cm.tolist(),
                }
            )

        result = {
            "name": ablation["name"],
            "group_mode": group_mode,
            "accuracy_mean": float(np.mean(fold_accs)),
            "accuracy_std": float(np.std(fold_accs)),
            "f1_mean": float(np.mean(fold_f1s)),
            "f1_std": float(np.std(fold_f1s)),
            "confusion_matrix_sum": cm_sum.tolist(),
        }

        ablation_log["summary"] = result
        log["ablation_runs"].append(ablation_log)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))


def main():
    sample_paths = [
        "data/samples.jsonl",
        "data/normal/normal_samples.jsonl",
        "data/competitive_programming/comp_samples.jsonl",
    ]

    for path in sample_paths:
        print(f"Running experiments for {path}")

        samples = load_samples(path)

        out_file = Path("results") / f"{Path(path).stem}_ablation.json"

        run_ablation_cv(
            samples,
            group_mode="problem",
            n_splits=5,
            log_path=out_file.with_name(out_file.stem + "_problem.json"),
        )

        run_ablation_cv(
            samples,
            group_mode="author_like",
            n_splits=5,
            log_path=out_file.with_name(out_file.stem + "_author.json"),
        )

        run_ablation_cv(
            samples,
            group_mode="strict",
            n_splits=5,
            log_path=out_file.with_name(out_file.stem + "_strict.json"),
        )


if __name__ == "__main__":
    main()
