import ast
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit

from detection.svm.ablation.preprocess import (
    CommentRemover,
    DocstringRemover,
    ScopedRenamer,
)
from detection.svm.classifier import (
    build_features,
    build_group_id,
    evaluate_model,
    get_label,
    train_model,
)
from models.models import Samples
from utils.utils import load_samples


class FeatureConfig(TypedDict):
    name: str
    remove_comments: bool
    remove_docstrings: bool
    normalize_identifiers: bool


class TFIDFConfig(TypedDict):
    name: str
    use_idf: bool
    sublinear_tf: bool
    ngram_range: tuple[int, int]
    analyzer: Literal["char_wb", "word"]
    min_df: int


class Log(TypedDict):
    group_mode: str
    n_splits: int
    ablation_runs: list[dict[str, str | float | list[list[int]]]]


FEATURE_SETS: list[FeatureConfig] = [
    {
        "name": "baseline",
        "remove_comments": False,
        "remove_docstrings": False,
        "normalize_identifiers": False,
    },
    {
        "name": "no_comments",
        "remove_comments": True,
        "remove_docstrings": False,
        "normalize_identifiers": False,
    },
    {
        "name": "no_docstrings",
        "remove_comments": False,
        "remove_docstrings": True,
        "normalize_identifiers": False,
    },
    {
        "name": "normalized_identifiers",
        "remove_comments": False,
        "remove_docstrings": False,
        "normalize_identifiers": True,
    },
    {
        "name": "fully_normalized",
        "remove_comments": True,
        "remove_docstrings": True,
        "normalize_identifiers": True,
    },
]

TFIDF_SETS: list[TFIDFConfig] = [
    {
        "name": "char_3_5_standard",
        "use_idf": True,
        "sublinear_tf": True,
        "ngram_range": (3, 5),
        "analyzer": "char_wb",
        "min_df": 2,
    },
    {
        "name": "char_3_5_no_idf",
        "use_idf": False,
        "sublinear_tf": True,
        "ngram_range": (3, 5),
        "analyzer": "char_wb",
        "min_df": 2,
    },
    {
        "name": "char_3_5_linear_tf",
        "use_idf": True,
        "sublinear_tf": False,
        "ngram_range": (3, 5),
        "analyzer": "char_wb",
        "min_df": 2,
    },
    {
        "name": "word_unigram",
        "use_idf": True,
        "sublinear_tf": True,
        "ngram_range": (1, 1),
        "analyzer": "word",
        "min_df": 2,
    },
    {
        "name": "char_4_6",
        "use_idf": True,
        "sublinear_tf": True,
        "ngram_range": (4, 6),
        "analyzer": "char_wb",
        "min_df": 2,
    },
    {
        "name": "char_high_min_df",
        "use_idf": True,
        "sublinear_tf": True,
        "ngram_range": (3, 5),
        "analyzer": "char_wb",
        "min_df": 5,
    },
]


def preprocess_code(
    code: str,
    *,
    comment_removal: bool,
    docstring_removal: bool,
    identifier_normalization: bool,
) -> str:

    if not comment_removal and not docstring_removal and not identifier_normalization:
        return code

    # ast.parse/unparse already removes comments, and on the other hand tokenize can lead to issues with ast parsing.
    # So if we only need to remove comments we use comment remover, else we use the ast based approach which removes
    # comments as a side effect.
    if comment_removal and (not docstring_removal and not identifier_normalization):
        return CommentRemover.remove_comments(code)

    tree = ast.parse(code)
    if identifier_normalization:
        tree = ScopedRenamer().visit(tree)
    if docstring_removal:
        tree = DocstringRemover().visit(tree)

    tree = ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def preprocess_samples(
    samples: Samples,
    config: FeatureConfig,
) -> Samples:
    preprocessed_samples: Samples = []

    for sample in samples:
        sample["code"] = preprocess_code(
            sample["code"],
            comment_removal=config["remove_comments"],
            docstring_removal=config["remove_docstrings"],
            identifier_normalization=config["normalize_identifiers"],
        )

        preprocessed_samples.append(sample)
    return preprocessed_samples


def split_samples(samples: Samples, group_mode: str, test_size: float = 0.2):
    labels = np.array([get_label(s) for s in samples])
    groups = np.array([build_group_id(s, mode=group_mode) for s in samples])

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=71615)

    train_idx, test_idx = next(splitter.split(samples, labels, groups))
    return train_idx, test_idx


def run_feature_ablation(
    samples: Samples,
    *,
    group_mode: str,
    log_path: Path,
):
    labels = np.array([get_label(s) for s in samples])

    log: Log = {"group_mode": group_mode, "n_splits": 1, "ablation_runs": []}
    train_idx, test_idx = split_samples(samples, group_mode=group_mode)

    for config in FEATURE_SETS:

        print(f"Running feature set: {config['name']}")

        sample_copy = copy.deepcopy(samples)

        processed_samples = preprocess_samples(sample_copy, config)

        train_samples = [processed_samples[i] for i in train_idx]

        test_samples = [processed_samples[i] for i in test_idx]

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
            use_tfidf=True,
            use_ast=False,
            use_token=False,
        )

        model = train_model(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        result = {
            "name": config["name"],
            "accuracy": float(metrics["accuracy"]),
            "f1_macro": float(metrics["f1_macro"]),
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }

        log["ablation_runs"].append(result)

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path.write_text(json.dumps(log, indent=2))


def run_tfidf_ablation(samples: Samples, group_mode: str, log_path: Path):

    labels = np.array([get_label(s) for s in samples])
    log: Log = {"group_mode": group_mode, "n_splits": 1, "ablation_runs": []}

    train_idx, test_idx = split_samples(samples, group_mode=group_mode)

    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    for config in TFIDF_SETS:

        tfidf = TfidfVectorizer(
            analyzer=config["analyzer"],
            ngram_range=config["ngram_range"],
            min_df=config["min_df"],
            sublinear_tf=config["sublinear_tf"],
            use_idf=config["use_idf"],
        )

        X_train, X_test = build_features(
            train_samples,
            test_samples,
            tfidf,
            use_tfidf=True,
            use_ast=False,
            use_token=False,
        )

        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        log["ablation_runs"].append(
            {
                "tfidf_set": config["name"],
                "accuracy": float(metrics["accuracy"]),
                "f1_macro": float(metrics["f1_macro"]),
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            }
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))


def main():
    sample_paths = [
        "data/normal/normal_samples.jsonl",
        "data/competitive_programming/comp_samples.jsonl",
    ]

    for path in sample_paths:
        print(f"Running experiments for {path}")

        samples = load_samples(path)

        # There are 19 samples which despite passing the validation check in validation.validation.py contain code that
        # cannot be parsed by ast.parse, which causes the feature extraction to fail. This filters those out.
        invalid = Counter()
        valid_samples = []
        for sample in samples:
            try:
                ast.parse(sample["code"])
                valid_samples.append(sample)
            except SyntaxError:
                invalid["syntax"] += 1
                print(f"Found invalid sample with label: {sample['label']}")
                continue
        print(
            f"Found {invalid['syntax']} samples with invalid syntax in {path}, skipping them."
        )

        samples = valid_samples

        out_file_features = Path("results") / f"{Path(path).stem}_preprocessing.json"

        run_feature_ablation(
            samples,
            group_mode="problem",
            log_path=out_file_features.with_name(
                out_file_features.stem + "_problem.json"
            ),
        )

        run_feature_ablation(
            samples,
            group_mode="author_like",
            log_path=out_file_features.with_name(
                out_file_features.stem + "_author.json"
            ),
        )

        run_feature_ablation(
            samples,
            group_mode="strict",
            log_path=out_file_features.with_name(
                out_file_features.stem + "_strict.json"
            ),
        )

        out_file_tfidf = Path("results") / f"{Path(path).stem}_tfidf.json"

        run_tfidf_ablation(
            samples,
            group_mode="problem",
            log_path=out_file_tfidf.with_name(out_file_tfidf.stem + "_problem.json"),
        )

        run_tfidf_ablation(
            samples,
            group_mode="author_like",
            log_path=out_file_tfidf.with_name(out_file_tfidf.stem + "_author.json"),
        )

        run_tfidf_ablation(
            samples,
            group_mode="strict",
            log_path=out_file_tfidf.with_name(out_file_tfidf.stem + "_strict.json"),
        )


if __name__ == "__main__":
    main()
