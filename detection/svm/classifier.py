import ast
import io
import json
import keyword
import tokenize
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

AST_NODE_TYPES = [
    "FunctionDef",
    "For",
    "While",
    "If",
    "ListComp",
    "DictComp",
    "GeneratorExp",
    "Lambda",
    "Try",
    "ClassDef",
    "Import",
    "ImportFrom",
    "With",
    "Match",
]


# ============================================================
# Data Loading
# ============================================================


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_label(sample: dict[str, Any]) -> int:
    return 0 if "repo" in sample else 1


def build_group_id(sample: dict[str, Any]) -> str:
    year = sample.get("year", "")
    day = sample.get("day", "")

    return f"{year}_{day}"


def extract_ast_features(code: str) -> list[float]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [0.0] * len(AST_NODE_TYPES)

    counter = Counter(type(node).__name__ for node in ast.walk(tree))
    total = sum(counter.values()) or 1

    return [counter[node_type] / total for node_type in AST_NODE_TYPES]


def extract_token_features(code: str) -> list[float]:
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except Exception:
        return [0.0] * 6

    total = len(tokens) or 1

    identifiers = 0
    keywords_count = 0
    operators = 0
    strings = 0
    numbers = 0

    identifier_lengths = []

    for tok in tokens:
        tok_type = tok.type
        tok_str = tok.string

        if tok_type == tokenize.NAME:
            if keyword.iskeyword(tok_str):
                keywords_count += 1
            else:
                identifiers += 1
                identifier_lengths.append(len(tok_str))

        elif tok_type == tokenize.OP:
            operators += 1

        elif tok_type == tokenize.STRING:
            strings += 1

        elif tok_type == tokenize.NUMBER:
            numbers += 1

    avg_identifier_length = np.mean(identifier_lengths) if identifier_lengths else 0.0

    return [
        identifiers / total,
        keywords_count / total,
        operators / total,
        strings / total,
        numbers / total,
        avg_identifier_length,
    ]


def extract_dense_features(code: str) -> list[float]:
    return extract_ast_features(code) + extract_token_features(code)


def split_samples(
    samples: list[dict[str, Any]],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.array([get_label(sample) for sample in samples])

    groups = np.array([build_group_id(sample) for sample in samples])

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(splitter.split(samples, labels, groups))

    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    return train_samples, test_samples


def build_feature_matrices(
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    *,
    max_features: int = 50_000,
) -> tuple[
    csr_matrix,
    csr_matrix,
    np.ndarray,
    np.ndarray,
    TfidfVectorizer,
    StandardScaler,
]:
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform([s["code"] for s in train_samples])

    X_test_tfidf = vectorizer.transform([s["code"] for s in test_samples])

    X_train_dense = np.array([extract_dense_features(s["code"]) for s in train_samples])

    X_test_dense = np.array([extract_dense_features(s["code"]) for s in test_samples])

    scaler = StandardScaler()

    X_train_dense = scaler.fit_transform(X_train_dense)
    X_test_dense = scaler.transform(X_test_dense)

    X_train_dense = csr_matrix(X_train_dense)
    X_test_dense = csr_matrix(X_test_dense)

    X_train = hstack([X_train_tfidf, X_train_dense])
    X_test = hstack([X_test_tfidf, X_test_dense])

    y_train = np.array([get_label(s) for s in train_samples])

    y_test = np.array([get_label(s) for s in test_samples])

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer,
        scaler,
    )


def train_model(
    X_train: csr_matrix,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
) -> LinearSVC:
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=random_state,
        max_iter=10_000,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(
    model: LinearSVC,
    X_test: csr_matrix,
    y_test: np.ndarray,
) -> dict[str, Any]:
    predictions = model.predict(X_test)

    return {
        "accuracy": float(np.mean(predictions == y_test)),
        "report": classification_report(
            y_test,
            predictions,
            target_names=["human", "llm"],
            output_dict=True,
        ),
    }
