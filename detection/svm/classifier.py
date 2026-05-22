import ast
import io
import keyword
import tokenize
from collections import Counter

import numpy as np
from scipy.sparse import hstack, spmatrix, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.base import clone
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from models.models import HumanSample, LLMSample, Sample, Samples

import warnings

warnings.filterwarnings("error")


def get_label(sample: Sample) -> int:
    return 0 if "repo" in sample else 1


def build_group_id(sample: HumanSample | LLMSample, mode: str = "problem") -> str:
    """
    Grouping strategy selector.
    """

    year = sample.get("year", "")
    day = sample.get("day", "")
    base = f"{year}_{day}"

    if mode == "problem":
        return base

    if mode == "author_like":
        if sample["label"] == "human":
            return f"human_{sample.get('author', '')}"
        return f"llm_{sample.get('model', '')}_{sample.get('code_variant', '')}_{sample.get('style_variant', '')}"

    if mode == "strict":
        if sample["label"] == "human":
            return f"{base}_human_{sample['author']}"
        return (
            f"{base}_llm_"
            f"{sample.get('model', 'unknown')}_"
            f"{sample.get('code_variant', '')}_"
            f"{sample.get('style_variant', '')}"
        )

    raise ValueError(f"Unknown mode: {mode}")


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


CONTROL_FLOW = {"If", "For", "While", "Try", "With", "Match"}
DECLARATIONS = {"FunctionDef", "ClassDef", "Lambda"}


def _ast_depth(node: ast.AST) -> int:
    """Compute max AST depth."""
    if not list(ast.iter_child_nodes(node)):
        return 1
    return 1 + max(_ast_depth(child) for child in ast.iter_child_nodes(node))


def extract_ast_features(code: str) -> list[float]:
    """
    Enhanced AST features:
    - node type distribution
    - control flow ratio
    - declaration ratio
    - AST entropy
    - depth statistics
    """

    tree = ast.parse(
        code
    )  # samples are prevalidated (via validation.validation.py) so this should not raise. However, it turns out there are some weird edge cases that pass validation but fail here, so the caller should make sure to catch SyntaxErrors and skip those samples.

    nodes = list(ast.walk(tree))
    total_nodes = len(nodes) or 1

    counter = Counter(type(n).__name__ for n in nodes)

    node_freqs = [counter[t] / total_nodes for t in AST_NODE_TYPES]

    control_flow = sum(counter[n] for n in CONTROL_FLOW) / total_nodes
    declarations = sum(counter[n] for n in DECLARATIONS) / total_nodes

    freqs = [c / total_nodes for c in counter.values()]
    entropy = -np.sum(np.array(freqs) * np.log2(np.array(freqs) + 1e-12))

    try:
        depth = _ast_depth(tree)
    except RecursionError:
        depth = 0

    avg_depth = depth / max(len(list(ast.iter_child_nodes(tree))) + 1, 1)

    return node_freqs + [
        control_flow,
        declarations,
        entropy,
        avg_depth,
    ]


def _token_entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counter = Counter(tokens)
    total = sum(counter.values())
    return -sum((c / total) * np.log2(c / total + 1e-12) for c in counter.values())


def extract_token_features(code: str) -> list[float]:
    """
    Enhanced token features:
    - token type ratios
    - identifier statistics
    - naming style signals
    - lexical entropy
    - structural density proxies
    """

    tokens = list(
        tokenize.generate_tokens(io.StringIO(code).readline)
    )  # samples are prevalidated, so this should not raise

    total = len(tokens) or 1

    identifiers: list[str] = []
    keywords = 0
    ops = 0
    strings = 0
    numbers = 0

    all_token_strings: list[str] = []

    for t in tokens:
        all_token_strings.append(t.string)

        if t.type == tokenize.NAME:
            if keyword.iskeyword(t.string):
                keywords += 1
            else:
                identifiers.append(t.string)

        elif t.type == tokenize.OP:
            ops += 1
        elif t.type == tokenize.STRING:
            strings += 1
        elif t.type == tokenize.NUMBER:
            numbers += 1

    identifier_ratio = len(identifiers) / total
    keyword_ratio = keywords / total
    op_ratio = ops / total
    string_ratio = strings / total
    number_ratio = numbers / total

    if identifiers:
        id_lengths = [len(x) for x in identifiers]
        avg_id_len = sum(id_lengths) / len(id_lengths)
        max_id_len = max(id_lengths)
        var_id_len = sum((x - avg_id_len) ** 2 for x in id_lengths) / len(id_lengths)

        id_entropy = _token_entropy(identifiers)
    else:
        avg_id_len = max_id_len = var_id_len = 0.0
        id_entropy = 0.0

    token_entropy = _token_entropy(all_token_strings)

    density = len(tokens) / (len(identifiers) + 1)

    return [
        identifier_ratio,
        keyword_ratio,
        op_ratio,
        string_ratio,
        number_ratio,
        id_entropy,
        avg_id_len,
        max_id_len,
        var_id_len,
        token_entropy,
        density,
    ]


def build_features(
    train_samples: Samples,
    test_samples: Samples,
    tf_idf: TfidfVectorizer,
    *,
    use_tfidf: bool = True,
    use_ast: bool = True,
    use_token: bool = True,
):
    """Builds the feature matrices for training and testing.

    Args:
        train_samples: List of training samples, each a dict with a "code" key.
        test_samples: List of testing samples, each a dict with a "code" key.
        tf_idf: A TfidfVectorizer instance to use for TF-IDF features. This will be cloned inside the function to avoid state issues.
        use_tfidf: Whether to include TF-IDF features.
        use_ast: Whether to include AST-based features.
        use_token: Whether to include token-based features.

    Raises:
        ValueError: If none of the feature types are enabled.

    Returns:
        A tuple (X_train, X_test) where each is a sparse matrix of features.
    """

    tf_idf = clone(tf_idf)

    if not (use_tfidf or use_ast or use_token):
        raise ValueError("At least one feature type must be enabled")
    train_codes = [s["code"] for s in train_samples]
    test_codes = [s["code"] for s in test_samples]

    X_train_parts: list[np.ndarray | spmatrix] = []
    X_test_parts: list[np.ndarray | spmatrix] = []

    if use_tfidf:
        X_train_parts.append(tf_idf.fit_transform(train_codes))
        X_test_parts.append(tf_idf.transform(test_codes))

    dense_parts_train: list[np.ndarray] = []
    dense_parts_test: list[np.ndarray] = []

    if use_ast:
        ast_train = np.array([extract_ast_features(c) for c in train_codes])
        ast_test = np.array([extract_ast_features(c) for c in test_codes])
        dense_parts_train.append(ast_train)
        dense_parts_test.append(ast_test)

    if use_token:
        token_train = np.array([extract_token_features(c) for c in train_codes])
        token_test = np.array([extract_token_features(c) for c in test_codes])
        dense_parts_train.append(token_train)
        dense_parts_test.append(token_test)

    if dense_parts_train:
        scaler = StandardScaler()
        dense_train = scaler.fit_transform(np.hstack(dense_parts_train))
        dense_test = scaler.transform(np.hstack(dense_parts_test))

        X_train_parts.append(csr_matrix(dense_train))
        X_test_parts.append(csr_matrix(dense_test))

    X_train = hstack(X_train_parts)
    X_test = hstack(X_test_parts)

    return X_train, X_test


def split_samples(
    samples: Samples,
    *,
    test_size: float = 0.2,
    random_state: int = 74339,
    group_mode: str = "problem",
):
    labels = np.array([get_label(s) for s in samples])
    groups = np.array([build_group_id(s, mode=group_mode) for s in samples])

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(splitter.split(samples, labels, groups))

    X_train = [samples[i] for i in train_idx]
    X_test = [samples[i] for i in test_idx]

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    return X_train, X_test, y_train, y_test


def train_model(
    X_train,
    y_train,
):
    """Trains a LinearSVC model on the provided training data.

    Args:
        X_train: Sparse matrix of training features.
        y_train: Array of training labels.

    Returns:
        A trained LinearSVC model.
    """

    model = LinearSVC(
        C=1.0, class_weight="balanced", max_iter=10_000, random_state=63813
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    report = classification_report(
        y_test,
        preds,
        target_names=["human", "llm"],
        output_dict=True,
    )

    cm = confusion_matrix(y_test, preds)

    return {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_human": report["human"]["f1-score"],
        "f1_llm": report["llm"]["f1-score"],
        "confusion_matrix": cm,
        "report": report,
    }
