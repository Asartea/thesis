import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import random
from os import environ
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from fast_detect_gpt.scripts.local_infer import FastDetectGPT

from utils.utils import load_samples
from models.models import Samples

SCORING_MODEL = "bigcode/starcoder2-15b"
SAMPLING_MODEL = "bigcode/starcoder2-15b"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CACHE_DIR = environ.get("HF_CACHE_DIR", None)


def calibrate_threshold(
    detector: FastDetectGPT,
    human_validation_samples: Samples,
    percentile: int = 95,
):
    """
    Learn threshold from human-written code only.

    Anything above threshold is classified as AI.
    """

    scores: list[float] = []

    for sample in tqdm(human_validation_samples):

        try:
            crit, _ = detector.compute_crit(sample)
            scores.append(float(crit))

        except Exception as e:
            print(f"[ERROR] {e}")

    threshold = np.percentile(scores, percentile)

    print()
    print(f"Threshold percentile : {percentile}")
    print(f"Threshold value      : {threshold:.4f}")

    return threshold, scores


def classify_samples(
    detector: FastDetectGPT,
    samples: Samples,
    threshold: float,
):

    results: list[dict[str, str | float]] = []

    for sample in tqdm(samples):

        crit, _ = detector.compute_crit(sample)

        crit = float(crit)

        is_ai = crit > threshold

        results.append(
            {
                "score": crit,
                "pred_label": is_ai,
                "actual_label": sample["label"],
            }
        )
    return results


def split_calibration_samples(samples: Samples, n: int = 300):
    rng = random.Random(227)  # fixed seed for reproducibility
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    calibration_samples = [s for s in shuffled if s["label"] == "human"][:300]
    test_samples = [s for s in shuffled if s not in calibration_samples]
    return calibration_samples, test_samples


def evaluate(results):
    y_true = []
    y_pred = []
    y_score = []

    for r in results:
        if "score" not in r or r["score"] is None:
            continue

        y_true.append(1 if r["actual_label"] != "human" else 0)
        y_pred.append(1 if r["pred_label"] else 0)
        y_score.append(r["score"])

    print("\n=== EVALUATION ===")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = None

    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"AUROC    : {auc:.4f if auc else 'N/A'}")
    print("\nConfusion matrix:")
    print(cm)


def main():
    samples_path = Path("data") / "samples.jsonl"
    samples = load_samples(samples_path)
    detector = FastDetectGPT(
        scoring_model_name=SCORING_MODEL,
        sampling_model_name=SAMPLING_MODEL,
        device=DEVICE,
        cache_dir=CACHE_DIR,
        extra_distrib_params={},
    )
    calibration_samples, test_samples = split_calibration_samples(samples)
    threshold, human_scores = calibrate_threshold(
        detector, calibration_samples, percentile=95
    )
    results = classify_samples(detector, test_samples, threshold)
    evaluate(results)


if __name__ == "__main__":
    main()
