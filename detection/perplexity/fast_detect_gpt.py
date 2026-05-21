import argparse
import random
from os import environ
from pathlib import Path

import numpy as np
import torch
from fast_detect_gpt.scripts.local_infer import FastDetectGPT
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from models.models import HumanSample, Samples
from utils.utils import load_samples

SCORING_MODEL = "bigcode/starcoder2-15b"
SAMPLING_MODEL = "bigcode/starcoder2-15b"

CACHE_DIR = environ.get("HF_CACHE_DIR", None)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.inference_mode()
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
            crit, _ = detector.compute_crit(sample["code"])
            scores.append(float(crit))

        except Exception as e:
            print(f"[ERROR] {e}")
        del crit
        torch.cuda.empty_cache()

    threshold = np.percentile(scores, percentile)

    print()
    print(f"Threshold percentile : {percentile}")
    print(f"Threshold value      : {threshold:.4f}")

    return threshold, scores


@torch.inference_mode()
def classify_samples(
    detector: FastDetectGPT,
    samples: Samples,
    threshold: float,
):

    results: list[dict[str, str | float]] = []

    for i, sample in enumerate(tqdm(samples)):
        try:

            crit, _ = detector.compute_crit(sample["code"])

            crit = float(crit)

            is_ai = crit > threshold
        except torch.OutOfMemoryError:
            print(f"OOM at sample {sample['code']}")
            print(f"Code length chars: {len(sample['code'])}")

            ntokens = len(detector.scoring_tokenizer.encode(sample["code"]))

            print(f"Token count: {ntokens}")
            torch.cuda.empty_cache()

            continue

        results.append(
            {
                "score": crit,
                "pred_label": is_ai,
                "actual_label": sample["label"],
            }
        )
        del crit
        torch.cuda.empty_cache()
    return results


def split_calibration_samples(samples: list[HumanSample], n: int = 300):
    rng = random.Random(227)  # fixed seed for reproducibility
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    calibration_samples = shuffled[:n]
    test_samples = shuffled[n:]
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

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auroc": auc,
        "confusion_matrix": cm.tolist(),
    }


def write_results(
    output_path: Path,
    model_name: str,
    dataset_name: str,
    threshold: float,
    metrics: dict[str, float | str | None],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Threshold: {threshold:.6f}\n\n")

        f.write("=== METRICS ===\n")
        f.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall   : {metrics['recall']:.4f}\n")
        f.write(f"F1       : {metrics['f1_score']:.4f}\n")

        if metrics["auroc"] is not None:
            f.write(f"AUROC    : {metrics['auroc']:.4f}\n")
        else:
            f.write("AUROC    : N/A\n")

        f.write("\nConfusion Matrix:\n")
        f.write(str(metrics["confusion_matrix"]))
        f.write("\n")


def run(model_name: str):
    base_dir = Path("data")
    human_samples_path = (
        base_dir / "normal" / "human_samples.jsonl"
    )  # identical between normal and comp

    normal_machine_samples_path = base_dir / "normal" / "machine_samples.jsonl"
    comp_machine_samples_path = (
        base_dir / "competitive_programming" / "machine_samples.jsonl"
    )
    normal_machine_samples = load_samples(normal_machine_samples_path)
    comp_machine_samples = load_samples(comp_machine_samples_path)
    human_samples = load_samples(human_samples_path)

    calibration_samples, human_test_samples = split_calibration_samples(human_samples)

    detector = FastDetectGPT(
        scoring_model_name=model_name,
        sampling_model_name=model_name,
        cache_dir=CACHE_DIR,
        extra_distrib_params={},
    )

    threshold, _ = calibrate_threshold(detector, calibration_samples, percentile=95)

    samples = {
        "Normal": normal_machine_samples + human_test_samples,
        "Competitive Programming": comp_machine_samples + human_test_samples,
    }
    for name, sample_set in samples.items():
        print(f"\n=== CLASSIFYING SAMPLE SET {name} ===")
        results = classify_samples(detector, sample_set, threshold)
        metrics = evaluate(results)
        write_results(
            output_path=Path(
                f"results/perplexity_{model_name.replace('/', '_')}_{name}.txt"
            ),
            model_name=model_name,
            dataset_name=name,
            threshold=threshold,
            metrics=metrics,
        )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Run FastDetectGPT on AoC samples")
    parser.add_argument("--model", type=str, default=SCORING_MODEL)
    return parser.parse_args()


def main():
    args = parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
