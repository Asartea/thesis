from fast_detect_gpt.scripts.local_infer import FastDetectGPT
from types import SimpleNamespace
from utils.utils import load_samples
import random


def main():
    scoring_model = "gpt-j-6B"
    sampling_model = "gpt-neo-2.7B"
    args = SimpleNamespace(
        scoring_model_name=scoring_model,
        sampling_model_name=sampling_model,
        cache_dir="./cache",
    )
    detector = FastDetectGPT(args)
    samples = load_samples("data/samples.jsonl")
    samples = random.sample(samples, 10)
    for s in samples:
        code = s["code"]
        prob, crit, ntokens = detector.compute_prob(code)
        print(f"Code: {code}")
        print(
            f"Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated."
        )
        print(f"Actual label: {s['label']}")


if __name__ == "__main__":
    main()
