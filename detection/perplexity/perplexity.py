import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# CONFIG
# ============================================================

SCORING_MODEL = "Salesforce/codegen-2B-mono"
MASK_MODEL = "microsoft/codebert-base-mlm"

MASK_RATE = 0.05
TOP_P = 0.1
SAMPLES = 30
THRESHOLD = 0.97

SEED = 42

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

score_tok = AutoTokenizer.from_pretrained(SCORING_MODEL)

score_model = (
    AutoModelForCausalLM.from_pretrained(
        SCORING_MODEL,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
    )
    .eval()
    .to(DEVICE)
)

mask_tok = AutoTokenizer.from_pretrained(MASK_MODEL)

mask_model = AutoModelForMaskedLM.from_pretrained(MASK_MODEL).eval().to(DEVICE)

MASK_ID = mask_tok.mask_token_id

LOW_INFO = {
    "{",
    "}",
    "(",
    ")",
    "[",
    "]",
    ":",
    ";",
    ",",
    ".",
    "#",
}


def informative(tok):

    t = tok.strip()

    if not t:
        return False

    if t in LOW_INFO:
        return False

    if all(c in "_-=+*/<>!&|" for c in t):
        return False

    return True


def normalize(x):

    x = np.array(x)

    if np.allclose(x.max(), x.min()):
        return np.ones_like(x) * 0.5

    return (x - x.min()) / (x.max() - x.min())


def code_nll(code):

    if not code.strip():
        return 0.0

    enc = score_tok(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(DEVICE)

    with torch.no_grad():
        out = score_model(**enc, labels=enc["input_ids"])

    return out.loss.item()


def line_losses(lines):
    vals = []
    for line in lines:
        if not line.strip():
            vals.append(0.0)
            continue

        enc = score_tok(
            line,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(DEVICE)

        with torch.no_grad():
            out = score_model(**enc, labels=enc["input_ids"])

        vals.append(np.exp(out.loss.item()))

    return np.array(vals)


def perturb_code(
    lines,
    weights,
    mask_rate=MASK_RATE,
):

    perturbed = []

    for line, w in zip(lines, weights):
        toks = mask_tok.tokenize(line)

        if not toks:
            perturbed.append(toks)
            continue

        p = mask_rate * (1 + 2 * w)

        new_toks = []

        i = 0

        while i < len(toks):
            tok = toks[i]

            if informative(tok) and random.random() < p:
                new_toks.append(mask_tok.mask_token)
                i += 1

            else:
                new_toks.append(tok)
                i += 1

        perturbed.append(new_toks)

    return perturbed


# ============================================================
# NUCLEUS SAMPLING
# ============================================================


def sample_top_p(
    logits,
    top_p=TOP_P,
):

    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    cumulative = torch.cumsum(sorted_probs, dim=-1)

    keep = cumulative <= top_p
    keep[0] = True

    filtered = sorted_probs * keep

    filtered /= filtered.sum()

    sampled = torch.multinomial(filtered, 1)

    return sorted_idx[sampled].item()


# ============================================================
# INFILLING
# ============================================================


def fill_masks(token_lists):

    outputs = []

    for toks in token_lists:
        toks = toks.copy()

        attempts = 0

        while mask_tok.mask_token in toks and attempts < 128:
            ids = mask_tok.convert_tokens_to_ids(toks)

            x = torch.tensor([ids], device=DEVICE)

            with torch.no_grad():
                logits = mask_model(x).logits[0]

            positions = [i for i, t in enumerate(toks) if t == mask_tok.mask_token]

            pos = random.choice(positions)

            sampled = sample_top_p(logits[pos])

            token = mask_tok.convert_ids_to_tokens(sampled)

            # avoid special tokens
            if token.strip() == "" or token in mask_tok.all_special_tokens:
                attempts += 1
                continue

            toks[pos] = token
            attempts += 1

        ids = mask_tok.convert_tokens_to_ids(toks)

        text = mask_tok.decode(ids, skip_special_tokens=True)

        outputs.append(text)

    return outputs


# ============================================================
# SCORE
# ============================================================


def compute_score(code):

    lines = code.splitlines()

    losses = line_losses(lines)

    ppl = np.mean(losses)

    std_ppl = np.std(losses)

    alpha, beta, gamma = 1.0, 0.5, 0.25

    burstiness = np.max(losses) / (ppl + 1e-8)

    return alpha * ppl + beta * std_ppl + gamma * burstiness


def detect(
    code,
    samples=SAMPLES,
):

    lines = code.splitlines()

    line_ppl = line_losses(lines)

    weights = normalize(line_ppl)

    original_score = compute_score(code)

    perturbed_scores = []

    for _ in range(samples):
        masked = perturb_code(
            lines,
            weights,
        )

        filled = fill_masks(masked)

        perturbed_code = "\n".join(filled)

        s = compute_score(perturbed_code)

        if not np.isfinite(s):
            continue

        perturbed_scores.append(s)

    perturbed_scores = np.array(perturbed_scores)

    if len(perturbed_scores) == 0:
        return 0.0

    probability = np.mean(perturbed_scores > original_score)

    return probability > THRESHOLD
