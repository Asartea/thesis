import random
import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODELS
# ============================================================

SCORING_MODEL = "Salesforce/codegen-350M-mono"
MASK_MODEL = "microsoft/codebert-base-mlm"

score_tok = AutoTokenizer.from_pretrained(SCORING_MODEL)
score_model = AutoModelForCausalLM.from_pretrained(
    SCORING_MODEL
).eval().to(DEVICE)

mask_tok = AutoTokenizer.from_pretrained(MASK_MODEL)
mask_model = AutoModelForMaskedLM.from_pretrained(
    MASK_MODEL
).eval().to(DEVICE)

MASK_ID = mask_tok.mask_token_id

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# LINE-LEVEL PPL
# ============================================================

def line_nll(lines):
    """
    True autoregressive NLL using causal LM.
    """

    vals = []

    for line in lines:
        if not line.strip():
            vals.append(0.0)
            continue

        enc = score_tok(
            line,
            return_tensors="pt",
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            out = score_model(
                **enc,
                labels=enc["input_ids"]
            )

        vals.append(out.loss.item())

    return np.array(vals)

# ============================================================
# MASK WEIGHTS
# ============================================================

def normalize(x):
    x = np.array(x)

    if np.allclose(x.max(), x.min()):
        return np.ones_like(x) * 0.5

    return (x - x.min()) / (x.max() - x.min())

# ============================================================
# TOKEN FILTERING
# ============================================================

LOW_INFO = {
    "{", "}", "(", ")", "[", "]",
    ":", ";", ",", ".", "#",
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

# ============================================================
# MASKING
# ============================================================

def perturb_code(lines, weights, mask_rate=0.15):

    out = []

    for line, w in zip(lines, weights):

        toks = mask_tok.tokenize(line)

        new = []

        p = mask_rate * (0.5 + w)

        for t in toks:

            if informative(t) and random.random() < p:
                new.append(mask_tok.mask_token)
            else:
                new.append(t)

        out.append(new)

    return out

# ============================================================
# NUCLEUS SAMPLING
# ============================================================

def sample_top_p(logits, top_p=0.95):

    probs = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(
        probs,
        descending=True
    )

    cumulative = torch.cumsum(sorted_probs, dim=-1)

    keep = cumulative <= top_p
    keep[0] = True

    filtered_probs = sorted_probs * keep
    filtered_probs /= filtered_probs.sum()

    sampled_idx = torch.multinomial(
        filtered_probs,
        1
    )

    return sorted_idx[sampled_idx].item()

# ============================================================
# ITERATIVE INFILLING
# ============================================================

def fill_masks(token_lists):

    outputs = []

    for toks in token_lists:

        toks = toks.copy()

        while mask_tok.mask_token in toks:

            ids = mask_tok.convert_tokens_to_ids(toks)

            x = torch.tensor(
                [ids],
                device=DEVICE
            )

            with torch.no_grad():
                logits = mask_model(x).logits[0]

            mask_positions = [
                i for i, t in enumerate(toks)
                if t == mask_tok.mask_token
            ]

            pos = random.choice(mask_positions)

            sampled = sample_top_p(
                logits[pos]
            )

            toks[pos] = mask_tok.convert_ids_to_tokens(
                sampled
            )

        ids = mask_tok.convert_tokens_to_ids(toks)

        text = mask_tok.decode(
            ids,
            skip_special_tokens=True
        )

        outputs.append(text)

    return outputs

# ============================================================
# SCORING
# ============================================================

def compute_score(line_losses):

    mu = np.mean(line_losses)

    sigma = np.std(line_losses)

    burstiness = (
        np.max(line_losses)
        / (mu + 1e-8)
    )

    return (
        1.0 * mu
        + 0.75 * sigma
        + 0.5 * burstiness
    )

# ============================================================
# DETECTOR
# ============================================================

def detect(
    code,
    samples=50,
    mask_rate=0.15,
    threshold=0.97,
):

    lines = code.splitlines()

    original_losses = line_nll(lines)

    original_score = compute_score(
        original_losses
    )

    weights = normalize(original_losses)

    perturbed_scores = []

    for _ in range(samples):

        masked = perturb_code(
            lines,
            weights,
            mask_rate=mask_rate,
        )

        filled = fill_masks(masked)

        losses = line_nll(filled)

        s = compute_score(losses)

        perturbed_scores.append(s)

    perturbed_scores = np.array(
        perturbed_scores
    )

    p = np.mean(
        perturbed_scores > original_score
    )

    outcome = "machine" if p > threshold else "human"

    return {
        "outcome": outcome,
        "probability": float(p),
        "original_score": float(original_score),
        "perturbed_mean": float(
            perturbed_scores.mean()
        ),
    }
