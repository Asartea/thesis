import random
import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ============================================================
# CONFIG
# ============================================================

SCORING_MODEL = "Salesforce/codegen-2B-mono"
MASK_MODEL = "microsoft/codebert-base-mlm"

MASK_RATE = 0.05
TOP_P = 0.3
SAMPLES = 30

SEED = 42

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# LOAD MODELS
# ============================================================

score_tok = AutoTokenizer.from_pretrained(
    SCORING_MODEL
)

score_model = AutoModelForCausalLM.from_pretrained(
    SCORING_MODEL,
    torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
).eval().to(DEVICE)

mask_tok = AutoTokenizer.from_pretrained(
    MASK_MODEL
)

mask_model = AutoModelForMaskedLM.from_pretrained(
    MASK_MODEL
).eval().to(DEVICE)

MASK_ID = mask_tok.mask_token_id

# ============================================================
# TOKEN FILTERING
# ============================================================

LOW_INFO = {
    "{", "}", "(", ")", "[", "]",
    ":", ";", ",", ".", "#",
}

KEYWORDS = {
    "if", "for", "while", "return",
    "class", "def", "import", "from",
    "try", "except", "finally",
    "with", "lambda",
}

def informative(tok):

    t = tok.strip()

    if not t:
        return False

    if t in LOW_INFO:
        return False

    if t in KEYWORDS:
        return False

    if len(t) <= 2:
        return False

    if all(c in "_-=+*/<>!&|" for c in t):
        return False

    return True

# ============================================================
# NORMALIZATION
# ============================================================

def normalize(x):

    x = np.array(x)

    if np.allclose(x.max(), x.min()):
        return np.ones_like(x) * 0.5

    return (
        (x - x.min())
        / (x.max() - x.min())
    )

# ============================================================
# FULL-SNIPPET NLL
# ============================================================

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

        out = score_model(
            **enc,
            labels=enc["input_ids"]
        )

    return out.loss.item()

# ============================================================
# LINE WEIGHTS
# ============================================================

def line_losses(lines):

    vals = []

    prefix = ""

    for line in lines:

        current = prefix + line

        if not line.strip():
            vals.append(0.0)
            prefix += line + "\n"
            continue

        vals.append(
            code_nll(current)
        )

        prefix += line + "\n"

    return np.array(vals)

# ============================================================
# MASKING
# ============================================================

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

        p = mask_rate * (0.5 + w)

        new_toks = []

        i = 0

        while i < len(toks):

            tok = toks[i]

            if (
                informative(tok)
                and random.random() < p
            ):

                span_len = random.choice([1, 1, 2])

                for _ in range(span_len):

                    if i < len(toks):
                        new_toks.append(
                            mask_tok.mask_token
                        )
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

    probs = F.softmax(
        logits,
        dim=-1
    )

    sorted_probs, sorted_idx = torch.sort(
        probs,
        descending=True
    )

    cumulative = torch.cumsum(
        sorted_probs,
        dim=-1
    )

    keep = cumulative <= top_p
    keep[0] = True

    filtered = sorted_probs * keep

    filtered /= filtered.sum()

    sampled = torch.multinomial(
        filtered,
        1
    )

    return sorted_idx[sampled].item()

# ============================================================
# INFILLING
# ============================================================

def fill_masks(token_lists):

    outputs = []

    for toks in token_lists:

        toks = toks.copy()

        attempts = 0

        while (
            mask_tok.mask_token in toks
            and attempts < 128
        ):

            ids = mask_tok.convert_tokens_to_ids(
                toks
            )

            x = torch.tensor(
                [ids],
                device=DEVICE
            )

            with torch.no_grad():

                logits = mask_model(
                    x
                ).logits[0]

            positions = [
                i for i, t in enumerate(toks)
                if t == mask_tok.mask_token
            ]

            pos = random.choice(positions)

            sampled = sample_top_p(
                logits[pos]
            )

            token = mask_tok.convert_ids_to_tokens(
                sampled
            )

            # avoid special tokens
            if token.startswith("["):
                attempts += 1
                continue

            toks[pos] = token
            attempts += 1

        ids = mask_tok.convert_tokens_to_ids(
            toks
        )

        text = mask_tok.decode(
            ids,
            skip_special_tokens=True
        )

        outputs.append(text)

    return outputs

# ============================================================
# SCORE
# ============================================================

def compute_score(code):

    lines = code.splitlines()

    losses = line_losses(lines)

    mu = np.mean(losses)

    sigma = np.std(losses)

    burstiness = (
        np.max(losses)
        / (mu + 1e-8)
    )

    return (
        1.0 * mu
        + 0.5 * sigma
        + 0.25 * burstiness
    )

# ============================================================
# DETECTOR
# ============================================================

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

        perturbed_code = "\n".join(
            filled
        )

        s = compute_score(
            perturbed_code
        )

        perturbed_scores.append(s)

    perturbed_scores = np.array(
        perturbed_scores
    )

    delta = (
        perturbed_scores.mean()
        - original_score
    )

    z = (
        delta
        / (perturbed_scores.std() + 1e-8)
    )

    # empirical heuristic
    probability = float(
        1 / (1 + np.exp(-z))
    )

    outcome = (
        "machine"
        if z > 1.0
        else "human"
    )

    return {
        "outcome": outcome,
        "probability": probability,
        "z_score": float(z),
        "delta": float(delta),
        "original_score": float(original_score),
        "perturbed_mean": float(
            perturbed_scores.mean()
        ),
        "perturbed_std": float(
            perturbed_scores.std()
        ),
    }
