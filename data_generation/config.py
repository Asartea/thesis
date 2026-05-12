from pathlib import Path

import torch

MODEL = "mistralai/Codestral-22B-v0.1"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
MAX_NEW_TOKENS = 8192

TEST_MODE = False
TEST_SAMPLE_SIZE = 10

YEARS = [2021, 2024]
DAYS = range(1, 25)

CODE_VARIANTS = [
    "Write a highly optimized solution focusing on performance.",
    "Use a functional programming style where possible.",
    "Avoid using advanced libraries; rely on basic Python constructs.",
    "Use concise code, minimizing line count.",
]

STYLE_VARIANTS = [
    "Prefer short variable names.",
    "Prefer descriptive variable names.",
    "Use helper functions.",
    "Avoid helper functions.",
    "Favor list comprehensions.",
    "Avoid comprehensions.",
    "Use recursion where reasonable.",
    "Prefer iterative solutions.",
]

SYSTEM_PROMPT = """
You are a code synthesis engine.

Output constraints:
- Output ONLY Python source code.
- The output must define functions/classes required to solve the problem.
- Do not include any executable code that demonstrates usage.
- Do not include sample inputs, sample outputs, or test data.
- Do not include a main function or __main__ guard.
- Do not include hardcoded example arrays or placeholder datasets.
- Do not include comments of any kind.

Violation examples (forbidden):
- input =
- example =
- sample =
- test =
"""

OUTPUT_PATH = Path("output.jsonl")
INPUT_DIR = Path("data_generation") / "data"
