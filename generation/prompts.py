from transformers import AutoTokenizer


def build_prompt(problem: str, code_variant: str, style_variant: str) -> str:
    return f"""
    You are solving a programming problem.

    OUTPUT FORMAT:
    - Output ONLY Python code
    - No markdown
    - No explanations

    BEGIN PROBLEM
    {problem}
    END PROBLEM

    BEGIN INSTRUCTIONS
    {code_variant}
    {style_variant}
    END INSTRUCTIONS

    Now write the solution.
    """.strip()


def render_chat_prompt(
    tokenizer: AutoTokenizer, system_prompt: str, user_prompt: str
) -> str:
    """Render a chat prompt for a given system and user prompt."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return rendered_prompt
