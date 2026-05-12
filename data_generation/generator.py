import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_generation.config import MODEL

tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
)


@torch.inference_mode()
def generate_batch(
    prompts: list[str],
    max_new_tokens: int = 512,
    seed: int | None = None,
) -> list[str]:
    if seed is not None:
        torch.manual_seed(seed)

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        truncation=True,
        return_attention_mask=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.05,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_lengths = inputs["attention_mask"].sum(dim=1)

    texts = [
        tokenizer.decode(
            output[prompt_length:],
            skip_special_tokens=True,
        )
        for output, prompt_length in zip(
            outputs,
            prompt_lengths,
            strict=True,
        )
    ]

    del outputs
    del inputs

    torch.cuda.empty_cache()

    return texts
