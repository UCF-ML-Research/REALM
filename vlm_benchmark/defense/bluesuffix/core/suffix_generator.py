"""BlueSuffix Suffix Generator — GPT-2 LoRA inference.

Loads a fine-tuned GPT-2 + LoRA adapter and generates a short defensive
suffix (~10 tokens) that is appended to the prompt to steer the VLM
toward safe responses.

Extracted from legacy/run_bluesuffix.py lines 158-188.
"""

import torch


def load_suffix_generator(suffix_dir, device="cpu"):
    """Load the fine-tuned GPT-2 LoRA suffix generator.

    Args:
        suffix_dir: Path to directory containing LoRA adapter + tokenizer
        device: torch device string

    Returns:
        (model, tokenizer) tuple
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(suffix_dir)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model = PeftModel.from_pretrained(base_model, suffix_dir).to(device)
    model.eval()
    return model, tokenizer


def generate_suffix(model, tokenizer, prompt, device="cpu"):
    """Generate a defensive suffix for the given prompt.

    Args:
        model: GPT-2 + LoRA model
        tokenizer: Matching tokenizer
        prompt: Input prompt to generate suffix for
        device: torch device string

    Returns:
        Generated suffix string (~10 tokens)
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            min_new_tokens=10,
            max_new_tokens=10,
            top_k=0,
            top_p=0.92,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Only decode the newly generated tokens
    suffix_ids = output_ids[0, input_ids.shape[1]:]
    suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True).strip()
    return suffix
