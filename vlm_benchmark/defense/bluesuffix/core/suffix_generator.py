"""BlueSuffix suffix generator using GPT-2 LoRA inference."""

import torch


def load_suffix_generator(suffix_dir, device="cpu"):
    """Load the fine-tuned GPT-2 LoRA suffix generator, returning (model, tokenizer)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(suffix_dir)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model = PeftModel.from_pretrained(base_model, suffix_dir).to(device)
    model.eval()
    return model, tokenizer


def generate_suffix(model, tokenizer, prompt, device="cpu"):
    """Generate a defensive suffix for the given prompt."""
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
    suffix_ids = output_ids[0, input_ids.shape[1]:]
    suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True).strip()
    return suffix
