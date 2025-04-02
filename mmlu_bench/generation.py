"""Model loading and inference utilities."""

import re
import time
from typing import Dict, Tuple, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .utils import get_device_settings

def load_model_and_tokenizer(model_name: str, config: Dict) -> Tuple[Any, Any]:
    """Loads model and tokenizer with specified configuration."""
    device, dtype = get_device_settings(config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map='auto',
        low_cpu_mem_usage=config['hardware'].get('low_cpu_mem_usage', False)
    )
    model.eval()
    
    return model, tokenizer

def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    generation_config: Dict
) -> Tuple[str, str, float]:
    """Generates model response and extracts the answer."""
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=False,
        truncation=True
    ).to(model.device)

    gen_config = GenerationConfig(
        max_new_tokens=generation_config.get('max_new_tokens', 5),
        do_sample=generation_config.get('do_sample', False),
        temperature=generation_config.get('temperature', 1.0),
        num_beams=generation_config.get('num_beams', 1),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(**inputs, generation_config=gen_config)
        inference_time = time.time() - start_time

    generated_text = tokenizer.decode(
        outputs[0, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()

    extracted_answer = extract_answer(generated_text)
    
    return generated_text, extracted_answer, inference_time

def extract_answer(text: str) -> str:
    """Extracts the answer letter from generated text."""
    patterns = [
        r'^([A-D])',
        r'(?:THE )?(?:ANSWER (?:IS )?)?([A-D])',
        r'([A-D])\.'
    ]
    
    for pattern in patterns:
        if match := re.search(pattern, text.upper()):
            return match.group(1)
    
    return text.split()[0].upper() if text else "" 