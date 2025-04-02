"""Evaluation metrics and result processing."""

from typing import Dict, Any
import traceback
from .utils import get_device_settings
import pandas as pd
from tqdm import tqdm
import torch
from .generation import load_model_and_tokenizer, generate_response
from .data import get_expected_answer_letter
from .prompting import format_mmlu_prompt

def evaluate_mmlu_response(
    extracted_answer: str,
    expected_letter: str,
    config: Dict
) -> int:
    """Evaluates if the extracted answer matches the expected letter."""
    if not extracted_answer or not expected_letter:
        return 0
    
    return 1 if extracted_answer.strip().upper() == expected_letter else 0

def run_benchmark(
    dataset: Any,
    config: Dict,
    subset_name: str
) -> pd.DataFrame:
    """Runs the benchmark for configured models on a dataset subset."""
    # Load models based on config
    models_to_load = config['models']
    loaded_models = {}
    for key, model_config in models_to_load.items():
        model, tokenizer = load_model_and_tokenizer(model_config['name'], config)
        loaded_models[model_config['display_name']] = {"model": model, "tokenizer": tokenizer}

    results = []
    device, _ = get_device_settings(config)

    print(f"\n--- Starting Benchmark: {subset_name} ({len(dataset)} samples) ---")
    print(f"Using {config['prompt'].get('num_shots', 0)}-shot prompting")

    # Create progress bars dictionary for each model
    pbars = {
        display_name: tqdm(
            total=len(dataset),
            desc=f"{display_name}",
            position=idx,
            leave=False
        )
        for idx, display_name in enumerate(loaded_models.keys())
    }

    # Loop through each sample in the loaded data
    for i, sample in enumerate(dataset):
        # Format prompt based on dataset type
        if config['dataset']['name'].lower() == 'mmlu':
            prompt = format_mmlu_prompt(sample, config, dataset, i)
            expected_letter = get_expected_answer_letter(sample)
        else:
            print(f"Warning: Prompt formatting not implemented for dataset '{config['dataset']['name']}'")
            prompt = str(sample)
            expected_letter = None

        # Generate and evaluate for each model
        for model_display_name, model_data in loaded_models.items():
            model = model_data["model"]
            tokenizer = model_data["tokenizer"]

            try:
                raw_generated, extracted_answer, inference_time = generate_response(
                    model,
                    tokenizer,
                    prompt,
                    config['generation']
                )

                # Evaluate based on dataset type
                if config['dataset']['name'].lower() == 'mmlu':
                    top1_score = evaluate_mmlu_response(extracted_answer, expected_letter, config)
                else:
                    top1_score = None

                # Update progress bar
                pbar = pbars[model_display_name]
                pbar.update(1)
                pbar.set_description(f"{model_display_name} ({i+1}/{len(dataset)})")

                # Store results
                results.append({
                    "model_name": model_display_name,
                    "dataset_name": config['dataset']['name'],
                    "dataset_subset": subset_name,
                    "prompt_truncated": prompt[:100] + "...",
                    "expected_answer": expected_letter,
                    "raw_generated_text": raw_generated,
                    "extracted_answer": extracted_answer,
                    "inference_time_s": inference_time,
                    "top1_accuracy": top1_score,
                    "num_shots": config['prompt'].get('num_shots', 0)
                })

            except Exception as e:
                print(f"\nError during generation/evaluation for model {model_display_name}, sample {i}: {e}")
                traceback.print_exc()
                results.append({
                    "model_name": model_display_name,
                    "dataset_name": config['dataset']['name'],
                    "dataset_subset": subset_name,
                    "prompt_truncated": prompt[:100] + "...",
                    "expected_answer": expected_letter,
                    "error": str(e),
                    "inference_time_s": 0,
                    "top1_accuracy": 0,
                    "num_shots": config['prompt'].get('num_shots', 0)
                })
                pbars[model_display_name].update(1)

            if device.type == "cuda":
                del raw_generated, extracted_answer
                torch.cuda.empty_cache()

    # Close progress bars
    for pbar in pbars.values():
        pbar.close()

    print(f"\n--- Benchmark Complete: {subset_name} ---")
    return pd.DataFrame(results) 