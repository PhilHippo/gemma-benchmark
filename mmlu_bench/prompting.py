"""Prompt formatting utilities."""

from typing import Dict, Any
import random
from .data import get_expected_answer_letter

def format_mmlu_prompt(sample: Dict, config: Dict, dataset: Any = None, current_index: int = None) -> str:
    """Formats a prompt for the MMLU task with optional few-shot examples."""
    num_shots = config['prompt'].get('num_shots', 0)
    include_instructions = config['prompt'].get('include_instructions', True)
    
    prompt = ""
    if include_instructions:
        prompt = "You are a helpful AI assistant. Answer the following multiple choice question by selecting the correct option (A, B, C, or D).\n\n"
    
    if num_shots > 0 and dataset is not None and current_index is not None:
        # Get random examples from the dataset, excluding the current sample
        available_indices = list(range(len(dataset)))
        if current_index in available_indices:
            available_indices.remove(current_index)
        
        if len(available_indices) >= num_shots:
            example_indices = random.sample(available_indices, num_shots)
            
            for idx in example_indices:
                example = dataset[idx]
                example_question = example['question']
                example_choices = example['choices']
                example_answer = get_expected_answer_letter(example)
                
                prompt += f"Question: {example_question}\nChoices:\n"
                options = ['A', 'B', 'C', 'D']
                for i, choice in enumerate(example_choices):
                    if i < len(options):
                        prompt += f"{options[i]}. {choice}\n"
                prompt += f"Answer: {example_answer}\n\n"
    
    question = sample['question']
    choices = sample['choices']
    prompt += f"Question: {question}\nChoices:\n"
    options = ['A', 'B', 'C', 'D']
    for i, choice in enumerate(choices):
        if i < len(options):
            prompt += f"{options[i]}. {choice}\n"
    prompt += "Answer:"
    
    return prompt 