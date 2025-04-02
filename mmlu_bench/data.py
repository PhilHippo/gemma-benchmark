"""Data loading and processing utilities for MMLU benchmarking."""

from typing import Dict, List, Optional
from datasets import load_dataset, get_dataset_split_names
import random
import torch
import numpy as np

def load_benchmark_data(subset_name: str, config: Dict) -> Dict:
    """Loads and processes MMLU dataset subsets."""
    dataset_provider = config['dataset']['provider']
    split = config['dataset']['split']
    n_samples = config['dataset']['sample_size']
    seed = config['dataset'].get('random_seed', None)

    if seed:
        random.seed(seed)
        np.random.seed(seed)

    try:
        available_splits = get_dataset_split_names(dataset_provider, subset_name)
        if split not in available_splits:
            split = 'test' if 'test' in available_splits else available_splits[0]

        dataset = load_dataset(
            dataset_provider, 
            subset_name, 
            split=split, 
            trust_remote_code=True
        )

        if n_samples == -1 or n_samples is None or n_samples >= len(dataset):
            return dataset.shuffle(seed=seed) if seed else dataset
        
        return dataset.shuffle(seed=seed).select(range(n_samples))

    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_provider}/{subset_name}: {e}")

def get_expected_answer_letter(sample: Dict) -> Optional[str]:
    """Extracts the expected answer letter from an MMLU sample."""
    options = ['A', 'B', 'C', 'D']
    try:
        return options[sample['answer']]
    except (IndexError, KeyError):
        return None 