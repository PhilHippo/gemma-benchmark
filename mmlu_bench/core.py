"""Core benchmarking functionality."""

import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .data import load_benchmark_data
from .evaluation import run_benchmark
from .utils import setup_output_directory, save_results
from .reporting import generate_report

def run_benchmark_pipeline(
    config: dict,
    run_dir: Path,
    subset: Optional[str] = None
) -> None:
    """Core benchmark pipeline that handles the entire benchmarking process.
    
    Args:
        config: Configuration dictionary
        run_dir: Output directory for results
        subset: Optional specific subset to run
    """
    # Set random seed if specified
    if config['dataset'].get('random_seed') is not None:
        seed = config['dataset']['random_seed']
        print(f"Setting random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Override subset if specified
    if subset:
        print(f"Overriding config subsets with: {subset}")
        config['dataset']['subsets'] = [subset]

    # Process each dataset subset
    results = []
    for subset_name in config['dataset']['subsets']:
        print(f"\n=== Processing Dataset Subset: {subset_name} ===")
        
        try:
            dataset = load_benchmark_data(subset_name, config)
            result_df = run_benchmark(dataset, config, subset_name)
            
            if not result_df.empty:
                results.append(result_df)
                
                if config['output'].get('save_results', False):
                    subset_dir = run_dir / subset_name
                    save_results(result_df, config, subset_dir)
                    print(f"Intermediate results for {subset_name} saved to {subset_dir}")
            else:
                print(f"No results generated for subset {subset_name}")
                
        except Exception as e:
            print(f"Error processing subset {subset_name}: {e}")
            continue

    # Process final results
    if not results:
        print("\nNo results were generated across all subsets.")
        return

    final_results = pd.concat(results, ignore_index=True)

    # Save combined results
    if config['output'].get('save_results', False):
        save_results(final_results, config, run_dir)
        print(f"Combined results saved to {run_dir}")

    # Generate final report
    generate_report(final_results, config, run_dir) 