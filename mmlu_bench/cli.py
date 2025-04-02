"""Command-line interface for running benchmarks."""

import fire
import traceback
from typing import Optional
from pathlib import Path

from .utils import load_config, setup_output_directory
from .core import run_benchmark_pipeline

def main(
    config_path: str = "config.yaml",
    subset: Optional[str] = None
) -> None:
    """Main entry point for running benchmarks.
    
    Args:
        config_path: Path to the configuration YAML file
        subset: Optional specific subset to run (overrides config subsets)
    """
    try:
        # Load configuration
        config = load_config(config_path)
        print("Configuration loaded successfully.")

        # Create output directory
        run_dir = setup_output_directory(config)
        print(f"Results will be saved in: {run_dir}")

        # Run the benchmark pipeline
        run_benchmark_pipeline(config, run_dir, subset)

    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

def cli():
    """Command-line interface entry point."""
    fire.Fire(main)

if __name__ == "__main__":
    cli() 