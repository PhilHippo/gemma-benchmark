"""Utility functions for configuration and setup."""

from typing import Dict, Tuple
import yaml
from pathlib import Path
import torch
import pandas as pd
from datetime import datetime

def load_config(config_path: str = "config.yaml") -> Dict:
    """Loads configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_device_settings(config: Dict) -> Tuple[torch.device, torch.dtype]:
    """Determines device and dtype from config."""
    device = torch.device("cuda" if torch.cuda.is_available() and 
                         config['hardware'].get('use_gpu', True) else "cpu")
    
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    dtype = dtype_map.get(config['hardware'].get('dtype', 'float16'), torch.float16)
    
    return device, dtype

def setup_output_directory(config: Dict) -> Path:
    """Create output directory structure."""
    # Make sure we have the output configuration
    if 'output' not in config or 'results_dir' not in config['output']:
        raise ValueError("Missing 'output.results_dir' in configuration")
    
    # Create base directory
    base_dir = Path(config['output']['results_dir'])
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(exist_ok=True)
    
    # Pre-create directories for all subsets
    for subset in config['dataset']['subsets']:
        subset_dir = run_dir / subset
        subset_dir.mkdir(exist_ok=True, parents=True)
    
    return run_dir  # Make sure we're returning the Path object

def save_results(results_df: pd.DataFrame, config: Dict, run_dir: Path) -> None:
    """Saves results in specified formats."""
    for format in config['output'].get('save_format', ['csv']):
        if format == 'csv':
            results_df.to_csv(run_dir / 'results.csv', index=False)
        elif format == 'json':
            results_df.to_json(run_dir / 'results.json', orient='records')
        elif format == 'pickle':
            results_df.to_pickle(run_dir / 'results.pkl') 