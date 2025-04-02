"""Results reporting and visualization."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path

def generate_report(results_df: pd.DataFrame, config: Dict, run_dir: Path) -> None:
    """Generates and prints a summary report and comparison charts."""
    if results_df.empty:
        print("No results to report.")
        return

    print("\n--- Benchmark Results Summary ---")

    dataset_name = config['dataset']['name']
    subsets_run = results_df['dataset_subset'].unique()

    valid_results = results_df.dropna(subset=['top1_accuracy'])
    if valid_results.empty:
        print("No valid results with scores found for summary.")
        return

    summary = valid_results.groupby("model_name").agg(
        avg_inference_time_s=('inference_time_s', 'mean'),
        top1_accuracy=('top1_accuracy', 'mean'),
        std_dev_time=('inference_time_s', 'std'),
        n_samples=('top1_accuracy', 'count')
    ).reset_index()

    print(f"\nAverage Performance Metrics per Model across {len(subsets_run)} subset(s) ({dataset_name}):")
    summary['top1_accuracy'] = summary['top1_accuracy'] * 100
    print(summary.to_string(index=False, float_format="%.2f", 
                          formatters={'top1_accuracy':'{:.2f}%'.format, 
                                    'n_samples':'{:.0f}'.format}))

    if config['output'].get('plot_results', False):
        plot_results(summary, subsets_run, dataset_name, run_dir)

def plot_results(summary: pd.DataFrame, subsets_run: list, dataset_name: str, run_dir: Path) -> None:
    """Generates and saves visualization of benchmark results."""
    model_names = summary["model_name"]
    accuracies = summary["top1_accuracy"]

    plt.figure(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
    bars = plt.bar(model_names, accuracies, color=colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, 
                f'{yval:.1f}%', va='bottom', ha='center')

    plt.xlabel("Model")
    plt.ylabel("Average Accuracy (%)")
    plt.title(f"Model Accuracy on {dataset_name} ({', '.join(subsets_run)})")
    plt.ylim(0, 105)
    plt.xticks(rotation=15, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = run_dir / "accuracy_comparison.png"
    plt.savefig(plot_path)
    print(f"\nComparison chart saved to: {plot_path}")
    plt.close() 