# LLM Benchmarking Suite (Focused on Gemma Models)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Project Status:** Initial Proof-of-Concept for GSoC 2025 @ Google DeepMind

## Overview

This repository contains the foundational code for a comprehensive benchmarking suite designed to evaluate Large Language Models (LLMs), with a primary focus on Google's Gemma family. The goal of this project, aligned with the Google Summer of Code 2025 proposal for Google DeepMind, is to develop a robust, extensible, and reproducible framework for assessing model performance across diverse tasks and datasets.

Evaluating LLMs rigorously is crucial for understanding their capabilities and limitations. This suite aims to provide standardized tools for researchers and developers to:
*   Benchmark various Gemma models (different sizes, instruction-tuned vs. base).
*   Compare Gemma models against other leading open models (like Llama, Mistral).
*   Automate the evaluation process across multiple academic and custom benchmarks.
*   Generate insightful reports and visualizations for easy comparison and analysis.

This initial implementation serves as a demonstration of the core architecture and capabilities envisioned for the full project.

## Current Features (Proof-of-Concept)

This initial version establishes the core structure and includes:

*   **Modular Framework:** Codebase organized into logical modules (`data`, `generation`, `evaluation`, `reporting`, `prompting`, `utils`, `core`, `cli`) for maintainability and extensibility.
*   **Configuration Driven:** Benchmarks are configured via a central `config.yaml` file, allowing easy modification of models, datasets, hardware settings, generation parameters, and output options.
*   **MMLU Benchmark Integration:** Includes specific support for running benchmarks on subsets of the MMLU dataset (`cais/mmlu`).
*   **Multi-Model Comparison:** Currently configured to compare Google's `gemma-1.1-2b-it` against `gpt2` (easily extensible via config).
*   **Few-Shot Prompting:** Supports configurable few-shot prompting (`num_shots` in config) with examples randomly selected from the dataset.
*   **Basic Evaluation:** Implements Top-1 accuracy calculation for MMLU-style multiple-choice tasks.
*   **Hardware Acceleration:** Supports GPU usage (via `device_map='auto'`) and configurable data types (`float16`, `bfloat16`, etc.).
*   **Result Reporting:**
    *   Console output summarizing performance metrics (accuracy, inference time).
    *   Saves detailed results per run in CSV and JSON formats.
    *   Generates a basic bar chart comparing model accuracy.
    *   Timestamped output directories for organized results.
*   **Command-Line Interface (CLI):** Provides a user-friendly CLI (`mmlu-bench`) using `python-fire` for running benchmarks with configuration overrides.
*   **Reproducibility:** Uses fixed random seeds (if specified in config) for data loading and sampling.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PhilHippo/gemma-benchmark
    cd llm-mmlu-bench
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the package in editable mode:** (This allows you to modify the code and run the CLI tool)
    ```bash
    pip install -e .
    ```
    *Optional: For development including testing and linting tools:*
    ```bash
    pip install -e ".[dev]"
    ```

## Usage

The benchmarking suite is run via the command-line interface:

```bash
mmlu-bench [OPTIONS]
```

**Common Options:**

*   `--config_path PATH`: Specify the path to the configuration file (default: `config.yaml`).
*   `--subset NAME`: Run the benchmark only on a specific dataset subset (e.g., `high_school_mathematics`), overriding the subsets listed in the config file.

**Examples:**

1.  **Run with default configuration (`config.yaml`):**
    ```bash
    mmlu-bench
    ```

2.  **Run using a custom configuration file:**
    ```bash
    mmlu-bench --config_path=configs/my_custom_config.yaml
    ```

3.  **Run only the 'high_school_physics' subset from the default config:**
    ```bash
    mmlu-bench --subset="high_school_physics"
    ```

Results, logs, and plots will be saved in a timestamped subdirectory within the `results_dir` specified in the configuration file (default: `benchmark_results/`).

## Configuration (`config.yaml`)

The `config.yaml` file controls all aspects of the benchmark run:

*   `models`: Define the models to benchmark (Hugging Face name, display name).
*   `hardware`: Specify GPU usage, data type (`float16`, `bfloat16`, `float32`), and memory options.
*   `dataset`: Configure the target dataset (name, provider, subsets, split, sample size, random seed).
*   `prompt`: Control few-shot settings (number of shots, instructions, format).
*   `generation`: Set parameters for the model's generation process (max new tokens, sampling, temperature, beams).
*   `output`: Define how results are saved (directory, formats, plotting, verbosity).

See the default `config.yaml` for detailed examples.

## Code Structure

The codebase is organized into the `mmlu_bench` package:

*   `cli.py`: Command-line interface definition using `fire`.
*   `core.py`: Orchestrates the main benchmarking pipeline.
*   `data.py`: Handles dataset loading and preprocessing.
*   `evaluation.py`: Contains evaluation logic (e.g., accuracy calculation) and the `run_benchmark` function for individual subsets.
*   `generation.py`: Manages model loading and text generation/inference.
*   `prompting.py`: Responsible for formatting prompts, including few-shot examples.
*   `reporting.py`: Generates console summaries, plots, and potentially more advanced reports.
*   `utils.py`: Contains helper functions for configuration, setup, and device management.

## Roadmap & Future Work (GSoC Project Goals)

This initial codebase provides a solid foundation. The full GSoC project aims to significantly expand upon this:

1.  **Expand Dataset Coverage:**
    *   Integrate standard academic benchmarks: GSM8K, Hellaswag, ARC, TruthfulQA, etc.
    *   Develop a clear interface for adding *custom datasets* easily.
2.  **Broaden Model Support:**
    *   Include various Gemma model sizes (e.g., 7B) and variants (base vs. IT).
    *   Add other key open models for comparison (e.g., Llama 2/3, Mistral variants, Phi).
    *   Refactor model loading to be more generic and easily extensible.
3.  **Enhance Evaluation Metrics:**
    *   Implement more sophisticated evaluation beyond simple accuracy (e.g., normalized accuracy for GSM8K, ROUGE/BLEU for summarization/translation tasks if added, execution-based evaluation for coding tasks).
    *   Develop more robust Top-K accuracy evaluation using model logits/probabilities.
4.  **Automate Benchmarking:**
    *   Create robust scripts for running sequences of benchmarks across multiple models and datasets automatically.
    *   Potentially integrate with CI/CD systems for regular benchmark runs.
5.  **Advanced Reporting & Visualization:**
    *   Generate comprehensive HTML reports or use tools like Streamlit/Gradio for interactive dashboards.
    *   Create detailed tables comparing models across multiple metrics and datasets.
    *   Implement leaderboards for easy ranking visualization.
    *   Include resource usage (time, memory) in reports.
6.  **Improve Reproducibility & Robustness:**
    *   Add comprehensive unit and integration tests.
    *   Refine error handling and logging.
    *   Document code thoroughly.
    *   Containerize the environment (e.g., Docker) for maximum reproducibility.
7.  **Regular Updates:** Ensure the framework design allows straightforward addition of new models, datasets, and evaluation methods as they emerge.

## Contributing

This is currently a proof-of-concept repository. For the GSoC project, contributions would follow standard open-source practices:
1.  Open an issue to discuss proposed changes.
2.  Fork the repository and create a feature branch.
3.  Implement changes, ensuring code quality (linting, testing).
4.  Submit a Pull Request with a clear description of the changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (Note: You'll need to add a LICENSE file, typically MIT or Apache 2.0 for open-source projects).