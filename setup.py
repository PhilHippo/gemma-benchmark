from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README.md
this_directory = Path(__file__).parent
try:
    # Attempt to read with utf-8, common for markdown
    long_description = (this_directory / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    # Fallback if README.md doesn't exist
    long_description = "A benchmarking tool for evaluating language models on the MMLU dataset"
except UnicodeDecodeError:
    # Fallback for different encoding if utf-8 fails
    try:
        long_description = (this_directory / "README.md").read_text(encoding="latin-1")
    except FileNotFoundError:
        long_description = "A benchmarking tool for evaluating language models on the MMLU dataset"


setup(
    name="llm-mmlu-bench",
    version="0.1.0",
    description="A benchmarking tool for evaluating language models, focusing on Gemma and MMLU",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Filippo Wang",
    author_email="wang.filippo02@gmail.com",
    url="https://github.com/yourusername/gemma-benchmark",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.6.0",
        "transformers>=4.50.3",
        "datasets>=3.5.0",
        "accelerate>=1.6.0",
        "evaluate>=0.4.3",
        "rouge_score>=0.1.2",
        "nltk>=3.9.1",
        "numpy>=2.2.4",
        "pandas>=2.2.3",
        "matplotlib>=3.10.1",
        "tqdm>=4.67.1",
        "PyYAML>=6.0.2",
        "fire>=0.7.0",
        "huggingface-hub>=0.30.1",
        "fsspec>=2024.12.0",
        "safetensors>=0.5.3",
        "tokenizers>=0.21.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mmlu-bench=mmlu_bench.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 