# Hugging Face Benchmark Scripts

OpenVINO benchmarking tools for Hugging Face models.

## Installation

It is recommended to use separate venv for running the benchmarks

```bash
python3 -m pip install -r requirements.txt
```

## Scripts

### NLP Benchmarking (`hf-bench.py`)
Benchmark inference performance on NLP models.

```bash
python hf-bench.py --text-file input.txt --models-file models.txt --output results.csv
```

### Image Generation (`text2image-bench.py`)
Generate images using Stable Diffusion models.

```bash
python text2image-bench.py --text-file input.txt --models-file sd-models.txt
```

## Options
See `--help`
