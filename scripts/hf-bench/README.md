# Hugging Face Benchmark

Benchmarks OpenVINO inference performance on Hugging Face models and exports execution graphs.

## Installation

```bash
pip install optimum[openvino]
```

## Usage

```bash
python hf-bench.py [options]
```

### Options
- `--text-file TEXT`: Input text file (default: `input.txt`)
- `--models-file MODELS`: Models list file (default: `models.txt`)  
- `--output CSV`: Output CSV file (default: `results.csv`)

### Example
```bash
python hf-bench.py --text-file input.txt --models-file models.txt --output results.csv
```

## Input Files

### Text File Format
```
This is sample text for inference.
Another line of text to benchmark.
```

### Models File Format (huggingface)
```
bert-base-uncased
distilbert-base-uncased
microsoft/DialoGPT-medium
```

## Output

- **CSV Results**: Contains `model_id`, `task`, `inference_time_ms`
- **Execution Graphs**: `<model-name>.xml` files for each model
- **Model Cache**: Saved in `models/` directory
