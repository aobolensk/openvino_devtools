#!/usr/bin/env python3
import argparse
import csv
import logging as py_logging
import time
from pathlib import Path

from huggingface_hub import HfApi
from optimum.intel.openvino import *  # noqa: F403,F401
from transformers import AutoTokenizer


def read_files(text_file, model_file):
    with open(text_file, "r") as f:
        text_lines = f.readlines()
    with open(model_file, "r") as f:
        models = f.readlines()
    models = [model.strip() for model in models]
    return text_lines, models


def save_results(result_file, results):
    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "task", "inference_time_ms"])
        for result in results:
            writer.writerow(result)


def go_benchmark(model_id, text_lines, device_name="CPU"):
    py_logging.disable(py_logging.INFO)

    info = HfApi().model_info(model_id)
    task = info.pipeline_tag
    MODEL_DIR = f"models/{model_id}"
    print(model_id, task)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model_class_str = info.transformersInfo["auto_model"]
    file_path = Path(MODEL_DIR) / "openvino_model.xml"
    model_class_str = model_class_str.replace("Auto", "OV")

    # Save OpenVINO model if it has not been saved before
    if not file_path.is_file():
        model_class = eval(model_class_str)
        print(f"Saving IR model {model_id}, {model_class_str}")
        model = model_class.from_pretrained(model_id, export=True)
        model.save_pretrained(MODEL_DIR)

    model_class = eval(model_class_str)
    model = model_class.from_pretrained(MODEL_DIR)

    if device_name != "CPU":
        model.to(device_name)

    # warmup inference
    warmup_text = [f"hello world! {tokenizer.mask_token}"]
    warmup_tokens = tokenizer.encode_plus(*warmup_text, return_tensors="pt")
    model(**warmup_tokens)

    # Time inference on input sentence
    total_time = 0
    for line in text_lines[:1]:
        tokens = tokenizer.encode_plus(line, return_tensors="pt")
        start = time.perf_counter()
        model(**tokens)
        end = time.perf_counter()
        total_time += (end - start)

    print(f"Time: {total_time * 1000:.2f} ms")

    return [model_id, task, f"{total_time * 1000:.2f}"]


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenVINO inference on Hugging Face models")
    parser.add_argument("--text-file", default="input.txt", help="Input text file (default: input.txt)")
    parser.add_argument("--models-file", default="models.txt", help="Models list file (default: models.txt)")
    parser.add_argument("--output", default="results.csv", help="Output CSV file (default: results.csv)")
    parser.add_argument("--device", default="CPU", help="Device for inference (default: CPU)")
    args = parser.parse_args()

    text_lines, model_list = read_files(args.text_file, args.models_file)
    results = []
    for model in model_list:
        result = go_benchmark(model, text_lines, args.device)
        results.append(result)

    save_results(args.output, results)


if __name__ == "__main__":
    main()
