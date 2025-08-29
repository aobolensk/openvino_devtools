#!/usr/bin/env python3
import argparse
import csv
import os
import time
from pathlib import Path

from optimum.intel import OVStableDiffusionPipeline


def read_models_file(models_file):
    """Read models from a text file, one model per line."""
    if not os.path.exists(models_file):
        return []

    with open(models_file, "r") as f:
        models = [line.strip() for line in f.readlines() if line.strip()]
    return models


def read_prompts_file(prompts_file):
    """Read prompts from a text file, one prompt per line."""
    if not os.path.exists(prompts_file):
        return []

    with open(prompts_file, "r") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts


def save_results(result_file, results):
    """Save results to CSV file."""
    with open(result_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id", "prompt", "steps", "generation_time_ms", "output_file"])
        for result in results:
            writer.writerow([
                result["model"],
                result["prompt"],
                result["steps"],
                f"{result['generation_time'] * 1000:.2f}",
                result["output_file"]
            ])


def generate_images(model_id, prompts, device, num_inference_steps, output_dir, precision):
    """Generate images for given model and prompts."""
    print(f"Loading model: {model_id}")

    # Create model cache directory
    model_cache_dir = Path("models") / model_id.replace("/", "--")
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    # Configure precision
    PRECISION_MAP = {"FP32": "f32", "FP16": "f16", "BF16": "bf16"}
    ov_config = {}
    if precision is not None:
        ov_config["INFERENCE_PRECISION_HINT"] = PRECISION_MAP[precision]

    # Load or create pipeline
    start_load = time.perf_counter()
    pipe = OVStableDiffusionPipeline.from_pretrained(
        model_id,
        export=True,
        cache_dir=str(model_cache_dir),
        ov_config=ov_config
    )
    pipe.to(device)
    end_load = time.perf_counter()
    print(f"Model loaded in {end_load - start_load:.2f}s")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate images for each prompt
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Generating image {i + 1}/{len(prompts)}: {prompt[:50]}...")

        start_gen = time.perf_counter()
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        end_gen = time.perf_counter()

        # Save image with sanitized filename
        safe_model_name = model_id.replace("/", "_")
        safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{safe_model_name}_{i + 1}_{safe_prompt}.png"
        image_path = output_path / filename
        image.save(str(image_path))

        generation_time = end_gen - start_gen
        print(f"Generated in {generation_time:.2f}s: {image_path}")

        results.append({
            "model": model_id,
            "prompt": prompt,
            "steps": num_inference_steps,
            "generation_time": generation_time,
            "output_file": str(image_path)
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Stable Diffusion models with OpenVINO"
    )
    parser.add_argument(
        "--models-file",
        default="sd-models.txt",
        help="File containing model IDs, one per line (default: sd-models.txt)"
    )
    parser.add_argument(
        "--text-file",
        default="input.txt",
        help="File containing text, one per line (default: input.txt)"
    )
    parser.add_argument(
        "--device",
        default="CPU",
        help="Device for inference (default: CPU)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of inference steps (default: 25)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for generated images (default: output)"
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV file (default: results.csv)"
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=["FP32", "FP16", "BF16"],
        help="Inference precision (default: unspecified)"
    )

    args = parser.parse_args()

    # Determine models to use
    models = read_models_file(args.models_file)
    if not models:
        models = ["stabilityai/stable-diffusion-2-1-base"]
        print(f"No models found in {args.models_file}, using default: {models[0]}")

    # Determine prompts to use
    prompts = read_prompts_file(args.text_file)
    if not prompts:
        prompts = ["a watercolor fox in the forest, high detail"]
        print(f"No prompts found in {args.text_file}, using default: {prompts[0]}")

    print(f"Using {len(models)} model(s) and {len(prompts)} prompt(s)")
    print(f"Device: {args.device}, Steps: {args.steps}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)

    # Generate images for all model/prompt combinations
    all_results = []
    total_start = time.perf_counter()

    for model in models:
        try:
            results = generate_images(
                model, prompts, args.device, args.steps, args.output_dir, args.precision
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    total_end = time.perf_counter()

    # Save results to CSV
    if all_results:
        save_results(args.output, all_results)
        print(f"Results saved to: {args.output}")

    # Summary
    print("-" * 50)
    print(f"Generated {len(all_results)} images in {total_end - total_start:.2f}s")
    if len(all_results) > 0:
        print(f"Average time per image: {(total_end - total_start) / len(all_results):.2f}s")
    else:
        print("No images were generated successfully")


if __name__ == "__main__":
    main()
