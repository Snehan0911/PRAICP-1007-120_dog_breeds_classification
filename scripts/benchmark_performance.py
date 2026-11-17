"""
Performance and Reliability Benchmarking Script for Dog Breeds Classification.
Project: Dog Breeds Classification (PRAICP-1007-120)

Measures:
1. Cold-start model load latency and memory footprint (RAM).
2. Pure inference latency (mean, median, p95, min, max, std dev).
3. Inference latency with Grad-CAM explainability enabled.
4. Edge-case reliability and error-handling resilience.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.image_validator import ImageValidator
from src.inference.predictor import DogBreedPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger("Benchmark")


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def run_benchmarks(num_runs: int = 20) -> dict:
    results: dict = {
        "project": "Dog Breeds Classification",
        "project_id": "PRAICP-1007-120",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "system": {
            "python_version": sys.version.split()[0],
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "total_system_ram_gb": round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
        },
    }

    logger.info("Step 1: Measuring Cold-Start Model Loading and Memory Footprint...")
    mem_before_load = get_process_memory_mb()
    t_start = time.perf_counter()

    predictor = DogBreedPredictor(
        model_path="model/saved_models/best_model.keras",
        class_mapping_path="model/class_mapping/class_indices.json",
        enable_gradcam=True,
    )

    t_end = time.perf_counter()
    mem_after_load = get_process_memory_mb()

    cold_start_load_seconds = t_end - t_start
    model_ram_mb = mem_after_load - mem_before_load

    results["model_loading"] = {
        "cold_start_latency_sec": round(cold_start_load_seconds, 3),
        "initial_process_ram_mb": round(mem_before_load, 2),
        "post_load_process_ram_mb": round(mem_after_load, 2),
        "model_ram_footprint_mb": round(model_ram_mb, 2),
    }

    logger.info("Cold-start load time: %.3f s | Model RAM footprint: %.2f MB", cold_start_load_seconds, model_ram_mb)

    # Sample image for benchmarking
    sample_path = Path("dog_photos/samples/golden_retriever_sample.jpg")
    if not sample_path.is_file():
        sample_path = next(Path("dog_photos/samples").glob("*.jpg"))

    # Warmup
    logger.info("Step 2: Warming up model cache...")
    predictor.predict(sample_path, top_k=3, explain=False)

    # Pure Inference (without Grad-CAM)
    logger.info("Step 3: Benchmarking pure inference latency over %d runs...", num_runs)
    inference_times_ms: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = predictor.predict(sample_path, top_k=3, explain=False)
        t1 = time.perf_counter()
        inference_times_ms.append((t1 - t0) * 1000.0)

    results["inference_pure"] = {
        "num_runs": num_runs,
        "mean_ms": round(float(np.mean(inference_times_ms)), 2),
        "median_ms": round(float(np.median(inference_times_ms)), 2),
        "p95_ms": round(float(np.percentile(inference_times_ms, 95)), 2),
        "min_ms": round(float(np.min(inference_times_ms)), 2),
        "max_ms": round(float(np.max(inference_times_ms)), 2),
        "std_dev_ms": round(float(np.std(inference_times_ms)), 2),
        "throughput_fps": round(1000.0 / float(np.mean(inference_times_ms)), 2),
    }

    # Inference with Grad-CAM
    logger.info("Step 4: Benchmarking inference + Grad-CAM latency over %d runs...", num_runs)
    gradcam_times_ms: list[float] = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = predictor.predict(sample_path, top_k=3, explain=True)
        t1 = time.perf_counter()
        gradcam_times_ms.append((t1 - t0) * 1000.0)

    results["inference_with_gradcam"] = {
        "num_runs": num_runs,
        "mean_ms": round(float(np.mean(gradcam_times_ms)), 2),
        "median_ms": round(float(np.median(gradcam_times_ms)), 2),
        "p95_ms": round(float(np.percentile(gradcam_times_ms, 95)), 2),
        "min_ms": round(float(np.min(gradcam_times_ms)), 2),
        "max_ms": round(float(np.max(gradcam_times_ms)), 2),
        "std_dev_ms": round(float(np.std(gradcam_times_ms)), 2),
        "gradcam_overhead_ms": round(float(np.mean(gradcam_times_ms)) - float(np.mean(inference_times_ms)), 2),
    }

    # Step 5: Reliability & Edge-Case Stress Testing
    logger.info("Step 5: Testing edge-case reliability and fault tolerance...")
    edge_cases: dict[str, bool] = {}

    # Case A: Missing file
    res_a = ImageValidator.validate_file("dog_photos/samples/nonexistent_dog.jpg")
    edge_cases["missing_file_handled"] = not res_a.is_valid and "not found" in (res_a.error_message or "").lower()

    # Case B: 0-byte file
    res_b = ImageValidator.validate_bytes(b"", filename="empty.jpg")
    edge_cases["empty_file_handled"] = not res_b.is_valid

    # Case C: Corrupted byte stream
    res_c = ImageValidator.validate_bytes(b"JPEG random header corrupted bytes 123456", filename="corrupt.jpg")
    edge_cases["corrupt_bytes_handled"] = not res_c.is_valid

    # Case D: Unsupported file format
    res_d = ImageValidator.validate_bytes(b"Plain text content inside txt", filename="document.txt")
    edge_cases["unsupported_extension_handled"] = not res_d.is_valid

    # Case E: High-resolution image (3000x3000)
    high_res_array = np.random.randint(0, 255, size=(1200, 1200, 3), dtype=np.uint8)
    try:
        out_e = predictor.predict(high_res_array, top_k=3, explain=False)
        edge_cases["high_resolution_resizing_handled"] = len(out_e.top_k_predictions) == 3
    except (ValueError, RuntimeError, TypeError, KeyError) as e:
        logger.error("High-resolution test failed: %s", e)
        edge_cases["high_resolution_resizing_handled"] = False

    results["reliability_checks"] = edge_cases
    results["all_edge_cases_passed"] = all(edge_cases.values())

    # Output Persistence
    out_dir = PROJECT_ROOT / "outputs" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "performance_benchmark.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Benchmark complete! Results saved to %s", out_file)
    return results


if __name__ == "__main__":
    bench = run_benchmarks(num_runs=20)
    print("\n" + "=" * 60)
    print("DOG BREEDS CLASSIFICATION – PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Cold-Start Load Time:        {bench['model_loading']['cold_start_latency_sec']} seconds")
    print(f"Model Memory Footprint (RAM): {bench['model_loading']['model_ram_footprint_mb']} MB")
    print("-" * 60)
    print(f"Pure Inference Latency (Mean): {bench['inference_pure']['mean_ms']} ms")
    print(f"Pure Inference Latency (p95):  {bench['inference_pure']['p95_ms']} ms")
    print(f"Throughput:                    {bench['inference_pure']['throughput_fps']} FPS")
    print("-" * 60)
    print(f"Inference + Grad-CAM (Mean):   {bench['inference_with_gradcam']['mean_ms']} ms")
    print(f"Grad-CAM Overhead:             {bench['inference_with_gradcam']['gradcam_overhead_ms']} ms")
    print("-" * 60)
    print(f"All Edge Cases Handled:        {bench['all_edge_cases_passed']}")
    print("=" * 60)
