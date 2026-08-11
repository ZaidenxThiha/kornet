from __future__ import annotations

import statistics
import time

import torch

from .device import synchronize


def parameter_count(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_size_mb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2


@torch.inference_mode()
def benchmark_latency(model, sample, device, warmup=5, repeats=20, **forward_kwargs):
    model.eval()
    sample = sample.to(device)
    for _ in range(warmup):
        model(sample, **forward_kwargs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(sample, **forward_kwargs)
        synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    return {
        "latency_ms_mean": float(statistics.mean(timings)),
        "latency_ms_median": float(statistics.median(timings)),
        "throughput_images_s": float(sample.shape[0] * 1000 / statistics.mean(timings)),
        "peak_gpu_memory_mb": peak_mb,
    }


def estimate_macs(model, sample):
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(sample,), verbose=False)
        return int(macs)
    except (ImportError, RuntimeError, TypeError):
        return None
