"""
gpu_benchmark.py
================
Run the same script on your laptop and on the server, then compare
the results files it produces.

Usage:
    python gpu_benchmark.py

Output:
    benchmark_<hostname>_<device>.json  (one file per device tested)

Requires:
    pip install torch numpy
"""

import json
import platform
import socket
import time
from pathlib import Path

import torch


# ---------- benchmarks ----------------------------------------------------

def bench_matmul(device, size=4096, iters=20, dtype=torch.float32):
    """Pure matrix multiplication: measures raw FP32 throughput."""
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)

    # Warmup
    for _ in range(3):
        _ = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    seconds = (t1 - t0) / iters
    # 2 * N^3 FLOPs per matmul
    tflops = (2 * size ** 3) / seconds / 1e12
    return {"seconds_per_iter": seconds, "tflops": tflops}


def bench_conv2d(device, batch=32, channels=64, hw=224, iters=20):
    """CNN-style conv2d: closer to real DL workload."""
    x = torch.randn(batch, channels, hw, hw, device=device)
    conv = torch.nn.Conv2d(channels, channels, 3, padding=1).to(device)

    # Warmup
    for _ in range(3):
        _ = conv(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = conv(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return {"seconds_per_iter": (t1 - t0) / iters}


def bench_train_step(device, batch=64, iters=20):
    """Toy training step: forward + loss + backward + optimizer."""
    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 10),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.randn(batch, 2048, device=device)
    y = torch.randint(0, 10, (batch,), device=device)

    # Warmup
    for _ in range(3):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    return {"seconds_per_iter": (t1 - t0) / iters}


# ---------- runner --------------------------------------------------------

def run_on_device(device):
    name = "cpu"
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)

    print(f"\n=== {device} ({name}) ===")
    results = {"device": str(device), "device_name": name}

    print("  matmul 4096x4096 ...", flush=True)
    results["matmul"] = bench_matmul(device)
    print(f"    {results['matmul']['seconds_per_iter']*1000:8.2f} ms/iter"
          f"   {results['matmul']['tflops']:6.2f} TFLOPS")

    print("  conv2d 32x64x224x224 ...", flush=True)
    results["conv2d"] = bench_conv2d(device)
    print(f"    {results['conv2d']['seconds_per_iter']*1000:8.2f} ms/iter")

    print("  train step (3-layer MLP, batch=64) ...", flush=True)
    results["train_step"] = bench_train_step(device)
    print(f"    {results['train_step']['seconds_per_iter']*1000:8.2f} ms/iter")

    return results


def main():
    host = socket.gethostname()
    info = {
        "hostname": host,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    print(json.dumps(info, indent=2))

    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(exist_ok=True)

    # CPU
    cpu_results = run_on_device(torch.device("cpu"))
    cpu_path = out_dir / f"benchmark_{host}_cpu.json"
    cpu_path.write_text(json.dumps({**info, **cpu_results}, indent=2))
    print(f"\nWrote {cpu_path}")

    # Each GPU
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            dev = torch.device(f"cuda:{i}")
            gpu_results = run_on_device(dev)
            gpu_path = out_dir / f"benchmark_{host}_cuda{i}.json"
            gpu_path.write_text(json.dumps({**info, **gpu_results}, indent=2))
            print(f"Wrote {gpu_path}")
    else:
        print("\nNo CUDA available — only CPU was benchmarked.")


if __name__ == "__main__":
    main()