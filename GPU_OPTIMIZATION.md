# GPU Training Optimization: The Fundamentals

**Goal**: This document explains *exactly* what happens during training and how to make it go faster. It assumes no prior knowledge of GPU architecture.

---

## 1. The Mental Model: The Factory Line

Imagine a car factory.
*   **The CPU (Processor)** is the supply chain manager. It unpacks parts (data), organizes them, and ships them to the assembly line.
*   **The PCIe Bus** is the highway between the warehouse and the factory.
*   **The GPU (Graphics Card)** is the automated assembly line. It is incredibly fast but "dumb"—it does exactly what it's told, over and over again, in massive parallel bursts.

**The Golden Rule of Optimization**:
> **The Assembly Line (GPU) must NEVER stop.**

If the GPU is waiting for data (from the CPU) or waiting for instructions, you are losing time. 100% of optimization is about keeping the GPU busy processing meaningful work.

---

## 2. What to Track (The Dashboard)

You need to monitor specific metrics to know which part of the factory is too slow.

### A. Throughput (The Output)
*   **What it is**: How many positions (boards) you process per second.
*   **Metric**: `samples/sec` or `positions/sec`.
*   **Why**: This is the *only* metric that truly matters. If this number goes up, you win. Everything else is just a proxy.
*   **Where to see it**: Your training logs (`train.py` output).

### B. GPU Utilization (The Engine Load)
*   **What it is**: Are the GPU cores actively calculating math?
*   **Metric**: `GPU %` (0-100%).
*   **Target**: **95-100%**.
*   **Diagnosis**:
    *   **99%**: Excellent. The GPU is the bottleneck (this is good!).
    *   **0-50%**: horrendous. The GPU is idling, waiting for data. **Bottleneck: CPU or Disk**.
    *   **Fluctuating (0% -> 100% -> 0%)**: "Starvation". The CPU gives a batch, GPU finishes instantly, then waits. **Bottleneck: Batch Size too small or CPU too slow**.

### C. GPU Memory (The Workbench)
*   **What it is**: The high-speed RAM (VRAM) upon the graphics card (e.g., 24GB on a 4090).
*   **Metric**: `VRAM Usage` (MB/GB).
*   **Target**: As close to max as possible without crashing.
*   **Why**: A larger batch size means more work per "upload", reducing the overhead of talking to the GPU.
*   **Rule of Thumb**: Unused memory is wasted performance.

### D. PCIe Bandwidth (The Highway)
*   **What it is**: How fast data moves from System RAM -> GPU VRAM.
*   **Metric**: `PCIe Tx` (Transmit) and `Rx` (Receive).
*   **Why**: If you are sending huge data (like raw video), this highway can jam. For Chess (small 18x8x8 boards), this is rarely the issue, but good to check.

---

## 3. The Tools

Use these command-line tools. Do not blindly trust `Mac Activity Monitor` or generic task managers.

### 1. `nvidia-smi dmon` (The Best for Tuners)
Runs a scrolling log of statistics every second.
```bash
# Run this in a separate terminal
nvidia-smi dmon -s u
```
*   **sm**: Streaming Multiprocessor (Compute) %. **You want this high.**
*   **mem**: Memory Controller %. **You want this high.**

### 2. `nvtop` (The Visualizer)
Like `htop` but for GPUs. Shows charts.
```bash
nvtop
```
*   Great for seeing if usage is "spiky" (bad) or a "flat line" (good).

---

## 4. The Optimization Game (How to Play)

Follow this step-by-step loop to maximize your `samples/sec`.

### Level 1: The Easy Wins (Batch Size)
The CPU takes time to tell the GPU what to do ("Kernel Launch Overhead").
*   **Scenario**: Launching 1 item takes 10µs. Processing it takes 1µs. **90% wasted time.**
*   **Fix**: Send 10,000 items at once. Launch takes 10µs. Processing takes 10,000µs. **0.1% wasted time.**
*   **Action**: Double your `--batch-size` until the program crashes with `CUDA out of memory`. Then back off slightly.

### Level 2: Feeding the Beast (Data Loading)
If the GPU is 10x faster than the CPU, the GPU will spend 90% of its time waiting for the CPU to load the next batch.
*   **Symptom**: GPU Util is volatile (bouncing).
*   **Action**:
    *   Increase `--num-workers` (Parallel CPU loaders). Usually set to number of CPU cores.
    *   Enable `pin_memory=True` (Standard in PyTorch, ensures faster transfer).
    *   **Pro Move**: Do less work on the CPU. Pre-process data offline if possible.

### Level 3: The Speed of Light (Precision)
GPUs have specialized "Tensor Cores" that are 4x-8x faster at doing math with fewer decimal places (FP16) than standard (FP32).
*   **Action**: Use **Automatic Mixed Precision (AMP)** via `torch.cuda.amp` (Already in your code!).
*   **Why**: It's free speed and half the memory usage (allowing 2x Batch Size).

### Level 4: Compilation (PyTorch 2.0+)
Python is slow. PyTorch reads your code line-by-line.
*   **Action**: Use `torch.compile(model)`.
*   **Effect**: PyTorch analyzes your entire model and fuses operations together into a single accelerated kernel.

---

## 5. Summary Checklist

| If you see... | It means... | Do this... |
|---------------|-------------|------------|
| **GPU Util < 80%** | Starvation | Increase `batch_size`. Increase `num_workers`. |
| **GPU Util Spiky** | CPU Bottleneck | Optimize `data.py` or add more workers. |
| **Out of Memory** | Too ambitious | Decrease `batch_size`. |
| **Low Throughput** | Inefficient Math | Enable `AMP`. Use `torch.compile`. |
| **PCIe Bandwidth High** | Bus choke | Compress data before sending (not needed for chess). |

**Your Goal**: Run `nvidia-smi dmon` and see **`sm` pinned at 98-100%** and **`mem` pinned at high usage**. That is a perfectly optimized system.
