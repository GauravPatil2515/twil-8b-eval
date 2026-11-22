# TwIL-8B Coding Evaluation & Resource-Guarded Benchmark Suite

[![CI](https://github.com/GauravPatil2515/twil-8b-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/GauravPatil2515/twil-8b-eval/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hardware: 6GB VRAM](https://img.shields.io/badge/hardware-CUDA%20%7C%206GB%20VRAM-orange.svg)](#hardware-requirements--gpu-offloading)

A production-grade, self-verifying evaluation harness designed to stress-test local Small Language Models (SLMs)—specifically **TwIL-8B** (`Granite8B L2Sv2 S30` Q4_K_M balanced)—against frontier models (**Gemini 3.8 Flash**) on algorithmic complexity, concurrency invariants, and systems programming tasks under strict consumer hardware constraints.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Benchmark Tasks](#benchmark-tasks)
- [Empirical Results & Comparative Analysis](#empirical-results--comparative-analysis)
- [Failure Mode Root Cause Analysis](#failure-mode-root-cause-analysis)
- [System Architecture & Resource Guard](#system-architecture--resource-guard)
- [Hardware Requirements & GPU Offloading](#hardware-requirements--gpu-offloading)
- [Quickstart Guide](#quickstart-guide)
- [CLI Reference](#cli-reference)
- [Repository Structure](#repository-structure)
- [License](#license)

---

## Overview & Architecture

Deploying 8-billion parameter models locally on edge and consumer hardware (such as laptops or workstations with 6 GB VRAM) introduces distinct engineering challenges:
1. **Memory & VRAM Contention**: Fitting an 8.8B model in Q4_K_M (~5.0 GB weights) leaves under 1.1 GB of headroom for KV cache and CUDA overhead.
2. **Thermal Throttling**: Sustained generation at high token budgets can rapidly push mobile and small-form GPUs past safe junction temperatures (>88°C).
3. **Reasoning Invariant Drift**: Quantized local SLMs frequently exhibit subtle syntax drift (e.g. emitting Python 2 idioms), asymptotic complexity degradations ($O(n^2)$ instead of $O(n \log n)$), or parsing truncations.

This suite provides an automated, sandboxed harness that manages `llama-server` life cycles, intercepts hardware telemetry via an active watchdog thread, executes self-verifying unit tests in isolated subprocesses, and records structured performance metrics.

```
+-------------------------------------------------------------------------+
|                              twil-8b-eval                               |
+-------------------------------------------------------------------------+
          |                                            |
          v                                            v
+--------------------+                       +--------------------+
|  coding_tasks.py   |                       | resource_guard.py  |
|  - Tasks & Tests   |                       | - RAM / Swap Telemetry
|  - Test Runner     |                       | - GPU Temp / VRAM  |
|  - Subprocess Exec |                       | - Thermal Watchdog |
+--------------------+                       +--------------------+
          |                                            |
          +---------------------+----------------------+
                                |
                                v
                   +------------------------+
                   |    run_twil_eval.py    |
                   | - llama-server Manager |
                   | - Dynamic Cooldowns    |
                   | - Safety Termination   |
                   +------------------------+
```

---

## Benchmark Tasks

The evaluation suite tests three programming challenges requiring strict asymptotic bounds and correctness:

| Task ID | Title | Key Invariants | Asymptotic Target |
| :--- | :--- | :--- | :--- |
| `weighted_intervals` | Weighted Interval Scheduling | Binary search + Dynamic Programming; must handle $n = 50{,}000$ intervals in $<10\text{s}$. | $O(n \log n)$ |
| `ttl_lru_cache` | $O(1)$ LRU Cache with Per-Key TTL | Doubly linked list or `OrderedDict`; amortized $O(1)$ `get`/`put`; expired items evicted prior to LRU items. | $O(1)$ amortized |
| `expr_eval` | Safe Arithmetic & Variable Parser | Multi-statement execution (`;`), Python precedence (`**` right-assoc), unary bind rules, no `eval()`/`exec()`. | $O(L)$ where $L = \text{len}$ |

---

## Empirical Results & Comparative Analysis

Evaluation executed on an NVIDIA GPU (6,144 MiB VRAM) and 6-core CPU:

| Metric | TwIL-8B (Q4_K_M Balanced) | Gemini 3.8 Flash (Baseline) |
| :--- | :--- | :--- |
| **Tasks Passed** | **0 / 3 (0%)** | **3 / 3 (100%)** |
| `weighted_intervals` | **FAIL** (Timed out after 10s) | **PASS** (0.061s) |
| `ttl_lru_cache` | **FAIL** (`AttributeError: iterkeys`) | **PASS** (0.018s) |
| `expr_eval` | **FAIL** (`AssertionError`) | **PASS** (0.021s) |
| **Throughput** | 8.28 – 9.29 tokens/sec | Cloud API / Instantaneous |
| **Peak VRAM** | 3,637 MiB (`ngl=26`) | 0 MiB (Host) |
| **Peak GPU Temp** | 83°C (cooldowns required) | Ambient (Host) |

---

## Failure Mode Root Cause Analysis

### 1. Asymptotic Degradation (`weighted_intervals`)
- **Observed Behavior**: The model correctly recognized that binary search was needed and imported `bisect`. However, it generated:
  ```python
  left = bisect.bisect_right([intervals[j][1] for j in range(i)], s) - 1
  for j in range(i):
      if intervals[j][1] <= s:
          best = max(best, dp[j])
  ```
- **Root Cause**: Rebuilding a slice list `[intervals[j][1] for j in range(i)]` inside an $O(n)$ loop transformed what should have been an $O(n \log n)$ algorithm into $O(n^2)$. On the $n = 50{,}000$ test input, execution exceeded the 10-second subprocess timeout.

### 2. Python 2 API Hallucination (`ttl_lru_cache`)
- **Observed Behavior**: `c.put("a", 1)` crashed immediately with:
  ```
  AttributeError: 'collections.OrderedDict' object has no attribute 'iterkeys'
  ```
- **Root Cause**: The model emitted the obsolete Python 2 dict method `iterkeys()` instead of `.keys()` or direct iteration. Additionally, it unpacked internal tuple state backwards (`expires_at, _ = entry` vs `(value, expires)`).

### 3. Parser Confusion & Premature Abort (`expr_eval`)
- **Observed Behavior**: Failed on the first unit test `2 + 3 * 4 == 14`.
- **Root Cause**: The generated recursive descent parser mixed tuple node tags (`INT`, `FLOAT`, `OP`) and aborted mid-implementation with an emergency fallback:
  ```python
  # need proper node representation, redo quickly
  # given time, return 0
  return 0
  ```

### 4. Reasoning Token Budgets & Thermal Runaway
- When invoked with unrestricted reasoning (`enable_thinking=True`), TwIL-8B generates extensive internal reasoning tokens inside `<think>...</think>`.
- In test runs generating $>2{,}800$ continuous reasoning tokens at `-ngl 34` (15.5 tok/s), GPU temperature escalated from 71°C to **89°C**, triggering the safety abort threshold (`88°C`).
- Conversely, setting `{reasoning effort: low}` suppressed thinking entirely (`0` reasoning tokens), forcing the 8B model to generate code zero-shot, resulting in the logic and syntax flaws documented above.

---

## System Architecture & Resource Guard

To prevent system lockups, out-of-memory kernel panics, or thermal damage to mobile/laptop GPUs, `resource_guard.py` runs a background monitoring thread:

```python
from resource_guard import SafetyLimits, take_snapshot, check_safety

limits = SafetyLimits(
    min_mem_available_mb=1024,  # Abort if available RAM < 1 GB
    min_swap_free_mb=1024,      # Abort if free swap < 1 GB
    min_disk_free_gb=2.0,       # Abort if free disk < 2 GB
    max_gpu_temp_c=88,          # Abort if GPU temp > 88°C
)
```

The watchdog monitors `/proc/meminfo` and `nvidia-smi` at 1.0-second intervals. If any safety boundary is crossed, the runner sends `SIGTERM` to `llama-server`, escalating to `SIGKILL` after 3 seconds.

---

## Hardware Requirements & GPU Offloading

Tested on an Ubuntu Linux system with an NVIDIA 6 GB GPU. The table below outlines recommended offloading profiles for 40-layer Granite-8B GGUF architectures:

| Profile | `--n-gpu-layers` | VRAM Allocated | Inference Speed | Thermal Stability |
| :--- | :--- | :--- | :--- | :--- |
| **Conservative** (Default) | `24` | ~3,375 MiB | ~7.5 – 8.8 tok/s | Excellent ($\le 78^\circ\text{C}$) |
| **Balanced** | `26` | ~3,637 MiB | ~8.4 – 9.5 tok/s | Moderate ($\le 83^\circ\text{C}$, needs inter-task cooldown) |
| **Max GPU** | `34` | ~4,589 MiB | ~13.7 – 15.6 tok/s | Warning: Thermal throttle on sustained generation |

---

## Quickstart Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/GauravPatil2515/twil-8b-eval.git
cd twil-8b-eval

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Reference Suite

Run the automated test suite to ensure test invariants and baseline reference solutions pass:

```bash
pytest -v
```

### 3. Check System Telemetry

Run the pre-flight resource diagnostic:

```bash
python3 resource_guard.py
```

### 4. Evaluate TwIL-8B

Point to your downloaded GGUF model and run the evaluation:

```bash
python3 run_twil_eval.py \
  --model models/TwIL-8B-Q4_K_M-balanced.gguf \
  --ngl 26 \
  --temperature 0.2 \
  --max-tokens 2048 \
  --effort low
```

To evaluate an external or already running `llama-server` instance without managing subprocesses:

```bash
python3 run_twil_eval.py --skip-server --host 127.0.0.1 --port 8080
```

---

## CLI Reference

### `run_twil_eval.py`

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | `models/TwIL-8B-Q4_K_M-balanced.gguf` | Path to the model GGUF file. |
| `--server-bin` | `bin/llama-server` | Path to the `llama-server` executable. |
| `--host` | `127.0.0.1` | HTTP host for the llama-server endpoint. |
| `--port` | `18080` | Port for the llama-server endpoint. |
| `--ngl` | `26` | Number of transformer layers to offload to GPU. |
| `--ctx` | `4096` | KV cache context window size in tokens. |
| `--max-tokens` | `2048` | Maximum output generation tokens. |
| `--temperature` | `0.2` | Sampling temperature (`0.2` recommended for code). |
| `--effort` | `low` | Reasoning effort: `off`, `low`, `medium`, `high`. |
| `--output` | `results_twil.json` | Path to save structured benchmark output. |
| `--skip-server`| `false` | Connect to existing server without spawning process. |

---

## Repository Structure

```
twil-8b-eval/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated multi-python CI pipeline
├── tests/
│   ├── test_coding_tasks.py     # Code fence extraction and task test verification
│   ├── test_gemini_solutions.py # Reference solution validation
│   └── test_resource_guard.py   # Watchdog telemetry and threshold tests
├── coding_tasks.py              # Benchmark task specifications & sandboxed runner
├── gemini_flash_eval.py         # Gemini 3.8 Flash reference implementations
├── resource_guard.py            # Hardware telemetry & thermal safety guard
├── run_twil_eval.py             # Main evaluation orchestrator for llama-server
├── run_twil_programs.py         # Utility program generator (DAG, Token Bucket, Diff)
├── results_twil.json            # Empirical benchmark output from TwIL-8B
├── results_gemini_flash.json    # Reference benchmark output from Gemini 3.8 Flash
├── pyproject.toml               # Package configuration & tool settings
├── requirements.txt             # Minimal dependencies
├── LICENSE                      # MIT Open Source License
└── README.md                    # Comprehensive documentation
```

---

## License

This project is licensed under the [MIT License](LICENSE).
