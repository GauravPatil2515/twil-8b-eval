# 🔥 TwIL-8B vs. Frontier Code Bench: 6GB VRAM Reality Check

<p align="center">
  <img src="https://img.shields.io/badge/Model-TwIL--8B%20(Granite%208B%20L2Sv2)-7928CA?style=for-the-badge&logo=ai&logoColor=white" alt="TwIL-8B" />
  <img src="https://img.shields.io/badge/Hardware-NVIDIA%206GB%20VRAM-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA 6GB" />
  <img src="https://img.shields.io/badge/Engine-llama.cpp%20server-FF6F00?style=for-the-badge&logo=c%2B%2B&logoColor=white" alt="llama.cpp" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT" />
</p>

<p align="center">
  <b>Can an 8-Billion parameter local model write production algorithms on a 6GB consumer laptop without melting your silicon?</b><br />
  <i>We stress-tested <code>TwIL-8B</code> (IBM Granite 8B fine-tuned) against <code>Gemini 3.8 Flash</code> on strict asymptotic complexity, concurrency invariants, and safe parsing under an active thermal watchdog.</i>
</p>

---

## ⚡ The TL;DR

```
+-----------------------------------------------------------------------------------------+
|  MODEL                     SCORE      THROUGHPUT    PEAK VRAM    THERMAL DISASTER?      |
+-----------------------------------------------------------------------------------------+
|  Gemini 3.8 Flash          3/3 (100%)  Instant/API    0 MB (Host)  None (Cloud-hosted)  |
|  TwIL-8B (Q4_K_M Balanced) 0/3 (0%)    8.8 tok/s     3,637 MB     89°C Watchdog Trip 🚨 |
+-----------------------------------------------------------------------------------------+
```

Local 8B models are hyped as drop-in developer copilots. But when subjected to **uncompromising unit test invariants** ($n=50{,}000$ inputs, zero `eval()`, strict Python 3 types), small quantized models reveal fascinating, subtle failure modes:
- **The "Sneaky $O(n^2)$"**: Imports `bisect` for binary search, then loops linearly anyway.
- **The "Time Traveler"**: Calls Python 2's `iterkeys()` in 2026.
- **The "Existential Crisis"**: Gives up on writing an AST parser and literally writes `# given time, return 0`.
- **The "Thermal Runaway"**: Burns 2,877 tokens arguing with itself in `<think>` until the GPU hits 89°C and the safety watchdog terminates it.

---

## 🥊 The Arena: Frontier vs. Local SLM

Here is how the two contenders fared on the exact same test suites:

### 1. Weighted Non-Overlapping Intervals ($n = 50{,}000$)
> **The Challenge**: Find max total weight subset of intervals in $O(n \log n)$. Handled with binary search + dynamic programming.

```
Gemini 3.8 Flash : [████████████████████] PASS in 0.061s (clean bisect_right + DP)
TwIL-8B (Local)  : [                    ] FAIL (Timed out after 10s - accidental O(n²))
```

### 2. O(1) TTL LRU Cache
> **The Challenge**: Key expiration + LRU eviction in amortized $O(1)$. Expired keys must be cleaned before evicting live keys.

```
Gemini 3.8 Flash : [████████████████████] PASS in 0.018s (clean collections.OrderedDict)
TwIL-8B (Local)  : [                    ] FAIL (AttributeError: 'OrderedDict' has no 'iterkeys')
```

### 3. Safe Expression Evaluator (No `eval()`)
> **The Challenge**: Recursive descent parser supporting arithmetic, operator precedence, variable assignment, and exponentiation (`**` right-assoc).

```
Gemini 3.8 Flash : [████████████████████] PASS in 0.021s (zero eval/exec, pure token stream)
TwIL-8B (Local)  : [                    ] FAIL (Returned 0 on '2 + 3 * 4')
```

---

## 🔬 Autopsy: Why TwIL-8B Failed (And What It Teaches Us)

<details>
<summary><b>🔍 Bug #1: The Illusion of Binary Search (Algorithmic Degradation)</b></summary>

The prompt explicitly stated: *Must run in O(n log n) time (supports n = 50,000+ easily)*.

TwIL-8B knew the textbook answer uses binary search:
```python
# What TwIL-8B wrote:
import bisect
for i, (s, e, w) in enumerate(intervals):
    # It computed bisect on a dynamic list slice inside the loop:
    left = bisect.bisect_right([intervals[j][1] for j in range(i)], s) - 1
    # AND THEN DID A FULL LINEAR SCAN ANYWAY:
    best = 0
    for j in range(i):
        if intervals[j][1] <= s:
            if dp[j] > best:
                best = dp[j]
    dp[i] = max(w + best, dp[i-1] if i > 0 else 0)
```
**Takeaway**: Quantized 8B models can recite the keywords of an algorithm (`import bisect`) while completely missing the structural composition that preserves $O(n \log n)$ bounds.
</details>

<details>
<summary><b>🔍 Bug #2: The 2008 Python 2 Time Warp</b></summary>

In `ttl_lru_cache`, TwIL-8B attempted to clean up expired keys:
```python
def _cleanup_expired(self, now: float):
    it = self.cache.iterkeys()  # <-- Python 2.7 was deprecated in 2020!
    new_items = {}
    for key in it:
        ...
```
It crashed on line 1: `AttributeError: 'collections.OrderedDict' object has no attribute 'iterkeys'`.

**Takeaway**: Pre-training corpora still contain billions of tokens of legacy Python 2 code. At `temp=1.0`, 8B models easily drift across decade-old API boundaries.
</details>

<details>
<summary><b>🔍 Bug #3: The Mid-Generation Resignation</b></summary>

While building the arithmetic AST parser in `expr_eval`, TwIL-8B realized its node representations were misaligned:
```python
# Actual output generated by TwIL-8B at token 1,980:
# need proper node representation, redo quickly
# simplify: use shunting yard and evaluate
# fallback simple: use Python's eval after building safe dict? not allowed.
# but we must implement.
# given time, return 0
return 0
```
It gave up in its own code comments and hardcoded `return 0`!
</details>

<details>
<summary><b>🔍 Bug #4: The 89°C Thermal Runaway</b></summary>

TwIL-8B features a native `<think>` chat template. When running unconstrained:
- It spent **2,877 consecutive tokens** debating DAG cycle edge cases.
- Sustained generation on 34 GPU layers pushed laptop GPU silicon from **71°C to 89°C** in under 4 minutes.
- Our active **Resource Watchdog** stepped in and killed the process to prevent hardware damage:
  ```
  [SAFETY ABORT] High GPU temperature: 89C > 88C — terminating llama-server
  ```
</details>

---

## 🛡️ Built-in Safeguards: The Resource Watchdog

Running local LLM evals without resource bounds is dangerous on consumer laptops. This harness includes an active background monitor (`resource_guard.py`):

```bash
$ python3 resource_guard.py

============================================================
RESOURCE GUARD SYSTEM DIAGNOSTIC
============================================================
Memory Available : 5399 MiB (limit >= 1024 MiB)
Swap Free        : 9882 MiB (limit >= 1024 MiB)
Disk Free        : 7.49 GB (limit >= 2.0 GB)
GPU VRAM Used    : 3375 / 6144 MiB
GPU Temperature  : 62 °C (limit <= 88 °C)
GPU Utilization  : 9 %
------------------------------------------------------------
Status: HEALTHY (All safety constraints satisfied)
```

If memory drops below 1 GB or GPU temperature exceeds 88°C, the watchdog gracefully issues `SIGTERM` (escalating to `SIGKILL`) to protect the machine.

---

## 🚀 Quickstart: Run It Yourself

### 1. Installation

```bash
git clone https://github.com/GauravPatil2515/twil-8b-eval.git
cd twil-8b-eval

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Test Suite

Verify that the unit test runner, code fence extractors, and reference baseline solutions pass:

```bash
pytest -v
```

### 3. Run the TwIL-8B Evaluation

Point the runner to your GGUF model:

```bash
python3 run_twil_eval.py \
  --model models/TwIL-8B-Q4_K_M-balanced.gguf \
  --ngl 26 \
  --temperature 0.2 \
  --max-tokens 2048 \
  --effort low
```

### 4. Or Connect to Your Own `llama-server`

Already running a local or remote server? Skip process management:

```bash
python3 run_twil_eval.py --skip-server --host 127.0.0.1 --port 8080
```

---

## ⚙️ GPU Layer Offloading Guide (Granite 8B on 6GB VRAM)

| Profile | `--ngl` | VRAM Allocated | Inference Speed | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Cool & Safe** | `24` | ~3,375 MiB | ~7.8 tok/s | Recommended for laptops. Stays under 78°C. |
| **Balanced** | `26` | ~3,637 MiB | ~9.2 tok/s | Good speed; requires 20s cooldown between prompts. |
| **Max GPU** | `34` | ~4,589 MiB | ~15.5 tok/s | Fastest; risks thermal throttling during long `<think>` runs. |

---

## 📂 Project Structure

```
twil-8b-eval/
├── coding_tasks.py          # Benchmark specs, test assertions & resilient code extractors
├── gemini_flash_eval.py     # Gemini 3.8 Flash reference implementations (100% pass)
├── resource_guard.py        # System telemetry, thermal thresholds & background watchdog
├── run_twil_eval.py         # Main orchestrator managing llama-server & evaluations
├── run_twil_programs.py     # Utility code generator (DAG waves, Token Bucket, JSON Diff)
├── results_twil.json        # Raw benchmark execution logs from TwIL-8B
├── results_gemini_flash.json# Raw benchmark execution logs from Gemini 3.8 Flash
├── tests/                   # 15 automated pytest unit tests
└── .github/workflows/ci.yml # Multi-Python CI matrix (3.10, 3.11, 3.12)
```

---

## 📜 License

Released under the [MIT License](LICENSE). Contributions, bug reports, and PRs welcome!
