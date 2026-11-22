"""Generate and verify standalone Python utility programs using TwIL-8B via llama-server."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from coding_tasks import TASKS, extract_python_code, run_task_tests
from resource_guard import SafetyLimits, check_safety, take_snapshot
from run_twil_eval import (
    BASE_DIR,
    BIN_DIR,
    DEFAULT_HOST,
    DEFAULT_MODEL_PATH,
    DEFAULT_PORT,
    DEFAULT_SERVER_BIN,
    ResourceWatchdog,
    _http_post_json,
    terminate_process,
    wait_for_server,
)

SYSTEM_PROMPT = (
    "You are an expert Python 3 software engineer. "
    "Write clean, bug-free, O(n log n) or better Python 3 code using standard library APIs. "
    "Avoid Python 2 idioms (e.g., never use iterkeys). "
    "Double-check edge cases and return only a single ```python``` code block."
)

PROGRAM_PROMPTS = [
    {
        "name": "task_dag",
        "prompt": (
            "Write a clean, self-contained Python 3 program with:\n"
            "1. `schedule_waves(deps: dict[str, list[str]]) -> list[list[str]]` where `deps[task]` lists "
            "prerequisites that must finish before `task`. Group tasks into ordered parallel waves "
            "(sorted alphabetically within each wave). Raise `ValueError` if there is a dependency cycle "
            "or a missing prerequisite task.\n"
            "2. An `if __name__ == '__main__':` block with comprehensive `assert` tests covering diamonds, "
            "independent components, empty input, missing dependency, and cycles.\n"
            "Output only valid Python code inside a single ```python``` block."
        ),
    },
    {
        "name": "token_bucket",
        "prompt": (
            "Write a clean, self-contained Python 3 program implementing a deterministic per-key "
            "`TokenBucketLimiter(rate_per_sec: float, capacity: float)`:\n"
            "- `allow(key: str, now: float, tokens: float = 1.0) -> bool` refills based on elapsed time "
            "since last check (starting full at `capacity`) and consumes `tokens` if available.\n"
            "- `retry_after(key: str, now: float, tokens: float = 1.0) -> float` returns `0.0` if enough "
            "tokens are available right now, otherwise seconds needed to wait.\n"
            "- Include an `if __name__ == '__main__':` block with `assert` tests covering burst, "
            "refill, exhaustion, `retry_after`, and independent keys.\n"
            "Output only valid Python code inside a single ```python``` block."
        ),
    },
    {
        "name": "json_diff",
        "prompt": (
            "Write a clean, self-contained Python 3 program implementing `diff_values(old, new, path='') -> list[dict]` "
            "that recursively diffs two JSON-like Python values (`dict`, `list`, primitives) and returns "
            "a list of change records with keys `'path'`, `'kind'` (`'added'`, `'removed'`, `'changed'`), "
            "`'old'`, and `'new'`. Use dot-notation for dict keys and `[idx]` for list indices.\n"
            "Include an `if __name__ == '__main__':` block with `assert` tests covering nested dicts, "
            "lists of differing lengths, type changes, and identical inputs.\n"
            "Output only valid Python code inside a single ```python``` block."
        ),
    },
]


def wait_cooldown(target_c: int = 77, max_wait_s: float = 45.0) -> None:
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        snap = take_snapshot()
        if snap.gpu_temp_c <= target_c:
            return
        time.sleep(3.0)


def run_generation(
    prompt: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_tokens: int = 1536,
    temp: float = 0.2,
    enable_thinking: bool = False,
) -> dict:
    payload = {
        "model": "TwIL-8B",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_p": 0.9,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    t0 = time.perf_counter()
    resp = _http_post_json(f"http://{host}:{port}/v1/chat/completions", payload, timeout=600.0)
    elapsed = time.perf_counter() - t0
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = resp.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    tps = round(comp_tokens / elapsed, 2) if elapsed > 0 and comp_tokens else 0.0
    return {
        "content": content,
        "reasoning": reasoning,
        "usage": usage,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_sec": tps,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Python utility programs using TwIL-8B.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to GGUF model")
    parser.add_argument("--server-bin", type=Path, default=DEFAULT_SERVER_BIN, help="Path to llama-server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="llama-server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="llama-server port")
    parser.add_argument("--ngl", type=int, default=24, help="GPU layers to offload")
    parser.add_argument("--ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable reasoning thought block")
    parser.add_argument("--skip-server", action="store_true", help="Connect to already running server")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=BASE_DIR / "generated_programs",
        help="Directory to save generated programs",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    limits = SafetyLimits(min_mem_available_mb=1024, min_swap_free_mb=1024, min_disk_free_gb=2.0, max_gpu_temp_c=88)
    wait_cooldown(target_c=77, max_wait_s=45.0)
    pre = take_snapshot()
    if err := check_safety(pre, limits):
        print(f"Pre-flight failed: {err}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    proc: subprocess.Popen[str] | None = None
    watchdog: ResourceWatchdog | None = None

    if not args.skip_server:
        if not args.model.exists():
            print(f"Model file not found: {args.model}", file=sys.stderr)
            return 1
        if not args.server_bin.exists():
            print(f"llama-server binary not found at: {args.server_bin}", file=sys.stderr)
            return 1

        env = os.environ.copy()
        if BIN_DIR.exists():
            env["LD_LIBRARY_PATH"] = f"{BIN_DIR}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

        cmd = [
            str(args.server_bin),
            "--model", str(args.model),
            "--host", args.host,
            "--port", str(args.port),
            "--n-gpu-layers", str(args.ngl),
            "--ctx-size", str(args.ctx),
            "--parallel", "1",
            "--threads", "6",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--no-warmup",
            "--jinja",
        ]
        log_path = BASE_DIR / "llama_server_programs.log"
        log_fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
        watchdog = ResourceWatchdog(proc, limits)
        watchdog.start()

    report: list[dict] = []
    try:
        wait_for_server(proc, watchdog, args.host, args.port, timeout_s=240)
        snap0 = take_snapshot()
        print(f"[LOADED] VRAM={snap0.gpu_mem_used_mb}/{snap0.gpu_mem_total_mb} MiB, Temp={snap0.gpu_temp_c}C")

        for idx, spec in enumerate(PROGRAM_PROMPTS):
            if idx > 0:
                wait_cooldown(target_c=77, max_wait_s=30.0)
            name = spec["name"]
            print(f"\n--- Generating {name}.py ---")
            gen = run_generation(
                spec["prompt"],
                host=args.host,
                port=args.port,
                max_tokens=args.max_tokens,
                temp=args.temperature,
                enable_thinking=args.enable_thinking,
            )
            code = extract_python_code(gen["content"] or gen["reasoning"])
            file_path = args.out_dir / f"{name}.py"
            file_path.write_text(code + "\n", encoding="utf-8")

            try:
                run_res = subprocess.run(
                    [sys.executable, str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                passed = run_res.returncode == 0
                out_text = (run_res.stdout + "\n" + run_res.stderr).strip()
            except subprocess.TimeoutExpired:
                passed = False
                out_text = "Timed out after 10s"

            snap = take_snapshot()
            print(
                f"[{name}] pass={passed} | time={gen['elapsed_s']}s | "
                f"tokens={gen['usage'].get('completion_tokens', 0)} ({gen['tokens_per_sec']} tok/s) | "
                f"Temp={snap.gpu_temp_c}C"
            )
            if not passed:
                print(f"  Output: {out_text[:400]}")
            report.append(
                {
                    "name": name,
                    "file": str(file_path),
                    "passed": passed,
                    "output": out_text,
                    "elapsed_s": gen["elapsed_s"],
                    "tokens_per_sec": gen["tokens_per_sec"],
                    "usage": gen["usage"],
                    "reasoning_chars": len(gen["reasoning"]),
                    "temp_after_c": snap.gpu_temp_c,
                }
            )

        wait_cooldown(target_c=77, max_wait_s=30.0)
        wi_task = TASKS[0]
        print(f"\n--- Re-testing benchmark task [{wi_task.task_id}] ---")
        gen_wi = run_generation(
            wi_task.prompt,
            host=args.host,
            port=args.port,
            max_tokens=args.max_tokens,
            temp=args.temperature,
            enable_thinking=args.enable_thinking,
        )
        wi_code = extract_python_code(gen_wi["content"] or gen_wi["reasoning"])
        (args.out_dir / "weighted_intervals_v2.py").write_text(wi_code + "\n", encoding="utf-8")
        wi_pass, wi_out = run_task_tests(wi_code, wi_task)
        snap_wi = take_snapshot()
        print(
            f"[{wi_task.task_id} temp={args.temperature}] pass={wi_pass} | time={gen_wi['elapsed_s']}s | "
            f"tokens={gen_wi['usage'].get('completion_tokens', 0)} ({gen_wi['tokens_per_sec']} tok/s) | "
            f"Temp={snap_wi.gpu_temp_c}C"
        )
        if not wi_pass:
            print(f"  Output: {wi_out[:400]}")
        report.append(
            {
                "name": "weighted_intervals_v2",
                "file": str(args.out_dir / "weighted_intervals_v2.py"),
                "passed": wi_pass,
                "output": wi_out,
                "elapsed_s": gen_wi["elapsed_s"],
                "tokens_per_sec": gen_wi["tokens_per_sec"],
                "usage": gen_wi["usage"],
                "reasoning_chars": len(gen_wi["reasoning"]),
                "temp_after_c": snap_wi.gpu_temp_c,
            }
        )
    finally:
        if watchdog:
            watchdog.stop()
        if proc:
            terminate_process(proc)

    summary = {
        "peak_gpu_mb": watchdog.peak_gpu_mb if watchdog else 0,
        "min_mem_available_mb": watchdog.min_mem_avail_mb if watchdog else pre.mem_available_mb,
        "max_gpu_temp_c": watchdog.max_gpu_temp_c if watchdog else pre.gpu_temp_c,
        "abort_reason": watchdog.abort_reason if watchdog else None,
        "results": report,
    }
    (args.out_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] Passed {sum(1 for r in report if r['passed'])}/{len(report)} | PeakTemp={summary['max_gpu_temp_c']}C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
