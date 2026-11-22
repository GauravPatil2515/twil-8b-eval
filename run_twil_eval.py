"""Safe, resource-guarded coding evaluation runner for TwIL-8B via llama-server."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from coding_tasks import TASKS, extract_python_code, run_task_tests
from resource_guard import SafetyLimits, check_safety, take_snapshot

BASE_DIR = Path(__file__).resolve().parent
BIN_DIR = BASE_DIR / "bin"
DEFAULT_SERVER_BIN = Path(shutil.which("llama-server") or BIN_DIR / "llama-server")
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "TwIL-8B-Q4_K_M-balanced.gguf"
DEFAULT_PORT = 18080
DEFAULT_HOST = "127.0.0.1"


def _http_get_json(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _http_post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class ResourceWatchdog:
    """Background monitor that terminates llama-server if RAM/VRAM/temp limits are breached."""

    def __init__(self, proc: subprocess.Popen[str], limits: SafetyLimits) -> None:
        self.proc = proc
        self.limits = limits
        self.stop_event = threading.Event()
        self.abort_reason: str | None = None
        self.peak_gpu_mb = 0
        self.min_mem_avail_mb = 10**9
        self.max_gpu_temp_c = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._thread.join(timeout=3)

    def _loop(self) -> None:
        while not self.stop_event.wait(1.0):
            if self.proc.poll() is not None:
                return
            snap = take_snapshot()
            self.peak_gpu_mb = max(self.peak_gpu_mb, snap.gpu_mem_used_mb)
            self.min_mem_avail_mb = min(self.min_mem_avail_mb, snap.mem_available_mb)
            self.max_gpu_temp_c = max(self.max_gpu_temp_c, snap.gpu_temp_c)
            reason = check_safety(snap, self.limits)
            if reason is not None:
                self.abort_reason = reason
                print(f"\n[SAFETY ABORT] {reason} — terminating llama-server", file=sys.stderr)
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                return


def wait_for_server(proc: subprocess.Popen[str] | None, watchdog: ResourceWatchdog | None, host: str, port: int, timeout_s: int = 120) -> None:
    """Wait until llama-server /health returns ok, aborting early on crash or safety breach."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if watchdog and watchdog.abort_reason:
            raise RuntimeError(f"Aborted during startup: {watchdog.abort_reason}")
        if proc and proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
        health = _http_get_json(f"http://{host}:{port}/health")
        if health and health.get("status") == "ok":
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for llama-server at http://{host}:{port}/health")


def terminate_process(proc: subprocess.Popen[str]) -> None:
    """Gracefully terminate llama-server, escalating to SIGKILL if needed."""
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TwIL-8B coding performance via llama-server.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("TWIL_MODEL_PATH", str(DEFAULT_MODEL_PATH))),
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--server-bin",
        type=Path,
        default=Path(os.environ.get("LLAMA_SERVER_BIN", str(DEFAULT_SERVER_BIN))),
        help="Path to llama-server binary",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="llama-server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="llama-server port")
    parser.add_argument("--ngl", type=int, default=int(os.environ.get("TWIL_NGL", "26")), help="Number of GPU layers to offload")
    parser.add_argument("--ctx", type=int, default=int(os.environ.get("TWIL_CTX", "4096")), help="Context size")
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("TWIL_MAX_TOKENS", "2048")), help="Max generation tokens")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("TWIL_THREADS", "6")), help="CPU threads for inference")
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("TWIL_TEMP", "0.2")), help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("TWIL_TOP_P", "0.95")), help="Top-p sampling")
    parser.add_argument(
        "--effort",
        choices=["off", "low", "medium", "high"],
        default=os.environ.get("TWIL_EFFORT", "low"),
        help="Reasoning effort: 'off' disables thinking, 'low' sets minimal thinking",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "results_twil.json",
        help="Output JSON results path",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Connect to already running llama-server without starting a new process",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    limits = SafetyLimits(
        min_mem_available_mb=1024,
        min_swap_free_mb=1024,
        min_disk_free_gb=2.0,
        max_gpu_temp_c=88,
    )
    pre_snap = take_snapshot()
    pre_err = check_safety(pre_snap, limits)
    if pre_err:
        print(f"Pre-flight safety check failed: {pre_err}", file=sys.stderr)
        return 1

    print(
        f"[PRE-FLIGHT] RAM avail={pre_snap.mem_available_mb} MiB, "
        f"Swap free={pre_snap.swap_free_mb} MiB, "
        f"Disk free={pre_snap.disk_free_gb} GB, "
        f"GPU used={pre_snap.gpu_mem_used_mb}/{pre_snap.gpu_mem_total_mb} MiB, "
        f"Temp={pre_snap.gpu_temp_c}C"
    )

    proc: subprocess.Popen[str] | None = None
    watchdog: ResourceWatchdog | None = None

    if not args.skip_server:
        if not args.model.exists():
            print(f"Model file not found: {args.model}", file=sys.stderr)
            print("Tip: Download or point to TwIL-8B GGUF with --model /path/to/model.gguf", file=sys.stderr)
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
            "--threads", str(args.threads),
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--temp", str(args.temperature),
            "--top-p", str(args.top_p),
            "--no-warmup",
            "--jinja",
        ]
        print(f"[STARTING] {' '.join(cmd)}")
        log_path = BASE_DIR / "llama_server.log"
        log_fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, text=True)
        watchdog = ResourceWatchdog(proc, limits)
        watchdog.start()

    results: list[dict] = []
    try:
        wait_for_server(proc, watchdog, args.host, args.port, timeout_s=240)
        loaded_snap = take_snapshot()
        print(
            f"[SERVER READY] GPU VRAM={loaded_snap.gpu_mem_used_mb}/{loaded_snap.gpu_mem_total_mb} MiB, "
            f"RAM avail={loaded_snap.mem_available_mb} MiB, Temp={loaded_snap.gpu_temp_c}C"
        )

        for idx, task in enumerate(TASKS):
            if watchdog and watchdog.abort_reason:
                raise RuntimeError(watchdog.abort_reason)
            if idx > 0:
                cool_deadline = time.time() + 25.0
                while time.time() < cool_deadline:
                    s_cool = take_snapshot()
                    if s_cool.gpu_temp_c <= 76:
                        break
                    time.sleep(2.0)

            print(f"\n=== Running TwIL-8B ({args.effort} effort, temp={args.temperature}) on [{task.task_id}] {task.title} ===")
            t0 = time.perf_counter()
            payload: dict = {
                "model": "TwIL-8B",
                "messages": [{"role": "user", "content": task.prompt}],
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
            if args.effort in ("low", "off"):
                payload["chat_template_kwargs"] = (
                    {"enable_thinking": False}
                    if args.effort == "off"
                    else {"reasoning_effort": "low"}
                )

            resp = _http_post_json(
                f"http://{args.host}:{args.port}/v1/chat/completions",
                payload,
                timeout=900.0,
            )
            elapsed = time.perf_counter() - t0
            choice = resp["choices"][0]["message"]
            content = choice.get("content") or ""
            reasoning = choice.get("reasoning_content") or ""
            usage = resp.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)
            tps = round(comp_tokens / elapsed, 2) if elapsed > 0 and comp_tokens else 0.0

            code = extract_python_code(content or reasoning)
            passed, test_out = run_task_tests(code, task)
            snap_now = take_snapshot()

            print(
                f"Result: {'PASS' if passed else 'FAIL'} | "
                f"time={elapsed:.2f}s | completion_tokens={comp_tokens} ({tps} tok/s) | "
                f"VRAM={snap_now.gpu_mem_used_mb} MiB | RAM avail={snap_now.mem_available_mb} MiB | "
                f"Temp={snap_now.gpu_temp_c}C"
            )
            if not passed:
                print(f"Test output: {test_out}")

            results.append(
                {
                    "task_id": task.task_id,
                    "title": task.title,
                    "passed": passed,
                    "test_output": test_out,
                    "elapsed_s": round(elapsed, 2),
                    "usage": usage,
                    "tokens_per_sec": tps,
                    "reasoning_chars": len(reasoning),
                    "reasoning_preview": reasoning[:500],
                    "extracted_code": code,
                    "raw_content": content,
                    "snapshot_after": snap_now.__dict__,
                }
            )
    finally:
        if watchdog:
            watchdog.stop()
        if proc:
            terminate_process(proc)

    summary = {
        "model": "webAI-Official/TwIL-8B (Q4_K_M balanced)",
        "config": {
            "ngl": args.ngl,
            "ctx": args.ctx,
            "max_tokens": args.max_tokens,
            "threads": args.threads,
            "temperature": args.temperature,
            "effort": args.effort,
        },
        "peak_gpu_mb": watchdog.peak_gpu_mb if watchdog else 0,
        "min_mem_available_mb": watchdog.min_mem_avail_mb if watchdog else pre_snap.mem_available_mb,
        "max_gpu_temp_c": watchdog.max_gpu_temp_c if watchdog else pre_snap.gpu_temp_c,
        "abort_reason": watchdog.abort_reason if watchdog else None,
        "tasks_passed": sum(1 for r in results if r["passed"]),
        "tasks_total": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n[SUMMARY] Passed {summary['tasks_passed']}/{summary['tasks_total']} | "
        f"Peak VRAM={summary['peak_gpu_mb']} MiB | "
        f"Min RAM avail={summary['min_mem_available_mb']} MiB | "
        f"Max GPU Temp={summary['max_gpu_temp_c']}C | Saved to {args.output}"
    )
    return 0 if not (watchdog and watchdog.abort_reason) else 2


if __name__ == "__main__":
    raise SystemExit(main())
