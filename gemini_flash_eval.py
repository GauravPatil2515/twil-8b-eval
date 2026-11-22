"""Verify the coding benchmark suite using Gemini 3.8 Flash solutions."""

from __future__ import annotations

import json
import time
from pathlib import Path

from coding_tasks import TASKS, run_task_tests

GEMINI_SOLUTIONS: dict[str, str] = {
    "weighted_intervals": '''
from bisect import bisect_right

def max_weighted_schedule(intervals: list[tuple[int, int, int]]) -> int:
    if not intervals:
        return 0
    sorted_iv = sorted(intervals, key=lambda x: x[1])
    ends = [iv[1] for iv in sorted_iv]
    n = len(sorted_iv)
    dp = [0] * (n + 1)
    for i, (start, _end, weight) in enumerate(sorted_iv, start=1):
        prev_idx = bisect_right(ends, start, 0, i - 1)
        dp[i] = max(dp[i - 1], dp[prev_idx] + weight)
    return dp[n]
'''.strip(),
    "ttl_lru_cache": '''
from collections import OrderedDict

class TTLLRUCache:
    def __init__(self, capacity: int, default_ttl: float) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if default_ttl <= 0:
            raise ValueError("default_ttl must be > 0")
        self.capacity = capacity
        self.default_ttl = default_ttl
        self._items: OrderedDict[str, tuple[object, float]] = OrderedDict()

    def get(self, key: str, now: float) -> object | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        val, expires_at = entry
        if now >= expires_at:
            del self._items[key]
            return None
        self._items.move_to_end(key)
        return val

    def put(self, key: str, value: object, now: float, ttl: float | None = None) -> None:
        effective_ttl = self.default_ttl if ttl is None else ttl
        expires_at = now + effective_ttl
        if key in self._items:
            self._items[key] = (value, expires_at)
            self._items.move_to_end(key)
            return
        if len(self._items) >= self.capacity:
            expired_keys = [k for k, (_v, exp) in self._items.items() if now >= exp]
            for k in expired_keys:
                del self._items[k]
            while len(self._items) >= self.capacity:
                self._items.popitem(last=False)
        self._items[key] = (value, expires_at)
'''.strip(),
    "expr_eval": '''
import re

_TOKEN_RE = re.compile(
    r"\\s*(?:(?P<NUM>(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)|"
    r"(?P<ID>[A-Za-z_][A-Za-z0-9_]*)|"
    r"(?P<OP>\\*\\*|//|[+\\-*/%()=;]))"
)

def eval_program(program: str) -> int | float:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(program):
        if program[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(program, pos)
        if not m:
            raise ValueError(f"Unexpected character at {pos}: {program[pos]!r}")
        kind = m.lastgroup
        assert kind is not None
        tokens.append((kind, m.group(kind)))
        pos = m.end()

    # Split statements on ';'
    stmts: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    for tok in tokens:
        if tok == ("OP", ";"):
            if cur:
                stmts.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        stmts.append(cur)
    if not stmts:
        raise ValueError("Empty program")

    env: dict[str, int | float] = {}

    def eval_tokens(st: list[tuple[str, str]]) -> int | float:
        idx = 0

        def peek() -> tuple[str, str] | None:
            return st[idx] if idx < len(st) else None

        def consume() -> tuple[str, str]:
            nonlocal idx
            tok = st[idx]
            idx += 1
            return tok

        def parse_expr() -> int | float:
            val = parse_term()
            while (t := peek()) in (("OP", "+"), ("OP", "-")):
                consume()
                rhs = parse_term()
                val = val + rhs if t[1] == "+" else val - rhs
            return val

        def parse_term() -> int | float:
            val = parse_unary()
            while (t := peek()) in (("OP", "*"), ("OP", "/"), ("OP", "//"), ("OP", "%")):
                consume()
                rhs = parse_unary()
                op = t[1]
                if op == "*":
                    val = val * rhs
                else:
                    if rhs == 0:
                        raise ValueError("Division by zero")
                    if op == "/":
                        val = val / rhs
                    elif op == "//":
                        val = val // rhs
                    else:
                        val = val % rhs
            return val

        def parse_unary() -> int | float:
            t = peek()
            if t in (("OP", "+"), ("OP", "-")):
                consume()
                val = parse_unary()
                return val if t[1] == "+" else -val
            return parse_power()

        def parse_power() -> int | float:
            base = parse_primary()
            if peek() == ("OP", "**"):
                consume()
                exp = parse_unary()
                return base ** exp
            return base

        def parse_primary() -> int | float:
            t = peek()
            if t is None:
                raise ValueError("Unexpected end of expression")
            kind, text = consume()
            if kind == "NUM":
                return float(text) if ("." in text or "e" in text or "E" in text) else int(text)
            if kind == "ID":
                if text not in env:
                    raise ValueError(f"Unknown variable: {text}")
                return env[text]
            if (kind, text) == ("OP", "("):
                val = parse_expr()
                if peek() != ("OP", ")"):
                    raise ValueError("Missing closing parenthesis")
                consume()
                return val
            raise ValueError(f"Unexpected token: {text!r}")

        res = parse_expr()
        if idx != len(st):
            raise ValueError(f"Unexpected trailing token: {st[idx]!r}")
        return res

    last_val: int | float = 0
    for st in stmts:
        if len(st) >= 2 and st[0][0] == "ID" and st[1] == ("OP", "="):
            var_name = st[0][1]
            rhs_tokens = st[2:]
            if not rhs_tokens:
                raise ValueError("Missing expression after '='")
            last_val = eval_tokens(rhs_tokens)
            env[var_name] = last_val
        else:
            last_val = eval_tokens(st)
    return last_val
'''.strip(),
}


def main() -> int:
    results = []
    for task in TASKS:
        code = GEMINI_SOLUTIONS[task.task_id]
        t0 = time.perf_counter()
        passed, out = run_task_tests(code, task)
        dt = time.perf_counter() - t0
        print(f"[Gemini 3.8 Flash] {task.task_id}: {'PASS' if passed else 'FAIL'} ({dt:.3f}s) -> {out}")
        results.append({"task_id": task.task_id, "passed": passed, "test_output": out, "test_time_s": round(dt, 4)})
    out_path = Path(__file__).resolve().parent / "results_gemini_flash.json"
    out_path.write_text(json.dumps({"model": "Gemini 3.8 Flash", "results": results}, indent=2), encoding="utf-8")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

