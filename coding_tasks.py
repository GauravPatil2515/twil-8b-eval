"""Self-verifying Python coding benchmark tasks and sandboxed test runner."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodingTask:
    """A single coding benchmark prompt paired with its automated test suite."""

    task_id: str
    title: str
    prompt: str
    tests: str


TASKS: list[CodingTask] = [
    CodingTask(
        task_id="weighted_intervals",
        title="Weighted Non-Overlapping Interval Scheduling",
        prompt=textwrap.dedent("""\
            Write a Python function `max_weighted_schedule(intervals: list[tuple[int, int, int]]) -> int`
            that takes a list of intervals `(start, end, weight)` where `start < end` and `weight >= 0`.
            Two intervals `[s1, e1)` and `[s2, e2)` are non-overlapping if `e1 <= s2` or `e2 <= s1`.
            Return the maximum total weight achievable by selecting a non-overlapping subset.
            Requirements:
            - Must run in O(n log n) time (supports n = 50,000+ easily).
            - Return 0 for an empty list.
            Put only the valid Python implementation inside a single ```python``` code block.
        """),
        tests=textwrap.dedent("""\
            assert max_weighted_schedule([]) == 0
            assert max_weighted_schedule([(1, 3, 5)]) == 5
            assert max_weighted_schedule([(1, 3, 5), (2, 5, 6), (4, 6, 5), (6, 7, 4)]) == 14
            assert max_weighted_schedule([(1, 2, 50), (3, 5, 20), (6, 19, 100), (2, 100, 200)]) == 250
            assert max_weighted_schedule([(0, 5, 10), (0, 5, 12), (0, 5, 8)]) == 12
            # Stress test n = 50000
            big = [(i, i + 2, 1 + (i % 7)) for i in range(50000)]
            res = max_weighted_schedule(big)
            assert res > 100000
            print("PASS weighted_intervals")
        """),
    ),
    CodingTask(
        task_id="ttl_lru_cache",
        title="O(1) LRU Cache with Per-Key TTL",
        prompt=textwrap.dedent("""\
            Implement a Python class `TTLLRUCache`:
            - `__init__(self, capacity: int, default_ttl: float)`: `capacity >= 1`, `default_ttl > 0`.
            - `get(self, key: str, now: float) -> object | None`:
              Return the value if `key` exists and is not expired (`now < expires_at`), updating it as most-recently used.
              If expired or missing, remove if expired and return `None`.
            - `put(self, key: str, value: object, now: float, ttl: float | None = None) -> None`:
              Insert or update `key` with expiration `now + (ttl if ttl is not None else default_ttl)` and mark as most-recently used.
              If inserting a new key at full capacity, evict expired entries first (if any), otherwise evict the least-recently used entry.
            Use `collections.OrderedDict` or a custom doubly linked list so operations stay O(1) amortized (except scanning expired on full capacity if needed, or clean O(1)).
            Put only the valid Python implementation inside a single ```python``` code block.
        """),
        tests=textwrap.dedent("""\
            c = TTLLRUCache(2, default_ttl=10.0)
            c.put("a", 1, now=0.0)
            c.put("b", 2, now=1.0, ttl=2.0)  # expires at 3.0
            assert c.get("a", now=2.0) == 1  # "a" is now MRU, "b" is LRU
            assert c.get("b", now=3.0) is None  # exact expiry now >= expires_at
            c.put("c", 3, now=3.0)  # "b" expired, "a" must remain!
            assert c.get("a", now=4.0) == 1
            assert c.get("c", now=4.0) == 3
            c.put("d", 4, now=5.0)  # full: "a" is LRU (accessed at 4.0 before "c")
            assert c.get("a", now=5.0) is None
            assert c.get("c", now=5.0) == 3
            assert c.get("d", now=5.0) == 4
            # Update existing key refreshes value, TTL, and recency
            c.put("c", 30, now=6.0, ttl=1.0)  # expires at 7.0
            c.put("e", 5, now=7.5)  # "c" expired at 7.0, so "d" stays
            assert c.get("c", now=7.5) is None
            assert c.get("d", now=7.5) == 4
            assert c.get("e", now=7.5) == 5
            print("PASS ttl_lru_cache")
        """),
    ),
    CodingTask(
        task_id="expr_eval",
        title="Safe Arithmetic & Variable Assignment Evaluator",
        prompt=textwrap.dedent("""\
            Implement a Python function `eval_program(program: str) -> int | float` WITHOUT using `eval`, `exec`, or `ast.literal_eval`.
            Specification:
            - `program` consists of one or more `;`-separated statements (trailing `;` allowed).
            - Each statement is either `identifier = expr` or `expr`.
            - Expressions support: integer and float literals, variable names (`[a-zA-Z_][a-zA-Z0-9_]*`),
              parentheses `( ... )`, unary `+` / `-`, binary `+`, `-`, `*`, `/`, `//`, `%`, `**` (right-associative!).
            - Standard Python operator precedence: `()` > `**` (right-assoc) > unary `+/-` > `* / // %` > `+ -`.
              Note: in Python `-2 ** 2 == -4` (unary binds looser than `**`).
            - Return the value of the last non-empty statement.
            - Raise `ValueError` on syntax errors, unknown variables, or division by zero.
            Put only the valid Python implementation inside a single ```python``` code block.
        """),
        tests=textwrap.dedent("""\
            assert eval_program("2 + 3 * 4") == 14
            assert eval_program("(2 + 3) * 4") == 20
            assert eval_program("2 ** 3 ** 2") == 512
            assert eval_program("-2 ** 2") == -4
            assert eval_program("(-2) ** 2") == 4
            assert eval_program("x = 10; y = x * 2 + 3; y // 4;") == 5
            assert abs(eval_program("a = 5.5; b = 2; a / b + 1.25") - 4.0) < 1e-9
            assert eval_program("x = 10 % 3; -x + +5") == 4
            for bad in ["", "unknown + 1", "1 / 0", "1 + * 2", "x = "]:
                try:
                    eval_program(bad)
                    raise AssertionError(f"Expected ValueError for {bad!r}")
                except ValueError:
                    pass
            print("PASS expr_eval")
        """),
    ),
]


def extract_python_code(text: str) -> str:
    """Extract Python code from markdown text, handling closed, unclosed, or raw blocks."""
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # 1. Match fully closed fenced code block
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # 2. Match unclosed fenced code block (e.g. truncated at max_tokens limit)
    unclosed = re.findall(r"```(?:python)?\s*\n(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if unclosed:
        candidate = unclosed[-1].strip()
        candidate = re.sub(r"```*$", "", candidate).strip()
        if candidate:
            return candidate

    # 3. Strip any stray markdown backticks
    cleaned = text.strip()
    if cleaned.startswith("```python"):
        cleaned = cleaned[9:].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


def run_task_tests(code: str, task: CodingTask, timeout_s: int = 10) -> tuple[bool, str]:
    """Execute generated code + unit tests in an isolated subprocess."""
    script = f"{code}\n\n# --- UNIT TESTS ---\n{task.tests}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(script)
        tmp_path = Path(tmp.name)

    try:
        proc = subprocess.run(
            [sys.executable, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = (proc.stdout + "\n" + proc.stderr).strip()
        return proc.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout_s}s"
    finally:
        tmp_path.unlink(missing_ok=True)

