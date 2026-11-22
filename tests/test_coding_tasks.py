"""Unit tests for coding task extraction and execution runner."""

from __future__ import annotations

import pytest

from coding_tasks import TASKS, CodingTask, extract_python_code, run_task_tests


def test_extract_python_code_closed_fence() -> None:
    text = """Here is the solution:
```python
def add(a, b):
    return a + b
```
Hope this helps!"""
    assert extract_python_code(text) == "def add(a, b):\n    return a + b"


def test_extract_python_code_unclosed_fence() -> None:
    text = """```python
def add(a, b):
    return a + b"""
    assert extract_python_code(text) == "def add(a, b):\n    return a + b"


def test_extract_python_code_with_think_tags() -> None:
    text = """<think>
Let's think about this carefully.
I need to implement a function.
</think>
```python
def multiply(x, y):
    return x * y
```"""
    assert extract_python_code(text) == "def multiply(x, y):\n    return x * y"


def test_extract_python_code_dangling_backticks() -> None:
    text = """```python
x = 10
y = 20
``"""
    assert extract_python_code(text) == "x = 10\ny = 20"


def test_extract_python_code_raw_fallback() -> None:
    text = "def identity(val):\n    return val"
    assert extract_python_code(text) == "def identity(val):\n    return val"


def test_tasks_registry() -> None:
    assert len(TASKS) >= 3
    task_ids = [t.task_id for t in TASKS]
    assert "weighted_intervals" in task_ids
    assert "ttl_lru_cache" in task_ids
    assert "expr_eval" in task_ids


def test_run_task_tests_pass() -> None:
    task = CodingTask(
        task_id="dummy_pass",
        title="Dummy Pass Task",
        prompt="Implement add(a, b)",
        tests="assert add(2, 3) == 5\nassert add(-1, 1) == 0",
    )
    passed, out = run_task_tests("def add(a, b): return a + b", task)
    assert passed is True


def test_run_task_tests_fail() -> None:
    task = CodingTask(
        task_id="dummy_fail",
        title="Dummy Fail Task",
        prompt="Implement add(a, b)",
        tests="assert add(2, 3) == 5",
    )
    passed, out = run_task_tests("def add(a, b): return a * b", task)
    assert passed is False
    assert "AssertionError" in out
