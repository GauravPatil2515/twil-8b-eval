"""Verify reference solutions for benchmark tasks."""

from __future__ import annotations

import pytest

from coding_tasks import TASKS, run_task_tests
from gemini_flash_eval import GEMINI_SOLUTIONS


@pytest.mark.parametrize("task", TASKS, ids=[t.task_id for t in TASKS])
def test_reference_solution_passes(task) -> None:
    code = GEMINI_SOLUTIONS[task.task_id]
    passed, output = run_task_tests(code, task, timeout_s=10)
    assert passed is True, f"Reference solution failed for {task.task_id}: {output}"
