```python
from collections import defaultdict, deque
from typing import Dict, List

def schedule_waves(deps: Dict[str, List[str]]) -> List[List[str]]:
    """
    Schedule tasks into parallel waves based on dependencies.
    Each wave contains tasks that can start simultaneously (all their prerequisites are finished in previous waves).
    Tasks within each wave are sorted alphabetically.
    Raises ValueError if there is a dependency cycle or a missing prerequisite task.
    """
    # Build reverse adjacency for topological sort and detect cycles
    graph = defaultdict(list)
    indegree = defaultdict(int)
    all_tasks = set(deps.keys())
    
    # Check for missing prerequisites
    for task, prereqs in deps.items():
        for prereq in prereqs:
            if prereq not in all_tasks:
                raise ValueError(f"Missing prerequisite task: {prereq}")
            graph[prereq].append(task)
            indegree[task] += 1
        # Tasks with no dependencies will have indegree 0 automatically
    
    # Ensure every task appears in indegree (tasks with no deps)
    for task in all_tasks:
        if task not in indegree:
            indegree[task] = 0
    
    # Kahn's algorithm for topological sort and cycle detection
    q = deque([task for task in all_tasks if indegree[task] == 0])
    topo_order = []
    
    while q:
        node = q.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    
    # If we didn't visit all tasks, there's a cycle
    if len(topo_order) != len(all_tasks):
        # Find remaining nodes (those still with indegree > 0)
        remaining = set(all_tasks) - set(topo_order)
        raise ValueError(f"Dependency cycle involving tasks: {remaining}")
    
    # Build waves: tasks that become available at the same level
    waves = []
    current_wave = []
    
    for task in topo_order:
        # If all prerequisites of task are already scheduled in previous waves,
        # it can be added to the current wave.
        # We check by verifying that all its deps appear before it in topo_order.
        # Since topo_order is a valid linearization, we can simply group by level:
        # A task belongs to wave i if all its deps are in waves < i.
        # We can compute level by longest path from sources.
        # Simpler: iterate topo_order and assign wave number as max(wave of deps)+1.
        # We'll compute level on the fly.
        pass
    
    # Re-implement level computation:
    level = {task: 0 for task in all_tasks}
    # Process in topological order to compute longest path length from sources
    for task in topo_order:
        # level[task] already computed from its deps (since deps come before task in topo_order)
        # Actually we need to compute level as max(level[dep])+1
        # We'll recompute properly:
        # Let's compute level using DP on topo_order
        pass
    
    # Clear level dict and recompute
    level = {}
    for task in topo_order:
        if task not in deps or not deps[task]:
            level[task] = 0
        else:
            level[task] = max(level[dep] for dep in deps[task]) + 1
    
    # Group by level
    waves_dict = defaultdict(list)
    for task in all_tasks:
        waves_dict[level[task]].append(task)
    
    # Sort levels and sort each wave alphabetically
    sorted_levels = sorted(waves_dict.keys())
    result = []
    for lvl in sorted_levels:
        wave = sorted(waves_dict[lvl])
        result.append(wave)
    
    return result


if __name__ == '__main__':
    # Test 1: Diamond shape
    deps1 = {
        'a': [],
        'b': ['a'],
        'c': ['a'],
        'd': ['b', 'c']
    }
    waves1 = schedule_waves(deps1)
    # Expected: [['a'], ['b', 'c'], ['d']] (alphabetical within wave)
    assert waves1 == [['a'], ['b', 'c'], ['d']], f"Diamond test failed: {waves1}"
    
    # Test 2: Independent components
    deps2 = {
        'x': [],
        'y': [],
        'z': []
    }
    waves2 = schedule_waves(deps2)
    # All can be in same wave, sorted
    assert sorted(waves2[0]) == ['x', 'y', 'z'], f"Independent test failed: {waves2}"
    
    # Test 3: Empty input
    deps3 = {}
    waves3 = schedule_waves(deps3)
    assert waves3 == [], f"Empty test failed: {waves3}"
    
    # Test 4: Missing prerequisite
    deps4 = {
        'task1': ['missing']
    }
    try:
        schedule_waves(deps4)
        assert False, "Missing prerequisite should raise ValueError"
    except ValueError as e:
        assert "Missing prerequisite task" in str(e), f"Wrong error message: {e}"
    
    # Test 5: Simple cycle
    deps5 = {
        'a': ['b'],
        'b': ['a']
    }
    try:
        schedule_waves(deps5)
        assert False, "Cycle should raise ValueError"
    except ValueError as e:
        assert "Dependency cycle" in str(e), f"Wrong error message: {e}"
    
    # Additional test: chain
    deps6 = {
        'a': [],
        'b': ['a'],
        'c': ['b'],
        'd': ['c']
    }
    waves6 = schedule_waves(deps6)
    assert waves6 == [['a'], ['b'], ['c'], ['d']], f"Chain test failed: {waves6}"
    
    # Additional test: two independent chains merging
    deps7 = {
        'a': [],
        'b': [],
        'c': ['a', 'b'],
        'd': ['a'],
        'e': ['b']
    }
    waves7 = schedule_waves(deps7)
    # Possible waves: [['a','b'], ['c','d','e']] or [['a','b'], ['d','e
