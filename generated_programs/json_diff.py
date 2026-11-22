```python
import json
from typing import Any, Dict, List

def diff_values(old: Any, new: Any, path: str = '') -> List[Dict[str, Any]]:
    """
    Recursively diff two JSON-like Python values (dict, list, primitives) and return
    a list of change records with keys: 'path', 'kind' ('added', 'removed', 'changed'),
    'old', and 'new'.

    Path uses dot notation for dict keys and [idx] for list indices.
    """
    changes = []

    # Helper to build full path
    def full_path(p: str, key_part: str) -> str:
        if not p:
            return key_part
        if p.endswith('.'):
            return p + key_part
        return p + '.' + key_part if p else key_part

    # Primitive comparison (including None, bool, int, float, str)
    if old == new:
        return changes

    # Type change (including None vs non-None, str vs int, etc.)
    if type(old) != type(new):
        # Record removal of old and addition of new at current path
        p = path if path else ''
        changes.append({
            'path': p,
            'kind': 'removed',
            'old': old,
            'new': None
        })
        changes.append({
            'path': p,
            'kind': 'added',
            'old': None,
            'new': new
        })
        return changes

    # Both are dicts
    if isinstance(old, dict) and isinstance(new, dict):
        # Collect keys from both
        all_keys = set(old.keys()) | set(new.keys())
        for key in all_keys:
            old_val = old.get(key)
            new_val = new.get(key)
            # Determine if key exists in old/new
            old_exists = key in old
            new_exists = key in new
            # Build child path
            child_path = f"{path}.{key}" if path else key
            # Recursively diff
            child_changes = diff_values(old_val if old_exists else None,
                                        new_val if new_exists else None,
                                        child_path)
            changes.extend(child_changes)
            # If key only in new, record added; if only in old, record removed
            if not old_exists and new_exists:
                changes.append({
                    'path': child_path,
                    'kind': 'added',
                    'old': None,
                    'new': new_val
                })
            elif old_exists and not new_exists:
                changes.append({
                    'path': child_path,
                    'kind': 'removed',
                    'old': old_val,
                    'new': None
                })
        return changes

    # Both are lists
    if isinstance(old, list) and isinstance(new, list):
        # Compare up to min length
        min_len = min(len(old), len(new))
        for idx in range(min_len):
            child_path = f"{path}[{idx}]" if path else f"[{idx}]"
            child_changes = diff_values(old[idx], new[idx], child_path)
            changes.extend(child_changes)
        # Handle extra elements in either list
        if len(old) > len(new):
            for idx in range(len(new), len(old)):
                child_path = f"{path}[{idx}]" if path else f"[{idx}]"
                changes.append({
                    'path': child_path,
                    'kind': 'removed',
                    'old': old[idx],
                    'new': None
                })
        elif len(new) > len(old):
            for idx in range(len(old), len(new)):
                child_path = f"{path}[{idx}]" if path else f"[{idx}]"
                changes.append({
                    'path': child_path,
                    'kind': 'added',
                    'old': None,
                    'new': new[idx]
                })
        return changes

    # Should not reach here due to type checks above, but fallback
    # Compare as primitives (already handled equality earlier)
    if old != new:
        changes.append({
            'path': path if path else '',
            'kind': 'changed',
            'old': old,
            'new': new
        })
    return changes


if __name__ == '__main__':
    # Test 1: identical inputs
    assert diff_values({'a': 1, 'b': [1, 2]}, {'a': 1, 'b': [1, 2]}) == []

    # Test 2: nested dict changes
    d1 = {'a': {'b': 1}, 'c': 2}
    d2 = {'a': {'b': 2}, 'c': 2}
    result = diff_values(d1, d2)
    assert len(result) == 1
    assert any(r['kind'] == 'changed' and r['path'] == 'a.b' for r in result)

    # Test 3: list length difference
    l1 = [1, 2, 3]
    l2 = [1, 2]
    result = diff_values(l1, l2)
    assert len(result) == 1
    assert any(r['kind'] == 'removed' and r['path'] == '[2]' for r in result)

    # Test 4: type change
    t1 = {'x': 1}
    t2 = {'x': '1'}
    result = diff_values(t1, t2)
    assert len(result) == 2  # one removed, one added
    kinds = {r['kind'] for r in result}
    assert kinds == {'removed', 'added'}

    # Test 5: list with nested dict changes
    l3 = [{'a': 1}, {'b': 2}]
    l4 = [{'a': 2}, {'b': 2}]
    result = diff_values(l3, l4)
    assert len(result) == 1
    assert any(r['kind'] == 'changed' and r['path'] == '0.a' for r in result)

    # Test 6: empty dict vs dict with key
    e1 = {}
    e2 = {'key': 'val'}
    result = diff_values(e1, e
