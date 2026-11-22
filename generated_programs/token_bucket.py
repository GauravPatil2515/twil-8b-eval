```python
import time
from typing import Dict, Tuple

class TokenBucketLimiter:
    """
    Deterministic per-key token bucket limiter.

    - `allow(key, now, tokens=1.0)` returns True if enough tokens are available
      at time `now`, consuming `tokens` and updating the bucket state.
    - `retry_after(key, now, tokens=1.0)` returns the minimum seconds to wait
      until enough tokens become available (0.0 if already available).
    """

    def __init__(self, rate_per_sec: float, capacity: float):
        """
        Initialize the limiter.

        Args:
            rate_per_sec: Tokens added per second (rate of refill).
            capacity: Maximum tokens the bucket can hold.
        """
        if rate_per_sec < 0 or capacity < 0:
            raise ValueError("rate_per_sec and capacity must be non-negative")
        self.rate = rate_per_sec
        self.capacity = capacity
        # state: key -> (last_now, current_tokens)
        self.state: Dict[str, Tuple[float, float]] = {}

    def _get_state(self, key: str) -> Tuple[float, float]:
        """Return (last_now, current_tokens) for key, creating if missing."""
        entry = self.state.get(key)
        if entry is None:
            # New key: start full at capacity
            self.state[key] = (now, self.capacity)
            return (now, self.capacity)
        last_now, tokens = entry
        return entry

    def allow(self, key: str, now: float, tokens: float = 1.0) -> bool:
        """
        Attempt to consume `tokens` from the bucket for `key` at time `now`.

        Refills tokens based on elapsed time since the last check for this key.
        Returns True if enough tokens were available and consumed; False otherwise.
        """
        if tokens <= 0:
            # No consumption needed; consider always allowed
            return True

        last_now, cur_tokens = self._get_state(key)
        elapsed = now - last_now
        # Refill up to capacity
        new_tokens = min(self.capacity, cur_tokens + elapsed * self.rate)
        if new_tokens < 0:
            new_tokens = 0.0  # safety clamp (should not happen with non-negative rate)

        if new_tokens >= tokens:
            # Consume tokens
            self.state[key] = (now, new_tokens - tokens)
            return True
        else:
            # Not enough tokens; do not consume and do not update state?
            # According to typical token bucket semantics, we do not consume if insufficient.
            # We keep the bucket state as after refill but before consumption? 
            # The spec says "consumes tokens if available". So if not available, we do not consume.
            # We should still update the state to reflect the refill? Usually yes, because time passes.
            # We'll update state to (now, new_tokens) to reflect current bucket after refill.
            self.state[key] = (now, new_tokens)
            return False

    def retry_after(self, key: str, now: float, tokens: float = 1.0) -> float:
        """
        Return seconds to wait until enough tokens are available for `key` at time `now`.

        If enough tokens are already available, returns 0.0.
        """
        if tokens <= 0:
            return 0.0

        last_now, cur_tokens = self._get_state(key)
        elapsed = now - last_now
        # Current tokens after refill
        current_tokens = min(self.capacity, cur_tokens + elapsed * self.rate)

        if current_tokens >= tokens:
            return 0.0

        # Need to wait for enough tokens to accumulate.
        # Solve: cur_tokens + (now + wait) * rate >= capacity? Actually we need:
        # current_tokens + wait * rate >= tokens  (since we start from current_tokens at now)
        # But careful: after waiting `wait`, the bucket will have current_tokens + wait * rate,
        # but cannot exceed capacity. So the required wait is max(0, (tokens - current_tokens) / rate)
        # provided rate > 0. If rate == 0, then if current_tokens < tokens, it will never reach -> infinite wait?
        # However, spec likely assumes rate > 0. We'll handle rate == 0: if not enough now, wait is infinite?
        # But we must return a finite float. We'll return a large number or raise? The spec says deterministic.
        # We'll assume rate > 0 for meaningful retry_after. If rate == 0, then if not enough now,
        # retry_after should be infinite? But we cannot return inf. We'll return a very large number or
        # raise? The problem statement doesn't specify. We'll assume rate > 0.
        if self.rate == 0:
            # No refill; if not enough now, it will never be enough -> return a large number?
            # But to be deterministic and finite, we can return a sentinel like 1e18.
            # However, typical token bucket with rate 0 is just a fixed capacity limiter.
            # retry_after would be infinite if not enough now. We'll return a large number.
            # But the spec likely expects rate > 0. We'll just compute normally; division by zero avoided.
            # Actually if rate == 0, then (tokens - current_tokens) / rate is undefined.
            # So we need to handle separately.
            if current_tokens >= tokens:
                return 0.0
            else:
                # No refill possible, so wait forever -> return a very large number
                # We'll use a large constant, e.g., 1e18 seconds.
                return 1e18

        need = tokens - current_tokens
        wait = need / self.rate
        # Because tokens cannot exceed capacity, the actual time to reach capacity might be less,
        # but we only need enough tokens, so wait computed as above is correct.
        # However, if capacity is less than tokens? tokens is the amount we want to consume now.
        # But tokens <= capacity typically? Not necessarily; but if tokens > capacity, then even full bucket
        # cannot satisfy, so retry_after would be infinite. We'll handle that.
        if tokens > self.capacity:
            # Even a full bucket cannot provide tokens; wait forever
            return 1e18
