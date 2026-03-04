import time
import threading
from functools import wraps
from typing import Callable, Any

class CircuitBreakerOpenException(Exception):
    pass

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.lock = threading.Lock()

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with self.lock:
                if self.state == "OPEN":
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        self.state = "HALF-OPEN"
                    else:
                        raise CircuitBreakerOpenException("Circuit is open. Request blocked.")

            try:
                result = func(*args, **kwargs)
                with self.lock:
                    if self.state == "HALF-OPEN":
                        self.state = "CLOSED"
                        self.failures = 0
                return result
            except Exception as e:
                with self.lock:
                    self.failures += 1
                    self.last_failure_time = time.time()
                    if self.failures >= self.failure_threshold:
                        self.state = "OPEN"
                raise e
        return wrapper