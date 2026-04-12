# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import wraps
from threading import RLock

logger = logging.getLogger(__name__)

# Keep a module-level cache shared by all decorated functions in the current
# process. This avoids import-time process creation on Windows and still gives
# us the retry-deduping behavior the callers rely on.
global_cache = {}
lock = RLock()


def multiprocessing_cache(max_calls):
    """
    A decorator that creates a global cache shared between multiple functions
    within the current process. The cache is invalidated after `max_calls`
    number of accesses for the decorated function.

    Args:
        max_calls (int): The number of calls after which the cache is cleared.

    Returns:
        function: The decorated function with global cache and invalidation logic.
    """

    def decorator(func):
        call_count = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal call_count
            key = (func.__name__, args, frozenset(kwargs.items()))

            with lock:
                call_count += 1

                if call_count > max_calls:
                    global_cache.clear()
                    call_count = 0

                if key in global_cache:
                    return global_cache[key]

            result = func(*args, **kwargs)

            with lock:
                global_cache[key] = result

            return result

        return wrapper

    return decorator
