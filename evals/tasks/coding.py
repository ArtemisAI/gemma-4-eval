"""Code generation eval tasks.

Complex, production-grade coding tasks with real-world constraints.
Tests whether the model can produce complete, runnable implementations
(not just snippets) under heavy thinking load.
"""


def get_tasks() -> list[dict]:
    return [
        {
            "name": "async_scraper_with_retries",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a Python async web scraper class with these requirements:\n"
                        "1. Use aiohttp with connection pooling (max 10 concurrent)\n"
                        "2. Exponential backoff retry (3 attempts, base 1s, jitter)\n"
                        "3. Rate limiting: max 5 requests/second per domain\n"
                        "4. robots.txt respect (parse and cache per domain)\n"
                        "5. Results saved to SQLite with deduplication by URL hash\n"
                        "6. Graceful shutdown on SIGINT/SIGTERM\n"
                        "7. Structured logging with request timing\n"
                        "Provide the complete implementation, not pseudocode."
                    ),
                }
            ],
        },
        {
            "name": "concurrent_lru_cache",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Implement a thread-safe LRU cache in Python with:\n"
                        "1. O(1) get and put operations\n"
                        "2. TTL (time-to-live) per entry\n"
                        "3. Max memory limit (not just count) with sys.getsizeof estimation\n"
                        "4. Eviction callback support\n"
                        "5. Cache statistics (hits, misses, evictions, hit rate)\n"
                        "6. Context manager support for batch operations\n"
                        "Include unit tests covering concurrent access, TTL expiry, "
                        "memory-based eviction, and edge cases."
                    ),
                }
            ],
        },
        {
            "name": "distributed_task_queue",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Design and implement a minimal distributed task queue in Python using only "
                        "the standard library plus redis-py. Requirements:\n"
                        "1. Producer: submit tasks with priority, retry policy, and timeout\n"
                        "2. Worker: claim tasks atomically using Redis BRPOPLPUSH pattern\n"
                        "3. Dead letter queue for tasks exceeding max retries\n"
                        "4. Heartbeat mechanism: workers report health every 10s, "
                        "   tasks from dead workers get re-queued\n"
                        "5. Result backend: store task results with configurable TTL\n"
                        "6. CLI interface: submit, status, retry-dead, stats\n"
                        "Provide complete, runnable code with docstrings."
                    ),
                }
            ],
        },
    ]
