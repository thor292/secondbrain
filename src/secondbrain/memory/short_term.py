"""Short-term memory in Redis: a per-user working buffer that expires.

It holds raw items until consolidation moves them into long-term memory.
Items are only ever read back by the user who wrote them.
"""

import json
import time

import redis


class ShortTermMemory:
    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        # No Redis configured: keep it in this process. Fine for one instance; with
        # several, a person's recent chat may land on another instance and be missed.
        self._redis = (redis.Redis.from_url(redis_url, decode_responses=True)
                       if redis_url else _LocalLists())
        self._ttl = ttl_seconds

    @staticmethod
    def _key(user_id: str) -> str:
        return f"sb:short_term:{user_id}"

    def push(self, user_id: str, item: dict) -> None:
        key = self._key(user_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, json.dumps({**item, "at": time.time()}))
        pipe.expire(key, self._ttl)
        pipe.execute()

    def recent(self, user_id: str, limit: int = 10) -> list[dict]:
        return [json.loads(raw) for raw in self._redis.lrange(self._key(user_id), -limit, -1)]

    def all(self, user_id: str) -> list[dict]:
        return [json.loads(raw) for raw in self._redis.lrange(self._key(user_id), 0, -1)]

    def drop_oldest(self, user_id: str, count: int) -> None:
        """Remove items that have been consolidated; newer ones stay."""
        self._redis.ltrim(self._key(user_id), count, -1)

    def clear(self, user_id: str) -> None:
        self._redis.delete(self._key(user_id))

    # Conversation history: recent question/answer turns, so follow-up
    # questions ("and for express?") can be understood. Kept apart from notes
    # and never consolidated into long-term memory.
    @staticmethod
    def _chat_key(user_id: str) -> str:
        return f"sb:chat:{user_id}"

    def add_turn(self, user_id: str, question: str, answer: str, keep: int = 10) -> None:
        key = self._chat_key(user_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, json.dumps({"question": question, "answer": answer, "at": time.time()}))
        pipe.ltrim(key, -keep, -1)
        pipe.expire(key, self._ttl)
        pipe.execute()

    def history(self, user_id: str, limit: int = 6) -> list[dict]:
        return [json.loads(raw) for raw in self._redis.lrange(self._chat_key(user_id), -limit, -1)]

    def clear_history(self, user_id: str) -> None:
        self._redis.delete(self._chat_key(user_id))


class _LocalLists:
    """The few Redis list commands ShortTermMemory uses, kept in memory (TTL ignored)."""

    def __init__(self) -> None:
        import threading
        self._data: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def pipeline(self) -> "_LocalLists":
        return self

    def execute(self) -> list:
        return []

    def rpush(self, key: str, value: str) -> None:
        with self._lock:
            self._data.setdefault(key, []).append(value)

    def expire(self, key: str, seconds: int) -> None:
        pass

    def _slice(self, items: list[str], start: int, end: int) -> list[str]:
        n = len(items)
        start = max(start + n, 0) if start < 0 else start
        end = end + n if end < 0 else end
        return items[start:end + 1]

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        with self._lock:
            return list(self._slice(self._data.get(key, []), start, end))

    def ltrim(self, key: str, start: int, end: int) -> None:
        with self._lock:
            self._data[key] = self._slice(self._data.get(key, []), start, end)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def flushdb(self) -> None:
        with self._lock:
            self._data.clear()
