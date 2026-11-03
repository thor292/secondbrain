"""Short-term memory in Redis: a per-user working buffer that expires.

It holds raw items until consolidation moves them into long-term memory.
Items are only ever read back by the user who wrote them.
"""

import json
import time

import redis


class ShortTermMemory:
    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
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
