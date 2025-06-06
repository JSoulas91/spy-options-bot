# utils/cache_manager.py

import time
from typing import Any, Dict, Optional


class CacheManager:
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}

    def set(self, key: str, value: Any, ttl: int = 15) -> None:
        """
        Stores a value in the cache with a time-to-live (TTL) in seconds.
        """
        expire_at = time.time() + ttl
        self._cache[key] = {"value": value, "expire_at": expire_at}

    def get(self, key: str) -> Optional[Any]:
        """
        Returns the cached value if it hasn't expired.
        """
        item = self._cache.get(key)
        if item and time.time() < item["expire_at"]:
            return item["value"]
        self._cache.pop(key, None)
        return None

    def clear_expired(self) -> None:
        """
        Removes all expired items from the cache.
        """
        now = time.time()
        self._cache = {
            k: v for k, v in self._cache.items() if v["expire_at"] > now
        }

    def clear_all(self) -> None:
        """
        Clears the entire cache.
        """
        self._cache.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# Singleton cache instance for global use
cache = CacheManager()