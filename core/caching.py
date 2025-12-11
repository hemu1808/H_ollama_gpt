from redis import asyncio as aioredis
import json
import hashlib
from typing import Optional, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

class RedisCache:
    """Production-ready Redis cache with async support"""
    
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.redis_url = redis_url
        self.ttl = ttl
        self._redis = None
    
    async def connect(self):
        if not self._redis:
            self._redis = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50
            )
    
    async def close(self):
        if self._redis:
            await self._redis.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
    async def get(self, key: str) -> Optional[Any]:
        await self.connect()
        value = await self._redis.get(f"rag:{key}")
        return json.loads(value) if value else None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        await self.connect()
        await self._redis.set(
            f"rag:{key}",
            json.dumps(value),
            ex=ttl or self.ttl
        )
    
    async def get_or_create(self, key: str, factory: callable) -> Any:
        value = await self.get(key)
        if value is None:
            value = await factory()
            await self.set(key, value)
        return value

class EmbeddingCache:
    """Thread-safe embedding cache with Redis backend"""
    
    def __init__(self, redis_url: str, max_size: int = 100000):
        self.cache = RedisCache(redis_url)
        self.lock = asyncio.Lock()
    
    def _hash_text(self, text: str) -> str:
        return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
    
    async def get(self, text: str) -> Optional[List[float]]:
        key = self._hash_text(text)
        async with self.lock:
            return await self.cache.get(f"emb:{key}")
    
    async def set(self, text: str, embedding: List[float]):
        key = self._hash_text(text)
        async with self.lock:
            await self.cache.set(f"emb:{key}", embedding, ttl=7200)  # 2 hours
    
    async def get_or_create(self, text: str, embed_fn: callable) -> List[float]:
        cached = await self.get(text)
        if cached is not None:
            return cached
        
        embedding = await embed_fn(text)
        await self.set(text, embedding)
        return embedding