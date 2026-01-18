import time
import logging
from typing import Optional
from fastapi import Request, HTTPException
from redis import Redis, ConnectionError

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, redis_url: str = "redis://localhost:6379", limit: int = 10, window: int = 60):
        self.limit = limit
        self.window = window
        self.redis_url = redis_url
        self.redis: Optional[Redis] = None
        
        try:
            self.redis = Redis.from_url(redis_url, decode_responses=True)
            self.redis.ping()
            logger.info(f"RateLimiter connected to Redis at {redis_url}")
        except (ConnectionError, Exception) as e:
            logger.warning(f"RateLimiter could not connect to Redis: {e}. Rate limiting will be disabled.")
            self.redis = None

    async def __call__(self, request: Request):
        """
        Pure Dependency implementation.
        Checks limits and returns. Does NOT call next middleware.
        """
        if not self.redis:
            return

        client_ip = request.client.host if request.client else "127.0.0.1"
        key = f"rate_limit:{client_ip}"

        try:
            current_requests = self.redis.incr(key)
            if current_requests == 1:
                self.redis.expire(key, self.window)
            
            if current_requests > self.limit:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                raise HTTPException(
                    status_code=429, 
                    detail=f"Too many requests. Limit is {self.limit} per {self.window} seconds."
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"RateLimiter Redis error: {e}")
            # Fail open
            return