from __future__ import annotations

import time
from typing import Dict, Optional, Set
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
from app.config import Settings
from app.errors import RateLimitError
import hashlib

logger = structlog.get_logger()


class APIKeyAuth:
    """API Key authentication system."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.valid_keys: Set[str] = set(settings.api_keys or [])
        # For demo purposes, add a hash-based key if no keys configured
        if not self.valid_keys and settings.enable_auth:
            demo_key = hashlib.sha256(b"demo-key-change-in-production").hexdigest()[:32]
            self.valid_keys.add(demo_key)
            logger.warning("Using demo API key. Change in production!", demo_key=demo_key)
    
    def authenticate(self, credentials: HTTPAuthorizationCredentials) -> bool:
        """Authenticate API key."""
        if not self.settings.enable_auth:
            return True
        
        if not credentials or not credentials.credentials:
            return False
        
        api_key = credentials.credentials.strip()
        is_valid = api_key in self.valid_keys
        
        if not is_valid:
            logger.warning("Invalid API key attempt", key_prefix=api_key[:8] + "...")
        else:
            logger.debug("API key authenticated", key_prefix=api_key[:8] + "...")
        
        return is_valid
    
    def add_key(self, api_key: str):
        """Add a new API key."""
        self.valid_keys.add(api_key)
        logger.info("API key added", key_prefix=api_key[:8] + "...")
    
    def remove_key(self, api_key: str):
        """Remove an API key."""
        self.valid_keys.discard(api_key)
        logger.info("API key removed", key_prefix=api_key[:8] + "...")
    
    def list_keys(self) -> list[str]:
        """List API key prefixes (for management)."""
        return [key[:8] + "..." for key in self.valid_keys]


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: Optional[int] = None):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or min(requests_per_minute, 100)  # Allow some burst
        self.rate_per_second = requests_per_minute / 60.0
        self.buckets: Dict[str, Dict] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def _get_bucket(self, key: str) -> Dict:
        """Get or create a token bucket for a key."""
        now = time.time()
        
        # Periodic cleanup of old buckets
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_buckets()
            self.last_cleanup = now
        
        if key not in self.buckets:
            self.buckets[key] = {
                'tokens': self.burst_size,
                'last_refill': now,
                'requests': 0,
                'last_request': now
            }
        
        return self.buckets[key]
    
    def _refill_bucket(self, bucket: Dict) -> None:
        """Refill tokens in the bucket based on elapsed time."""
        now = time.time()
        elapsed = now - bucket['last_refill']
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.rate_per_second
        bucket['tokens'] = min(self.burst_size, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now
    
    def _cleanup_old_buckets(self) -> None:
        """Remove old buckets to prevent memory leaks."""
        now = time.time()
        old_keys = []
        
        for key, bucket in self.buckets.items():
            # Remove buckets not used for more than 1 hour
            if now - bucket['last_request'] > 3600:
                old_keys.append(key)
        
        for key in old_keys:
            del self.buckets[key]
        
        if old_keys:
            logger.debug("Cleaned up old rate limit buckets", count=len(old_keys))
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed and consume a token."""
        bucket = self._get_bucket(key)
        self._refill_bucket(bucket)
        
        bucket['last_request'] = time.time()
        bucket['requests'] += 1
        
        if bucket['tokens'] >= 1.0:
            bucket['tokens'] -= 1.0
            return True
        
        logger.warning(
            "Rate limit exceeded",
            key=key,
            tokens=bucket['tokens'],
            requests=bucket['requests']
        )
        return False
    
    def get_stats(self, key: str) -> Dict:
        """Get rate limit statistics for a key."""
        if key not in self.buckets:
            return {
                'tokens': self.burst_size,
                'requests': 0,
                'rate_limit': self.requests_per_minute,
                'burst_size': self.burst_size
            }
        
        bucket = self.buckets[key]
        self._refill_bucket(bucket)
        
        return {
            'tokens': bucket['tokens'],
            'requests': bucket['requests'],
            'rate_limit': self.requests_per_minute,
            'burst_size': self.burst_size,
            'last_request': bucket['last_request']
        }


class AuthMiddleware:
    """Combined authentication and rate limiting middleware."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.auth = APIKeyAuth(settings)
        self.rate_limiter = RateLimiter(settings.rate_limit_rpm)
        self.bearer_auth = HTTPBearer(auto_error=False)
    
    async def authenticate_request(self, request: Request) -> Optional[str]:
        """Authenticate request and return client identifier."""
        if not self.settings.enable_auth:
            # Use IP address for rate limiting when auth is disabled
            return self._get_client_ip(request)
        
        # Get authorization header
        credentials = await self.bearer_auth(request)
        
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not self.auth.authenticate(credentials):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Use API key as client identifier for rate limiting
        return credentials.credentials
    
    def check_rate_limit(self, client_id: str) -> None:
        """Check rate limit for client."""
        if not self.rate_limiter.is_allowed(client_id):
            # Get current stats for the error message
            stats = self.rate_limiter.get_stats(client_id)
            raise RateLimitError(
                f"Rate limit exceeded. Limit: {self.settings.rate_limit_rpm} requests/minute. "
                f"Available tokens: {stats['tokens']:.2f}"
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address for rate limiting."""
        # Check for forwarded headers (behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    async def __call__(self, request: Request):
        """Middleware entry point."""
        # Skip auth/rate limiting for health checks and metrics
        if request.url.path in ["/healthz", "/metrics", "/health/detailed"]:
            return None
        
        # Authenticate and get client identifier
        client_id = await self.authenticate_request(request)
        
        # Check rate limit
        self.check_rate_limit(client_id)
        
        # Store client info in request state
        request.state.client_id = client_id
        request.state.auth_enabled = self.settings.enable_auth
        
        return client_id


# Global auth middleware instance
_auth_middleware: Optional[AuthMiddleware] = None


def get_auth_middleware(settings: Settings = None) -> AuthMiddleware:
    """Get or create the global auth middleware instance."""
    global _auth_middleware
    if _auth_middleware is None and settings:
        _auth_middleware = AuthMiddleware(settings)
    return _auth_middleware


def require_auth(settings: Settings):
    """Dependency for endpoints requiring authentication."""
    async def auth_dependency(request: Request):
        auth_middleware = get_auth_middleware(settings)
        return await auth_middleware(request)
    
    return auth_dependency