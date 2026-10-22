"""API Governance and Security Module

Implements:
- Rate limiting
- API authentication
- Request validation
- Security headers
- CORS security
- API versioning
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"


class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET):
        self.strategy = strategy
        self.user_buckets: Dict[str, Dict] = {}
        self.rate_limits: Dict[str, Dict] = {
            "default": {"requests": 100, "window": 3600},  # 100 req/hour
            "ai_prediction": {"requests": 50, "window": 3600},
            "data_export": {"requests": 10, "window": 86400},
            "training": {"requests": 5, "window": 86400}
        }
    
    def is_allowed(self, user_id: str, endpoint: str = "default") -> Tuple[bool, Dict]:
        """
        Check if request is allowed.
        
        Returns:
            Tuple of (allowed: bool, metadata: Dict)
        """
        limit_config = self.rate_limits.get(endpoint, self.rate_limits["default"])
        
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket_check(user_id, endpoint, limit_config)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_check(user_id, endpoint, limit_config)
        else:
            return self._fixed_window_check(user_id, endpoint, limit_config)
    
    def _token_bucket_check(self, user_id: str, endpoint: str,
                           limit_config: Dict) -> Tuple[bool, Dict]:
        """Token bucket algorithm for rate limiting"""
        now = datetime.now(timezone.utc)
        
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = {
                endpoint: {
                    "tokens": limit_config["requests"],
                    "last_refill": now
                }
            }
        
        if endpoint not in self.user_buckets[user_id]:
            self.user_buckets[user_id][endpoint] = {
                "tokens": limit_config["requests"],
                "last_refill": now
            }
        
        bucket = self.user_buckets[user_id][endpoint]
        
        # Calculate tokens to add based on time passed
        time_passed = (now - bucket["last_refill"]).total_seconds()
        tokens_to_add = time_passed * (limit_config["requests"] / limit_config["window"])
        bucket["tokens"] = min(limit_config["requests"], bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True, {
                "remaining": int(bucket["tokens"]),
                "reset_at": (now + timedelta(seconds=limit_config["window"])).isoformat()
            }
        
        return False, {
            "remaining": int(bucket["tokens"]),
            "retry_after": 60
        }
    
    def _sliding_window_check(self, user_id: str, endpoint: str,
                             limit_config: Dict) -> Tuple[bool, Dict]:
        """Sliding window algorithm for rate limiting"""
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=limit_config["window"])
        
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = {}
        
        if endpoint not in self.user_buckets[user_id]:
            self.user_buckets[user_id][endpoint] = []
        
        # Remove old requests outside window
        self.user_buckets[user_id][endpoint] = [
            req_time for req_time in self.user_buckets[user_id][endpoint]
            if req_time > window_start
        ]
        
        if len(self.user_buckets[user_id][endpoint]) < limit_config["requests"]:
            self.user_buckets[user_id][endpoint].append(now)
            return True, {
                "remaining": limit_config["requests"] - len(self.user_buckets[user_id][endpoint]),
                "reset_at": (window_start + timedelta(seconds=limit_config["window"])).isoformat()
            }
        
        return False, {
            "remaining": 0,
            "retry_after": limit_config["window"]
        }
    
    def _fixed_window_check(self, user_id: str, endpoint: str,
                           limit_config: Dict) -> Tuple[bool, Dict]:
        """Fixed window algorithm for rate limiting"""
        now = datetime.now(timezone.utc)
        window_key = int(now.timestamp() // limit_config["window"])
        
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = {}
        
        key = f"{endpoint}_{window_key}"
        
        if key not in self.user_buckets[user_id]:
            self.user_buckets[user_id][key] = {"count": 0, "window_key": window_key}
        
        bucket = self.user_buckets[user_id][key]
        
        if bucket["count"] < limit_config["requests"]:
            bucket["count"] += 1
            next_window = (now + timedelta(seconds=limit_config["window"])).timestamp()
            return True, {
                "remaining": limit_config["requests"] - bucket["count"],
                "reset_at": datetime.fromtimestamp(next_window).isoformat()
            }
        
        return False, {"remaining": 0}


class APIKeyManager:
    """API Key management and validation"""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict] = {}
        self.key_permissions: Dict[str, List[str]] = {}
    
    def create_api_key(self, client_name: str, permissions: List[str],
                      expiration_days: int = 365) -> str:
        """Create a new API key"""
        key = self._generate_key()
        
        self.api_keys[key] = {
            "client_name": client_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=expiration_days)).isoformat(),
            "active": True,
            "last_used": None
        }
        
        self.key_permissions[key] = permissions
        
        logger.info(f"API key created for {client_name}")
        return key
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[str]]:
        """Validate API key and return client name if valid"""
        if api_key not in self.api_keys:
            return False, None
        
        key_info = self.api_keys[api_key]
        
        if not key_info["active"]:
            return False, None
        
        # Check expiration
        expiration = datetime.fromisoformat(key_info["expires_at"])
        now = datetime.now(timezone.utc)
        if now > expiration:
            key_info["active"] = False
            return False, None
        # Warn about near-expiry keys (within 7 days)
        if (expiration - now).days <= 7:
            logger.warning(
                "API key for %s expires in %d days (%s)",
                key_info["client_name"],
                (expiration - now).days,
                expiration.isoformat(),
            )
        
        # Update last used
        key_info["last_used"] = now.isoformat()
        
        return True, key_info["client_name"]
    
    def has_permission(self, api_key: str, required_permission: str) -> bool:
        """Check if API key has required permission"""
        permissions = self.key_permissions.get(api_key, [])
        return required_permission in permissions or "admin" in permissions
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            logger.info("API key revoked")
            return True
        
        return False
    
    @staticmethod
    def _generate_key() -> str:
        """Generate a secure API key"""
        import secrets
        return f"sk_{secrets.token_urlsafe(32)}"

    def expire_old_keys(self) -> int:
        """Deactivate all expired keys.

        Returns:
            Number of keys deactivated.
        """
        now = datetime.now(timezone.utc)
        deactivated = 0
        for _key, info in list(self.api_keys.items()):
            if not info.get("active"):
                continue
            try:
                expiration = datetime.fromisoformat(info["expires_at"])
            except Exception:
                continue
            if now > expiration:
                info["active"] = False
                deactivated += 1
        if deactivated:
            logger.info("Deactivated %d expired API keys", deactivated)
        return deactivated

    def rotate_api_key(self, old_key: str, expiration_days: int = 365) -> Optional[str]:
        """Revoke an existing API key and create a replacement.

        The new key preserves the original permissions and client name.

        Returns:
            The new API key, or ``None`` if the old key was not found.
        """
        if old_key not in self.api_keys:
            return None
        info = self.api_keys[old_key]
        client_name = info.get("client_name", "unknown")
        permissions = self.key_permissions.get(old_key, [])
        self.revoke_api_key(old_key)
        new_key = self.create_api_key(client_name, permissions, expiration_days=expiration_days)
        logger.info("Rotated API key for %s", client_name)
        return new_key

    def get_expiring_soon(self, days: int = 7) -> List[Dict[str, Any]]:
        """Return keys expiring within ``days`` days."""
        now = datetime.now(timezone.utc)
        results = []
        for key, info in self.api_keys.items():
            if not info.get("active"):
                continue
            try:
                expiration = datetime.fromisoformat(info["expires_at"])
            except Exception:
                continue
            remaining = (expiration - now).total_seconds() / 86400
            if 0 <= remaining <= days:
                results.append({
                    "key": key,
                    "client_name": info.get("client_name"),
                    "expires_at": info.get("expires_at"),
                    "remaining_days": round(remaining, 1),
                })
        results.sort(key=lambda item: item["remaining_days"])
        return results


class RequestValidator:
    """Validate API requests"""
    
    def __init__(self):
        self.max_payload_size = int(os.getenv('MAX_PAYLOAD_SIZE', 1048576))  # 1MB
    
    def validate_request(self, request: Any, require_auth: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Validate incoming request.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check content-type
        content_type = request.headers.get('Content-Type', '')
        if request.method in ['POST', 'PUT'] and 'json' not in content_type:
            return False, "Content-Type must be application/json"
        
        # Check payload size
        content_length = request.headers.get('Content-Length')
        if content_length and int(content_length) > self.max_payload_size:
            return False, f"Payload exceeds maximum size of {self.max_payload_size} bytes"
        
        # Check for required headers
        if require_auth and not request.headers.get('Authorization'):
            return False, "Missing Authorization header"
        
        return True, None
    
    def validate_json_schema(self, data: Dict, required_fields: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate JSON against required fields"""
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        return True, None


class SecurityHeadersMiddleware:
    """Apply security headers to responses"""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get recommended security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' https://cdn.socket.io; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; connect-src 'self' wss: ws: https://cdn.socket.io; img-src 'self' data:;",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }


class APIGovernanceManager:
    """Central API governance management"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.api_key_manager = APIKeyManager()
        self.request_validator = RequestValidator()
        self.api_versions = {
            "v1": {"released": "2024-01-01", "deprecated": False},
            "v2": {"released": "2024-06-01", "deprecated": False}
        }
        self.request_log: List[Dict] = []
        self.key_permissions = self.api_key_manager.key_permissions
    
    def create_api_protection_decorator(self, endpoint_name: str,
                                       required_permission: str = None):
        """
        Create a decorator for API endpoint protection.
        
        Usage:
            @api_governance_manager.create_api_protection_decorator("prediction_endpoint")
            def predict():
                pass
        """
        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def decorated_function(*args, **kwargs):
                from flask import request as flask_request
                request = flask_request
                valid, validation_error = self.request_validator.validate_request(request, require_auth=True)
                if not valid:
                    return {"error": validation_error}, 400

                # Get API key
                auth_header = request.headers.get('Authorization', '')
                if not auth_header.startswith('Bearer '):
                    return {"error": "Invalid authorization header"}, 401

                api_key = auth_header[7:]

                # Validate API key
                is_valid, client_name = self.api_key_manager.validate_api_key(api_key)
                if not is_valid:
                    return {"error": "Invalid or expired API key"}, 401

                # Check permissions if required
                if required_permission:
                    if not self.api_key_manager.has_permission(api_key, required_permission):
                        return {"error": "Insufficient permissions"}, 403

                # Check rate limit
                allowed, limit_info = self.rate_limiter.is_allowed(client_name, endpoint_name)
                if not allowed:
                    return {
                        "error": "Rate limit exceeded",
                        "retry_after": limit_info.get("retry_after", 60)
                    }, 429

                # Log request
                self._log_api_request(endpoint_name, client_name, request)

                # Call original function
                return f(*args, **kwargs)

            return decorated_function
        
        return decorator
    
    def _log_api_request(self, endpoint: str, client: str, request: Any) -> None:
        """Log API request for auditing"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "client": client,
            "method": request.method,
            "path": request.path
        }
        self.request_log.append(log_entry)
    
    def get_api_usage_report(self, client: str = None, days: int = 30) -> Dict:
        """Generate API usage report"""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        filtered_log = [
            entry for entry in self.request_log
            if datetime.fromisoformat(entry["timestamp"]) > start_date
        ]
        
        if client:
            filtered_log = [e for e in filtered_log if e["client"] == client]
        
        endpoint_stats = {}
        for entry in filtered_log:
            endpoint = entry["endpoint"]
            endpoint_stats[endpoint] = endpoint_stats.get(endpoint, 0) + 1
        
        return {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "period": f"Last {days} days",
            "client": client or "all_clients",
            "total_requests": len(filtered_log),
            "endpoint_statistics": endpoint_stats
        }

    def enforce_key_rotation(self) -> Dict:
        """Enforce API key expiration and rotation hygiene.

        Returns:
            Summary dict with expired key count and keys nearing expiry.
        """
        expired_count = self.api_key_manager.expire_old_keys()
        expiring_soon = self.api_key_manager.get_expiring_soon(days=7)
        return {
            "expired_count": expired_count,
            "expiring_soon_count": len(expiring_soon),
            "expiring_soon": expiring_soon,
            "total_active_keys": sum(
                1 for info in self.api_key_manager.api_keys.values() if info.get("active")
            ),
        }
