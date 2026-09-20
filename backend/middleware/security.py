"""
Security middleware for BIS Recommendation Engine.

Applies:
  1. Trusted proxy rate limiting — sliding-window in-memory (per-IP) with trusted proxy verification
  2. Optional API Key authentication — enforces X-API-Key when API_KEY setting is configured
  3. Request size & text length caps — prevents memory exhaustion & payload abuse
  4. Error sanitisation — strips Python tracebacks from 500 responses with unique error_id
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, List

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from backend.config.settings import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Rate-limit store (in-process)
# ─────────────────────────────────────────────────────────────────────────────

_rate_store: Dict[str, Deque[float]] = defaultdict(deque)
_rate_lock = asyncio.Lock()

RATE_LIMIT_REQUESTS = 60     # max requests per IP per window
RATE_LIMIT_WINDOW_S = 60.0   # sliding window in seconds
MAX_BODY_BYTES = 12 * 1024 * 1024   # 12 MB hard cap at middleware layer
MAX_QUERY_CHARS = 20_000

OPEN_PATH_PREFIXES = (
    "/health",
    "/healthz",
    "/readyz",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/health",
    "/api/healthz",
    "/api/readyz",
    "/api/docs",
    "/api/redoc",
    "/api/openapi.json",
)


def _is_trusted_proxy(host: str, trusted_list: List[str]) -> bool:
    """Check if host is in trusted proxies list / CIDR ranges."""
    if not host:
        return False
    if host in ("localhost", "127.0.0.1", "::1", "testclient"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        for trusted in trusted_list:
            if trusted in ("localhost", "testclient"):
                if host == trusted:
                    return True
                continue
            if "/" in trusted:
                if ip in ipaddress.ip_network(trusted, strict=False):
                    return True
            else:
                if ip == ipaddress.ip_address(trusted):
                    return True
    except Exception:
        pass
    return False


def _client_ip(request: Request) -> str:
    """Extract client IP, inspecting X-Forwarded-For ONLY if caller is a trusted proxy."""
    client_host = request.client.host if request.client else ""
    trusted_proxies = getattr(settings, "TRUSTED_PROXIES", [])
    if _is_trusted_proxy(client_host, trusted_proxies):
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
    return client_host or "unknown"


def _is_open_path(path: str) -> bool:
    """Check if path is exempt from API key requirement."""
    return any(path == p or path.startswith(p + "/") for p in OPEN_PATH_PREFIXES)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Applies trusted-proxy rate limiting + API key auth + body size caps + error sanitisation."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        ip = _client_ip(request)
        path = request.url.path

        # ── API Key Authentication ─────────────────────────────────────────────
        api_key_configured = getattr(settings, "API_KEY", "")
        if api_key_configured and not _is_open_path(path):
            provided_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
            if not provided_key or provided_key != api_key_configured:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "unauthorized",
                        "message": "Invalid or missing X-API-Key header.",
                    },
                )

        # ── Rate limiting ──────────────────────────────────────────────────────
        now = time.monotonic()
        async with _rate_lock:
            window = _rate_store[ip]
            while window and now - window[0] > RATE_LIMIT_WINDOW_S:
                window.popleft()
            if len(window) >= RATE_LIMIT_REQUESTS:
                logger.warning("Rate limit exceeded: ip=%s requests=%d", ip, len(window))
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "message": f"Too many requests. Limit: {RATE_LIMIT_REQUESTS} per {int(RATE_LIMIT_WINDOW_S)}s.",
                        "retry_after_seconds": int(RATE_LIMIT_WINDOW_S - (now - window[0])) if window else int(RATE_LIMIT_WINDOW_S),
                    },
                    headers={"Retry-After": str(int(RATE_LIMIT_WINDOW_S))},
                )
            window.append(now)

        # ── Body size cap ──────────────────────────────────────────────────────
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large",
                    "message": f"Request body exceeds maximum allowed size of {MAX_BODY_BYTES // (1024*1024)} MB.",
                },
            )

        # ── Call next handler ──────────────────────────────────────────────────
        try:
            response = await call_next(request)
        except Exception as exc:
            error_id = str(uuid.uuid4())
            logger.error("Unhandled exception [error_id=%s] for %s %s: %s", error_id, request.method, path, exc, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An internal error occurred. Please try again.",
                    "error_id": error_id,
                },
            )

        return response
