"""
Security middleware for BIS Recommendation Engine.

Applies:
  1. Rate limiting   — sliding-window in-memory (per-IP), configurable via settings
  2. Request size    — hard cap on request body size to prevent memory exhaustion
  3. Error sanitisation — strips Python tracebacks from 500 responses in production
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Dict, Deque

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rate-limit store (in-process; replace with Redis for multi-worker prod)
# ─────────────────────────────────────────────────────────────────────────────

_rate_store: Dict[str, Deque[float]] = defaultdict(deque)
_rate_lock = asyncio.Lock()

RATE_LIMIT_REQUESTS = 30     # max requests per IP per window
RATE_LIMIT_WINDOW_S = 60.0   # sliding window in seconds
MAX_BODY_BYTES = 12 * 1024 * 1024   # 12 MB hard cap at middleware layer


def _client_ip(request: Request) -> str:
    """Extract real client IP, honouring X-Forwarded-For (single-hop only)."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


class SecurityMiddleware(BaseHTTPMiddleware):
    """Applies rate limiting + body-size cap + error sanitisation."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        ip = _client_ip(request)

        # ── Rate limiting ──────────────────────────────────────────────────────
        now = time.monotonic()
        async with _rate_lock:
            window = _rate_store[ip]
            # Drop events older than the window
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
            # Sanitise unhandled exceptions — never expose tracebacks to clients
            logger.error("Unhandled exception for %s %s: %s", request.method, request.url.path, exc, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An internal error occurred. The request has been logged. Please try again.",
                    "request_id": request.headers.get("x-request-id", ""),
                },
            )

        return response
