"""Minimal ASGI bearer-token guard for the HTTP MCP transports.

The MCP server is safe by default because only the stdio transport is enabled.
When an operator explicitly exposes it over SSE or streamable HTTP, a static
token can be required so that the tools are not reachable by any host on the
network. TLS termination, rate limiting and client authentication still belong to
the reverse proxy in front of the process.
"""

from __future__ import annotations

import hmac
import json
from typing import Any

Scope = dict[str, Any]
Receive = Any
Send = Any


class BearerTokenMiddleware:
    """Reject HTTP requests that do not carry the configured bearer token."""

    def __init__(self, app: Any, token: str, *, realm: str = "edge-forge") -> None:
        if not token:
            raise ValueError("a non-empty token is required")
        self._app = app
        self._token = token
        self._realm = realm

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        if self._authorized(scope):
            await self._app(scope, receive, send)
            return
        await _send_unauthorized(send, self._realm)

    def _authorized(self, scope: Scope) -> bool:
        headers = scope.get("headers") or []
        for name, value in headers:
            if name.lower() != b"authorization":
                continue
            scheme, _, credential = value.decode("latin-1").partition(" ")
            if scheme.lower() == "bearer" and hmac.compare_digest(credential.strip(), self._token):
                return True
        return False


async def _send_unauthorized(send: Send, realm: str) -> None:
    body = json.dumps(
        {"error": "Unauthorized", "message": "a valid bearer token is required"}
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"www-authenticate", f'Bearer realm="{realm}"'.encode("latin-1")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
