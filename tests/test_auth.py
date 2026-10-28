"""Tests for the ASGI bearer-token middleware protecting the HTTP transports."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from edge_forge.mcp_server.auth import BearerTokenMiddleware


def _run(scope: dict[str, Any], token: str = "secret") -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def app(inner_scope: dict[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    asyncio.run(BearerTokenMiddleware(app, token)(scope, receive, send))
    return sent


def test_requests_without_a_token_are_rejected() -> None:
    sent = _run({"type": "http", "headers": []})
    assert sent[0]["status"] == 401
    assert sent[0]["headers"] == [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(sent[1]["body"])).encode("ascii")),
        (b"www-authenticate", b'Bearer realm="edge-forge"'),
    ]
    assert json.loads(sent[1]["body"])["error"] == "Unauthorized"


def test_requests_with_the_wrong_token_are_rejected() -> None:
    scope = {"type": "http", "headers": [(b"authorization", b"Bearer guess")]}
    assert _run(scope)[0]["status"] == 401


def test_requests_with_the_right_token_pass_through() -> None:
    scope = {"type": "http", "headers": [(b"authorization", b"Bearer secret")]}
    assert _run(scope)[0]["status"] == 200


def test_lifespan_scopes_are_not_filtered() -> None:
    assert _run({"type": "lifespan"})[0]["status"] == 200


def test_empty_tokens_are_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="non-empty token"):
        BearerTokenMiddleware(lambda scope, receive, send: None, "")
