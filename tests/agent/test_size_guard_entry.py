"""End-to-end guard tests at the compress_context entry point.

Review-2 findings #1 and #6: the start-of-attempt size guard had no test
at its actual entry point (deleting it passed CI), and its placement
silently blocked the codex app-server route, which never serializes
in-process. These tests drive compress_context itself.
"""

import inspect
from typing import Any, Dict
from unittest.mock import MagicMock

import agent.conversation_compression as cc
from agent.conversation_compression import (
    _COMPRESSION_SIZE_GUARD_TOKENS,
    compress_context,
)


class _RefusingCompressor:
    """Any serialization during a refused attempt fails the test."""

    def _serialize_for_summary(self, turns):
        raise AssertionError(
            "oversized session must never reach serialization"
        )


def _fake_agent(api_mode="chat_completions"):
    class _Agent:
        pass

    agent = _Agent()
    agent.session_id = "sess-guard-e2e"
    agent.model = "test-model"
    agent.api_mode = api_mode
    agent.context_compressor = _RefusingCompressor()
    agent._cached_system_prompt = "existing system prompt"
    return agent


def test_start_guard_refuses_oversized_session_before_serialization():
    """Driving compress_context with approx_tokens over the guard must
    return the ORIGINAL messages + system prompt unchanged, before any
    serialization runs (review-2 finding #6: this entry point had no
    test — deleting the guard passed CI)."""
    agent = _fake_agent("chat_completions")
    messages: list[Dict[str, Any]] = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    out_messages, out_system = compress_context(
        agent,
        messages=[{"role": "user", "content": "hello"}],
        system_message="sys",
        approx_tokens=_COMPRESSION_SIZE_GUARD_TOKENS + 50_000,
        task_id="test",
    )
    # Refusal contract: original messages unchanged, existing prompt kept.
    assert out_messages == [{"role": "user", "content": "hello"}]
    assert out_system == "sys"


def test_codex_route_exempt_from_size_guard():
    """Review-2 finding #1: the codex app-server route never serializes
    in-process, so the start guard must exempt it — otherwise an
    oversized codex session loses ALL compression paths."""
    src = inspect.getsource(cc.compress_context)
    assert 'getattr(agent, "api_mode", None) == "codex_app_server"' in src, (
        "the size-guard exemption must cover the codex app-server route"
    )
