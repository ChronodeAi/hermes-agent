"""Large-text redaction offloads out-of-process (structural fix, G3 step 1).

redact_sensitive_text on blocks >= ~1M chars must run in the bounded
worker pool (agent/_gil_offload.py) instead of holding the GIL inline:
measured 5.1s of GIL for one 44MB URL-dense call (super-linear regex
worst case). Tests: structural proof the pool is used (F11), result
equivalence with inline execution, loop responsiveness during a large
redaction, inline fallback for small texts and pool failures, and the
no-nested-pool guard in offload children.
"""

import asyncio
import contextlib
import concurrent.futures as cf

import concurrent.futures as cf

import pytest

import agent._gil_offload as off
import agent.redact as r
from agent._gil_offload import DEFAULT_OFFLOAD_MIN_CHARS
from agent.redact import _redact_sensitive_text_inline, redact_sensitive_text


def _url_heavy_text(mb: float) -> str:
    unit = "https://user:tok1234567890abcdef@host.example.com/path "
    return unit * int(mb * 1024 * 1024 / len(unit))


def test_small_text_stays_inline(monkeypatch):
    """Under the threshold, nothing is submitted to the pool."""

    def _fail(*a, **k):
        raise AssertionError("offload must not be attempted for small text")

    monkeypatch.setattr("agent._gil_offload.offload_text_call", _fail)
    assert r.redact_sensitive_text("hello world") == "hello world"
    assert r.redact_sensitive_text(None) is None
    assert r.redact_sensitive_text("") == ""


@pytest.mark.parametrize(
    "text,kwargs",
    [
        ("sk-abcdefghijklmnopqrstuvwxyz0123456789 in a log line", {}),
        (
            "https://user:secretpass@host.example.com/path?access_token=zzz",
            {"redact_url_credentials": True},
        ),
        ("MAX_TOKENS=4096\napiKey = 'test'", {"code_file": True}),
        ("Bearer eyJhbGciOiJIUzI1NiJ9.e30.abc", {"file_read": True}),
    ],
)
def test_dispatcher_matches_inline_exactly(text, kwargs):
    assert r.redact_sensitive_text(
        text, **kwargs
    ) == r._redact_sensitive_text_inline(text, **kwargs)


def test_large_text_offloads_and_matches_inline(monkeypatch):
    """F11 (review-2): dispatcher == inline holds whether or not the
    offload ran, so assert STRUCTURALLY that a pool submission happened -
    deleting the offload must fail this test, not just the timing one."""
    submitted = {"n": 0}

    class _SpyPool:
        def __init__(self, *a, **k):
            self._inner = cf.ProcessPoolExecutor(*a, **k)

        def submit(self, fn, *a, **k):
            submitted["n"] += 1
            return self._inner.submit(fn, *a, **k)

        def shutdown(self, *a, **k):
            return self._inner.shutdown(*a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(off, "ProcessPoolExecutor", _SpyPool)
    # A pool cached by an earlier test would bypass the spy - force a
    # fresh executor under the spy.
    monkeypatch.setattr(off, "_executor", None)
    text = _url_heavy_text(2.0)  # ~4.3M chars - over the threshold
    assert len(text) >= DEFAULT_OFFLOAD_MIN_CHARS

    via_dispatcher = r.redact_sensitive_text(text, redact_url_credentials=True)
    inline = _redact_sensitive_text_inline(
        text, force=True, redact_url_credentials=True
    )
    assert via_dispatcher == inline
    assert via_dispatcher != text  # actually redacted something
    assert submitted["n"] >= 1, (
        "dispatcher must have submitted to the worker pool for oversized "
        "text (offload deleted -> this fails structurally)"
    )


def test_loop_stays_responsive_during_large_redaction():
    """The choke-point fix: redacting URL-dense multi-MB text from a
    turn-like thread must not stall the event loop (previously a single
    44MB call held the GIL for 5.1s)."""
    import time

    text = _url_heavy_text(40)  # ~42M chars
    gaps = []

    async def main():
        async def ticker():
            last = time.perf_counter()
            while True:
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        tick = asyncio.create_task(ticker())
        try:
            await asyncio.to_thread(
                r.redact_sensitive_text, text, redact_url_credentials=True
            )
        finally:
            tick.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tick

    asyncio.run(main())
    settled = gaps[5:] or gaps
    assert settled, "ticker must have recorded ticks"
    max_gap = max(settled)
    assert max_gap < 0.2, (
        f"event loop stalled during offloaded redaction "
        f"(max tick gap {max_gap*1000:.0f} ms)"
    )


def test_offload_child_runs_inline(monkeypatch):
    """Inside an offload child (env marker), the dispatcher must not spawn
    a nested pool - offload_text_call short-circuits before any pool use."""

    def _fail(*a, **k):
        raise AssertionError("child guard failed: executor was reached")

    monkeypatch.setenv("_HERMES_GIL_OFFLOAD_CHILD", "1")
    monkeypatch.setattr("agent._gil_offload._get_executor", _fail)
    text = _url_heavy_text(2.0)  # over threshold; guard must skip pool
    result = r.redact_sensitive_text(text, redact_url_credentials=True)
    assert result is not None and result != text  # real inline result


def test_pool_failure_falls_back_inline(monkeypatch):
    """A broken pool degrades to inline - correct output, old worst case."""

    class _Broken:
        def submit(self, *a, **k):
            raise BrokenProcessPool("simulated")

    monkeypatch.setattr("agent._gil_offload._get_executor", lambda: _Broken())
    text = _url_heavy_text(2.0)
    out = r.redact_sensitive_text(text, redact_url_credentials=True)
    assert out == _redact_sensitive_text_inline(
        text, force=True, redact_url_credentials=True
    )


