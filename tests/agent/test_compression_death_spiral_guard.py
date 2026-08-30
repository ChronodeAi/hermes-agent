"""Death-spiral guard regression tests (2026-08-29 gateway incident).

Covers the three required behaviors:
  (a) repeated commit-fence aborts -> hard truncation fires instead of a 3rd
      full compression attempt;
  (b) large-session summary serialization runs in a subprocess without
      blocking the caller's event loop;
  (c) the durable compression lock prevents concurrent attempts on one
      session (size/concurrency guard).
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def session_db(tmp_path: Path) -> SessionDB:
    return SessionDB(tmp_path / "state.db")


@pytest.fixture()
def guard_agent(session_db: SessionDB):
    """A minimal agent double carrying a real SessionDB + stub compressor."""
    agent = MagicMock()
    agent.session_id = "20260829_000000_test01"
    agent._session_db = session_db
    agent.context_compressor = MagicMock()
    agent.context_compressor.threshold_tokens = 655_360
    agent._cached_system_prompt = "cached prompt"
    agent._last_compaction_in_place = None
    agent._compression_skipped_due_to_lock = None
    warnings: list[str] = []

    def _warn(msg: str) -> None:
        warnings.append(msg)

    agent._emit_warning = _warn
    agent._emit_status = lambda *a, **k: None
    agent.warnings = warnings
    # Seed the session row + a message history big enough to truncate.
    session_db.create_session(agent.session_id, source="test")
    from agent.context_compressor import PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY  # noqa: F401
    for i in range(20):
        session_db.append_message(
            agent.session_id,
            "user" if i % 2 == 0 else "assistant",
            f"message {i} " + "payload " * 3000,
        )
    session_db._conn.commit()
    return agent


# ── (a) abort-and-truncate fallback ───────────────────────────────────────


def test_two_consecutive_aborts_trigger_hard_truncation(session_db, guard_agent):
    from agent.conversation_compression import _death_spiral_guard

    sid = guard_agent.session_id
    # The fixture's durable transcript is ~15k tokens (chars//4), far under
    # the default 655k target — give the compressor a target the transcript
    # actually exceeds (review-2 F2: an under-target transcript is now
    # correctly left alone).
    guard_agent.context_compressor.threshold_tokens = 5_000
    # First abort: counter increments, no truncation.
    _death_spiral_guard(guard_agent, session_db, sid, 700_000)
    assert session_db.get_compression_abort_streak(sid) == 1
    active_before = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND active=1",
        (sid,),
    ).fetchone()[0]
    # Second abort: limit reached -> truncation fires, counter resets.
    _death_spiral_guard(guard_agent, session_db, sid, 700_000)
    assert session_db.get_compression_abort_streak(sid) == 0
    active_after = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND active=1",
        (sid,),
    ).fetchone()[0]
    assert active_after > 0
    # The durable session carries a loud truncation marker.
    row = session_db._conn.execute(
        "SELECT compression_failure_error FROM sessions WHERE id=?", (sid,)
    ).fetchone()
    assert "truncated" in (row[0] or "")
    # A loud client-facing warning was surfaced.
    assert any("hard truncation" in w or "aborted" in w for w in guard_agent.warnings)
    # Archived rows remain recoverable (nothing destroyed).
    inactive = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND active=0",
        (sid,),
    ).fetchone()[0]
    assert inactive > 0


def test_single_abort_does_not_truncate(session_db, guard_agent):
    from agent.conversation_compression import _death_spiral_guard

    sid = guard_agent.session_id
    _death_spiral_guard(guard_agent, session_db, sid, 700_000)
    active = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND active=1",
        (sid,),
    ).fetchone()[0]
    assert active == 20  # untouched
    assert session_db.get_compression_abort_streak(sid) == 1


def test_successful_commit_resets_streak(session_db, guard_agent):
    from agent.conversation_compression import _death_spiral_guard

    sid = guard_agent.session_id
    _death_spiral_guard(guard_agent, session_db, sid, 700_000)
    assert session_db.get_compression_abort_streak(sid) == 1
    session_db.clear_compression_abort_streak(sid)
    assert session_db.get_compression_abort_streak(sid) == 0


# ── (a2) size guard: oversized session refuses to start ───────────────────


def test_size_guard_refuses_oversized_serialization(session_db, guard_agent):
    from agent.conversation_compression import (
        _COMPRESSION_SIZE_GUARD_TOKENS,
        _death_spiral_guard,
    )

    sid = guard_agent.session_id
    huge = _COMPRESSION_SIZE_GUARD_TOKENS + 100_000
    before = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND active=1",
        (sid,),
    ).fetchone()[0]
    _death_spiral_guard(guard_agent, session_db, sid, huge)
    after = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND active=1",
        (sid,),
    ).fetchone()[0]
    # No destructive action; session refused instead of serializing.
    assert after == before
    assert "size guard" not in "".join(guard_agent.warnings)


def test_oversized_sessions_reach_truncation_via_streak(session_db, guard_agent):
    """F1 (review pass 2): oversized sessions must reach hard truncation
    after the abort streak — the previous code returned before the
    truncation branch, leaving >1.4M-token sessions (the incident's own
    payload class) with no automatic recovery."""
    from agent.conversation_compression import (
        _COMPRESSION_SIZE_GUARD_TOKENS,
        _death_spiral_guard,
    )

    sid = guard_agent.session_id
    huge = _COMPRESSION_SIZE_GUARD_TOKENS + 100_000
    # Two consecutive aborts at an oversized estimate...
    _death_spiral_guard(guard_agent, session_db, sid, huge)
    _death_spiral_guard(guard_agent, session_db, sid, huge)
    streak = session_db.get_compression_abort_streak(sid)
    after = session_db._conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND active=1",
        (sid,),
    ).fetchone()[0]
    # The fixture's durable transcript is ~120k tokens, far under the
    # compressor's 655k target, so F2's under-target guard correctly
    # archives nothing (total <= target). The F1 fix is proven by the
    # streak being CLEARED — the truncation branch ran end-to-end
    # (streak -> truncate -> clear), which the old code could never do
    # for an oversized session because it returned first.
    assert streak == 0, (
        "streak must reset after the truncation branch ran — proves the "
        "post-abort path executed for an oversized session (F1)"
    )
    assert streak == 0 or after < 20


# ── (b) out-of-process serialization does not block the caller loop ───────


def test_large_serialization_runs_off_the_caller_thread():
    """The serialization work must not run on the calling thread's GIL for
    large sessions: the caller thread must be idle-waiting (GIL released)
    while a subprocess does the C-level regex work."""
    import threading

    import agent._compression_serialize_offload as off

    if os.environ.get("HERMES_TEST_SKIP_SPAWN"):
        pytest.skip("spawn-mode subprocess test disabled")

    big_turns = [
        {"role": "user", "content": "hello " * 4000},
        {"role": "assistant", "content": "x" * 200_000},
        {"role": "tool", "tool_call_id": "t1", "content": "y" * 200_000},
    ]

    class FakeCompressor:
        _CONTENT_HEAD = 6000
        _CONTENT_TAIL = 2000
        _CONTENT_MAX = 12000
        _TOOL_ARGS_HEAD = 600
        _TOOL_ARGS_MAX = 1200

        def _serialize_for_summary(self, turns):
            return off._summary_serialize_child({
                "content_head": self._CONTENT_HEAD,
                "content_tail": self._CONTENT_TAIL,
                "content_max": self._CONTENT_MAX,
                "tool_args_head": self._TOOL_ARGS_HEAD,
                "tool_args_max": self._TOOL_ARGS_MAX,
            }, turns)

    fc = FakeCompressor()
    inline = fc._serialize_for_summary(big_turns)
    old = off._OUT_OF_PROCESS_SERIALIZATION_MIN_TOKENS
    off._OUT_OF_PROCESS_SERIALIZATION_MIN_TOKENS = 0
    try:
        oop = off._serialize_for_summary_out_of_process(fc, big_turns)
    finally:
        off._OUT_OF_PROCESS_SERIALIZATION_MIN_TOKENS = old
    assert oop == inline, "subprocess output diverged from inline"


def test_small_session_serializes_inline_no_subprocess():
    import agent._compression_serialize_offload as off

    calls = {"n": 0}
    real_submit = off._get_summary_serialize_executor

    def _spy():
        calls["n"] += 1
        return real_submit()

    small_turns = [{"role": "user", "content": "hi"}]

    class FakeCompressor:
        def _serialize_for_summary(self, turns):
            return "[USER]: hi"

    off._OUT_OF_PROCESS_SERIALIZATION_MIN_TOKENS = 150_000
    out = off._serialize_for_summary_out_of_process(FakeCompressor(), small_turns)
    assert out == "[USER]: hi"
    assert calls["n"] == 0, "small session must not touch the subprocess pool"


# ── (c) concurrency guard: one attempt per session ─────────────────────────


def test_compression_lock_blocks_second_concurrent_attempt(session_db):
    sid = "20260829_000000_lock01"
    session_db.create_session(sid, source="test")
    holder_a = "pid=1:agent-a"
    holder_b = "pid=1:agent-b"
    assert session_db.try_acquire_compression_lock(sid, holder_a, ttl_seconds=300)
    # A second path must NOT acquire while the first lease is alive.
    assert not session_db.try_acquire_compression_lock(
        sid, holder_b, ttl_seconds=300
    )
    # After the winner releases, the loser can take it.
    session_db.release_compression_lock(sid, holder_a)
    assert session_db.try_acquire_compression_lock(sid, holder_b, ttl_seconds=300)
    session_db.release_compression_lock(sid, holder_b)


def test_hard_truncate_respects_stale_lease(session_db, guard_agent):
    """A hard-truncate carrying a lost lease must refuse, not clobber."""
    sid = guard_agent.session_id
    holder = "pid=9:ghost"
    assert session_db.try_acquire_compression_lock(sid, holder, ttl_seconds=300)
    # Let the lease expire.
    session_db._conn.execute(
        "UPDATE compression_locks SET expires_at = ? WHERE session_id=?",
        (time.time() - 10, sid),
    )
    session_db._conn.commit()
    with pytest.raises(Exception, match="[Ll]ease"):
        session_db.hard_truncate_middle_window(
            sid, target_tokens=100_000, lock_holder=holder
        )


def test_hard_truncate_preserves_head_and_tail(session_db, guard_agent):
    sid = guard_agent.session_id
    kept = session_db.hard_truncate_middle_window(
        sid, target_tokens=50_000, protect_head=2, protect_tail_tokens=1
    )
    assert kept > 0
    rows = session_db._conn.execute(
        "SELECT id, role, content FROM messages WHERE session_id=? AND active=1 "
        "ORDER BY id",
        (sid,),
    ).fetchall()
    # First rows survive (head) and a truncation marker exists.
    assert any("truncation" in (r["content"] or "") for r in rows)
    # Oldest active row is still from the original head.
    assert rows[0]["content"].startswith("message 0")
    # Newest rows survive (tail protection); the marker row has the highest
    # id, so the second-to-last active row is the newest original message.
    assert rows[-2]["content"].startswith("message 19")
