"""Regression tests for the adversarial-review fixes (2026-08-30).

Findings addressed:
- #2: the compression abort streak lived in sessions.compression_failure_error,
  which the cooldown writer overwrites on every summary failure — silently
  resetting the streak in exactly the incident's failure shape. The streak
  now lives in a dedicated table and must survive cooldown writes.
- #4: hard_truncate_middle_window archived rows as active=0, compacted=0 —
  the rewind/undo class, excluded from display reads AND session search,
  while user-facing text claimed "still searchable". Rows must land in the
  compacted class (active=0, compacted=1) like the sibling compactor.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    return SessionDB(tmp_path / "state.db")


def _make_session(db: SessionDB, session_id: str = "sess-abort") -> None:
    conn = getattr(db, "conn", None) or db._connect()  # repo-varies
    try:
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            (session_id, "test", 1.0),
        )
        conn.commit()
    finally:
        pass


def test_streak_survives_cooldown_overwrite(db: SessionDB, tmp_path: Path):
    """The incident shape: a summary timeout between two fence aborts used
    to reset the streak to 0 because both shared a free-text column."""
    session_id = "sess-streak"
    db.record_compression_abort(session_id, "commit_fence_cancelled")
    db.record_compression_abort(session_id, "commit_fence_cancelled")
    assert db.get_compression_abort_streak(session_id) == 2

    # A summary failure fires the pre-existing cooldown writer...
    db.record_compression_failure_cooldown(
        session_id, "host compress_context timeout (no summary progress)"
    )
    # ...and the streak must SURVIVE it (dedicated storage, not the
    # cooldown column).
    assert db.get_compression_abort_streak(session_id) == 2

    db.clear_compression_abort_streak(session_id)
    assert db.get_compression_abort_streak(session_id) == 0


def test_hard_truncate_keeps_history_searchable(db: SessionDB, tmp_path: Path):
    """Rows archived by hard_truncate_middle_window must remain discoverable
    via session search (active=0, compacted=1 — the compactor class), not
    soft-deleted (active=0, compacted=0)."""
    db.record_compression_abort  # touch method existence
    session_id = "sess-trunc"
    db.get_messages  # noqa: B018 - attribute existence probe

    # Seed a session with a message that must remain searchable.
    db.create_session(session_id, source="test")
    db.append_message(session_id, "user", "the quick brown fox jumps")
    import time as _t

    row = db._execute_write(
        lambda conn: conn.execute(
            "SELECT id FROM messages WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
    )
    assert row is not None, "seed message must exist"

    db.hard_truncate_middle_window(
        session_id, target_tokens=10, reason="test-truncate"
    )
    # Search must still find the archived message (marker text promises it).
    results = db.search_messages(
        "quick brown fox", include_inactive=False, limit=10
    )
    hits = [r for r in (results or []) if r.get("session_id") == session_id]
    assert hits, "truncated middle-window rows must stay searchable"
