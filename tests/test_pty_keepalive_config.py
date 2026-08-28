"""Config propagation for the dashboard keep-alive PTY registry knobs.

``dashboard.pty_session_ttl_s`` controls how long a detached Chat-tab PTY
stays reattachable before the reaper closes it (the next connect then
re-spawns ``hermes --tui`` and re-loads conversation history — the slow path
mobile users hit after leaving the dashboard idle). ``dashboard.pty_buffer_cap``
is the replay ring size per keep-alive PTY.
"""
import textwrap
import time

import pytest


@pytest.fixture()
def _temp_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _write_config(home, body: str) -> None:
    (home / "config.yaml").write_text(textwrap.dedent(body))


def test_dashboard_pty_defaults_present(_temp_home):
    from hermes_cli.config import load_config

    _write_config(_temp_home, "model:\n  default: test-model\n")
    cfg = load_config()
    dash = cfg.get("dashboard") or {}
    assert dash.get("pty_session_ttl_s") == 1800.0
    assert dash.get("pty_buffer_cap") == 1048576


def test_dashboard_pty_values_propagate_from_yaml(_temp_home):
    from hermes_cli.config import load_config

    _write_config(
        _temp_home,
        """
        dashboard:
          pty_session_ttl_s: 86400
          pty_buffer_cap: 4194304
        """,
    )
    cfg = load_config()
    dash = cfg["dashboard"]
    assert dash["pty_session_ttl_s"] == 86400
    assert dash["pty_buffer_cap"] == 4194304
    # Deep-merge: sibling defaults survive a partial user section.
    assert dash.get("theme") == "default"


def test_registry_configure_applies_ttl_and_cap():
    from hermes_cli.pty_session import PtySessionRegistry

    reg = PtySessionRegistry(ttl=1800.0, max_sessions=16,
                             buffer_cap=1048576, read_timeout=0.2)
    reg.configure(ttl=86400, buffer_cap=4194304)
    assert reg._ttl == 86400.0
    assert reg._buffer_cap == 4194304
    # Out-of-range values clamp instead of raising.
    reg.configure(ttl=-5, buffer_cap=0)
    assert reg._ttl == 0.0
    assert reg._buffer_cap == 1
    # Partial updates leave the other field untouched.
    reg.configure(ttl=60)
    assert reg._ttl == 60.0
    assert reg._buffer_cap == 1


@pytest.mark.asyncio
async def test_reaper_honors_configured_ttl():
    """A session detached longer than the configured TTL is reaped on sweep."""
    from hermes_cli.pty_session import PtySession, PtySessionRegistry

    reg = PtySessionRegistry(ttl=0.0, max_sessions=16,
                             buffer_cap=4096, read_timeout=0.05)
    session = PtySession("k", object(), buffer_cap=4096, read_timeout=0.05)
    session.attached = False
    # Detached long ago: (now - detached) > ttl=0 → eligible for the reaper.
    session.last_detached_at = time.monotonic() - 30
    reg._sessions["k"] = session
    await reg.reap_idle()
    assert "k" not in reg._sessions


@pytest.mark.asyncio
async def test_reaper_keeps_session_within_configured_ttl():
    """Same session survives the sweep while inside the TTL window."""
    from hermes_cli.pty_session import PtySession, PtySessionRegistry

    reg = PtySessionRegistry(ttl=3600.0, max_sessions=16,
                             buffer_cap=4096, read_timeout=0.05)
    session = PtySession("k", object(), buffer_cap=4096, read_timeout=0.05)
    session.attached = False
    session.last_detached_at = time.monotonic() - 30
    reg._sessions["k"] = session
    await reg.reap_idle()
    assert "k" in reg._sessions
