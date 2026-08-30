"""Desktop-spawned loopback backends must not inherit the public_url gate.

The desktop app spawns a private headless ``hermes serve --host 127.0.0.1
--port 0`` (``HERMES_DESKTOP=1``) and authenticates its WebSocket with the
legacy ``?token=<HERMES_DASHBOARD_SESSION_TOKEN>`` credential. When the
operator configures ``dashboard.public_url`` for a real dashboard elsewhere
(e.g. a Tailscale gateway), ``should_require_dashboard_auth`` engaged the auth
gate on the desktop's loopback child too — where ``_ws_auth_reason``
unconditionally rejects the legacy ``?token=`` path in gated mode. Result: the
desktop app failed every boot with "HTTP-reachable but the WebSocket (/api/ws)
rejected the session token" (observed 2026-08-30, mac-studio).

The exemption is deliberately narrow:

* only ``desktop_spawn=True`` AND a loopback bind skips the public-hosts arm;
* a desktop-spawned backend on a non-loopback bind still gates (bind arm);
* a plain loopback ``dashboard``/``serve`` (no HERMES_DESKTOP) still gates
  when ``public_url`` declares external exposure — that's the reverse-proxy
  deployment the arm exists to protect.
"""

import inspect
from pathlib import Path

from hermes_cli.web_server import (
    _LOOPBACK_HOST_VALUES,
    should_require_dashboard_auth,
    start_server,
)

_PUBLIC_HOSTS = frozenset({"mac-studio.tail10294f.ts.net"})
_LOOPBACK = "127.0.0.1"
_PUBLIC_BIND = "100.116.96.25"

assert _LOOPBACK in _LOOPBACK_HOST_VALUES  # guard the fixture premise


def test_legacy_loopback_bind_still_gates_on_public_url():
    """Unchanged behavior: public_url gates a loopback bind without the exemption."""
    assert should_require_dashboard_auth(
        _LOOPBACK, _PUBLIC_HOSTS, desktop_spawn=False
    )
    assert should_require_dashboard_auth(_LOOPBACK, _PUBLIC_HOSTS)


def test_desktop_spawn_on_loopback_skips_public_url_gate():
    """The fix: the app's process-private loopback child boots in token mode."""
    assert not should_require_dashboard_auth(
        _LOOPBACK, _PUBLIC_HOSTS, desktop_spawn=True
    )


def test_desktop_spawn_on_non_loopback_bind_still_gates():
    """Defense in depth: the exemption never disarms the bind arm."""
    assert should_require_dashboard_auth(
        _PUBLIC_BIND, _PUBLIC_HOSTS, desktop_spawn=True
    )
    assert should_require_dashboard_auth(_PUBLIC_BIND, frozenset(), desktop_spawn=True)


def test_desktop_spawn_without_public_url_stays_token_mode():
    """No public hosts → loopback is token mode regardless of spawn origin."""
    assert not should_require_dashboard_auth(
        _LOOPBACK, frozenset(), desktop_spawn=True
    )
    assert not should_require_dashboard_auth(_LOOPBACK, frozenset())


def _source(path: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    with open(repo_root / path, encoding="utf-8") as fh:
        return fh.read()


def test_start_server_wires_the_desktop_spawn_flag():
    """Structural guard: removing the wiring must fail here, not in the field.

    Regression shape seen before (2026-08-30 review pass 2): a feature whose
    only enforcement was incidental survived deletion because no test asserted
    the wiring itself.
    """
    src = inspect.getsource(start_server)
    assert 'os.getenv("HERMES_DESKTOP") == "1"' in src
    assert "desktop_spawn=_desktop_spawn" in src


def test_dashboard_preflight_wires_the_desktop_spawn_flag():
    """The interactive provider preflight must apply the same exemption.

    Otherwise a desktop user with public_url set but no auth provider reaches
    start_server's fail-closed SystemExit — the boot dies before the gate
    decision ever runs.
    """
    main_src = _source("hermes_cli/main.py")
    assert 'desktop_spawn=os.getenv("HERMES_DESKTOP") == "1"' in main_src
