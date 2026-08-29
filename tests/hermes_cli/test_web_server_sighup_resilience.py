"""SIGHUP resilience for the headless dashboard backend (2026-08-29 incident).

A stray SIGHUP (terminal teardown / process-group cleanup) used to silently
terminate the dashboard backend — the default SIGHUP disposition — dropping
every dashboard WS session until launchd keepalive respawned it ~1s later.
The backend must log-and-ignore SIGHUP; SIGTERM/SIGINT keep their graceful
shutdown semantics (dashboard_procs stops the backend via SIGTERM→SIGKILL).
"""

import logging
import os
import signal
import threading
import time

import pytest

from hermes_cli.web_server import _install_sighup_resilience

_NO_SIGHUP = not hasattr(signal, "SIGHUP")


def _restore(prev):
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, prev)


@pytest.mark.skipif(_NO_SIGHUP, reason="platform has no SIGHUP")
def test_installs_logging_handler_on_main_thread():
    prev = signal.getsignal(signal.SIGHUP)
    try:
        assert threading.current_thread() is threading.main_thread()
        assert _install_sighup_resilience() is True
        handler = signal.getsignal(signal.SIGHUP)
        assert callable(handler)
        # Our logging handler, not a bare default/ignore disposition.
        assert handler not in (signal.SIG_DFL, signal.SIG_IGN)
    finally:
        _restore(prev)


@pytest.mark.skipif(_NO_SIGHUP, reason="platform has no SIGHUP")
def test_install_refuses_non_main_thread():
    if not hasattr(signal, "SIGHUP"):
        pytest.skip("platform has no SIGHUP")
    prev = signal.getsignal(signal.SIGHUP)
    result = {}

    def worker():
        result["ok"] = _install_sighup_resilience()

    try:
        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)
        assert result["ok"] is False
        # Handler untouched by the off-thread attempt.
        assert signal.getsignal(signal.SIGHUP) is prev
    finally:
        _restore(prev)


@pytest.mark.skipif(_NO_SIGHUP, reason="platform has no SIGHUP")
def test_sighup_delivery_does_not_terminate_process(caplog):
    prev = signal.getsignal(signal.SIGHUP)
    try:
        assert _install_sighup_resilience() is True
        with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
            # Delivered synchronously to self: under the default disposition
            # this would terminate the pytest process before returning.
            os.kill(os.getpid(), signal.SIGHUP)
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if any("SIGHUP" in r.getMessage() for r in caplog.records):
                    break
                time.sleep(0.02)
        assert any("SIGHUP" in r.getMessage() for r in caplog.records), (
            "SIGHUP delivery must be logged with attribution"
        )
    finally:
        _restore(prev)
