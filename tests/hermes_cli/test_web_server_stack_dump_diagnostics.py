"""SIGUSR1 stack-dump diagnostics for the gateway (forensics enabler).

SIGUSR1 must append an all-thread Python traceback to
$HERMES_HOME/logs/gateway-stacks.log so "event loop stalled (GIL pressure
suspected)" warnings can be attributed to exact frames without root.
"""

import os
import signal
import threading
import time

import pytest

from hermes_cli.web_server import _install_stack_dump_diagnostics

_NO_SIGUSR1 = not hasattr(signal, "SIGUSR1")


@pytest.fixture()
def stacks_log(tmp_path, monkeypatch):
    import hermes_constants

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: str(hermes_home)
    )
    return hermes_home / "logs" / "gateway-stacks.log"


@pytest.mark.skipif(_NO_SIGUSR1, reason="no SIGUSR1")
def test_installs_on_main_thread_and_dumps_all_threads(stacks_log):
    assert threading.current_thread() is threading.main_thread()
    prev = signal.getsignal(signal.SIGUSR1)
    try:
        assert _install_stack_dump_diagnostics() is True
        started = threading.Event()

        def _probe_marker_for_dump():
            started.set()
            time.sleep(1.0)

        t = threading.Thread(target=_probe_marker_for_dump, daemon=True)
        t.start()
        assert started.wait(timeout=2)
        os.kill(os.getpid(), signal.SIGUSR1)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if stacks_log.exists() and "_probe_marker_for_dump" in stacks_log.read_text():
                break
            time.sleep(0.05)
        content = stacks_log.read_text()
        assert "stack dump" in content  # timestamped marker present
        assert "_probe_marker_for_dump" in content, (
            "all_threads must include the worker's frame"
        )
    finally:
        signal.signal(signal.SIGUSR1, prev)


@pytest.mark.skipif(_NO_SIGUSR1, reason="no SIGUSR1")
def test_install_refuses_non_main_thread(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: tmp_path / "hh"
    )
    prev = signal.getsignal(signal.SIGUSR1)
    result = {}

    def worker():
        result["ok"] = _install_stack_dump_diagnostics()

    try:
        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)
        assert result["ok"] is False
        assert signal.getsignal(signal.SIGUSR1) is prev
    finally:
        signal.signal(signal.SIGUSR1, prev)
