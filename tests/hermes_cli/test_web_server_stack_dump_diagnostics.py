"""SIGUSR1 stack-dump diagnostics for the gateway (forensics enabler).

SIGUSR1 must produce an all-thread Python traceback in
$HERMES_HOME/logs/gateway-stacks.log - crucially DURING a GIL-held C
stall on the main thread (the exact scenario it exists to attribute).
The mechanism is therefore faulthandler.register's C-level handler,
which runs without the GIL at delivery; a Python-level signal handler
would defer until the stall ends (adversarial review finding 13,
2026-08-30). Also guards wiring (finding 15): start_server must call
both install functions.
"""

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from hermes_cli.web_server import (
    _install_sighup_resilience,
    _install_stack_dump_diagnostics,
    start_server,
)

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="platform has no SIGUSR1"
)


@pytest.fixture()
def stacks_log(tmp_path, monkeypatch):
    import hermes_constants

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: str(hermes_home)
    )
    return hermes_home / "logs" / "gateway-stacks.log"


def test_installs_and_dumps_all_threads(stacks_log):
    prev = signal.getsignal(signal.SIGUSR1)
    try:
        assert threading.current_thread() is threading.main_thread()
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
        assert "_probe_marker_for_dump" in content, (
            "all_threads=True must include the worker's frame"
        )
    finally:
        import faulthandler
        from hermes_cli import web_server as _ws

        faulthandler.unregister(signal.SIGUSR1)
        _ws._stack_dump_installed = False


_OBSERVER_SCRIPT = """
import os, signal, sys, time
stacks, done, pid = sys.argv[1], sys.argv[2], int(sys.argv[3])
time.sleep(0.4)  # main thread is inside its GIL-held C call by now
os.kill(pid, signal.SIGUSR1)
landed = False
end = time.time() + 15
while time.time() < end:
    if not landed:
        try:
            if os.path.getsize(stacks) > 10:
                landed = True
        except OSError:
            pass
    if os.path.exists(done):
        print('DUMP_FIRST' if landed else 'DONE_FIRST')
        sys.exit(0)
    time.sleep(0.02)
print('DEFERRED')
"""


def test_dump_lands_during_gil_held_c_call(stacks_log):
    """Discriminating test: the main thread holds the GIL inside a long C
    call while an independent observer process sends SIGUSR1 and watches
    for the dump vs. the call-done marker. C-level delivery writes the
    dump before the marker; a Python-level handler would defer until the
    C call returns and report DONE_FIRST/DEFERRED."""
    import hashlib

    prev = signal.getsignal(signal.SIGUSR1)
    assert _install_stack_dump_diagnostics() is True
    call_done = stacks_log.parent / "call-done.marker"
    try:
        observer = subprocess.Popen(
            [sys.executable, "-c", _OBSERVER_SCRIPT, str(stacks_log),
             str(call_done), str(os.getpid())],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

        blob = b"x" * (256 * 1024 * 1024)
        for _ in range(6):  # ~1.5-2s of GIL-holding C work on main thread
            hashlib.sha256(blob).digest()
        call_done.write_text("done")

        out, _err = observer.communicate(timeout=30)
        verdict = (out or "").strip().splitlines()
        assert verdict and verdict[-1] == "DUMP_FIRST", (
            f"dump must land during the GIL-held C call; got {verdict!r}"
        )
    finally:
        import faulthandler

        try:
            faulthandler.unregister(signal.SIGUSR1)
        except (ValueError, RuntimeError):
            pass
        from hermes_cli import web_server as _ws

        _ws._stack_dump_installed = False


def test_off_thread_install_is_allowed_or_cleanly_refused(tmp_path, monkeypatch):
    """faulthandler.register installs a process-wide C handler, so an
    off-thread call is permitted (unlike signal.signal-based installs).
    Either way it must not raise and must not corrupt the main install."""
    import hermes_constants

    hermes_home = tmp_path / "hh-alt"
    hermes_home.mkdir()
    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: str(hermes_home)
    )
    result = {}

    def worker():
        result["ok"] = _install_stack_dump_diagnostics()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=5)
    assert result["ok"] in (True, False)  # must not raise from the worker


def test_start_server_wires_both_signal_installers():
    """Wiring guard (reviewer finding 15): start_server must actually call
    both install functions - the wiring was silently lost once before."""
    import inspect

    src = inspect.getsource(start_server)
    assert "_install_sighup_resilience()" in src
    assert "_install_stack_dump_diagnostics()" in src
