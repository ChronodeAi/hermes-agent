

import logging as _logging
import os
import signal

import pytest

from hermes_cli.web_server import _install_sighup_resilience


class _CaptureHandler(_logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_tty_gate_refuses_when_stdin_is_a_tty(monkeypatch):
    """Interactive serve (tty stdin) keeps terminal-close semantics -
    SIGHUP resilience must NOT install (review-2 F12)."""
    import sys as _sys

    class _FakeStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(_sys, "stdin", _FakeStdin())
    prev = signal.getsignal(signal.SIGHUP)
    assert _install_sighup_resilience() is False
    assert signal.getsignal(signal.SIGHUP) is prev


def test_non_clobber_existing_handler():
    """An existing Python-level SIGHUP handler must not be clobbered."""

    def _existing_handler(signum, frame):
        pass

    prev = signal.getsignal(signal.SIGHUP)
    try:
        signal.signal(signal.SIGHUP, _existing_handler)
        assert _install_sighup_resilience() is False
        assert signal.getsignal(signal.SIGHUP) is _existing_handler
    finally:
        signal.signal(signal.SIGHUP, prev)


def test_rate_limit_caps_flood_logging():
    """Ten rapid SIGHUP deliveries must log at most one attribution
    warning in the 5s window (review-2 F12)."""
    prev = signal.getsignal(signal.SIGHUP)
    try:
        assert _install_sighup_resilience() is True
        import logging

        capture = _CaptureHandler()
        lg = logging.getLogger("hermes_cli.web_server")
        lg.addHandler(capture)
        try:
            for _ in range(10):
                os.kill(os.getpid(), signal.SIGHUP)
        finally:
            lg.removeHandler(capture)
            signal.signal(signal.SIGHUP, prev)
        sighup_records = [
            rec for rec in capture.records if "SIGHUP" in rec.getMessage()
        ]
        assert len(sighup_records) <= 1, (
            "rate-limit must cap attribution logging in the flood window "
            f"(got {len(sighup_records)})"
        )
    finally:
        pass


    def emit(self, record):
        self.records.append(record)
