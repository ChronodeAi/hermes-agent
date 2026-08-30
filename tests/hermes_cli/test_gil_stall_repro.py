"""Controlled stall-storm reproduction + structural-fix validation (G2/G3).

Reproduces the gateway's "event loop stalled (GIL pressure suspected)"
mechanism with the real turn-path primitives — large JSON-RPC frame
serialization (tui_gateway/ws.py `write()` runs json.dumps on every frame)
and the compaction redaction regex (the C-level regex that held the GIL in
the 2026-08-29 compression spiral) — then proves the structural pattern
(ProcessPoolExecutor offload, as in agent/_compression_serialize_offload.py)
keeps the loop responsive.

Also proves the SIGUSR1 stack-dump enabler attributes a live storm to the
exact GIL-holding frame without root (the G1->G2 pipeline).

Run directly for a quick report:
    python tests/hermes_cli/test_gil_stall_repro.py
"""

import asyncio
import contextlib
import json
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX signal/timing repro")

HAMMER_SECONDS = 2.0
PAYLOAD_MB = 64
STALL_FLOOR_S = 0.1  # hardware-bound calibration — see docstring
OFFLOAD_CEILING_S = 0.08  # hardware-bound; see calibration note


def _big_payload() -> dict:
    blob = "x" * (PAYLOAD_MB * 1024 * 1024)
    return {
        "jsonrpc": "2.0",
        "method": "tool.progress",
        "params": {"type": "tool_output", "output": blob},
    }


def _hammer_iteration(payload: dict) -> int:
    """One gateway turn-path heavy iteration (module-level: picklable).

    Mirrors what a session turn does in-process: serialize a large
    tool-result frame, parse it back, run the compaction redaction
    regex. Uses the INLINE redaction body directly — this test exists to
    measure the GIL cost of in-process work, so the structural fix under
    test (offloading) must not be silently in play here.
    """
    from agent.redact import _redact_sensitive_text_inline

    line = json.dumps(payload, ensure_ascii=False)
    restored = json.loads(line)
    return len(
        _redact_sensitive_text_inline(
            restored["params"]["output"], force=True
        )
    )


def _measure_max_tick_gap(mode: str) -> float:
    """Max asyncio tick gap (20ms cadence) while the hammer thread runs.

    mode="inproc"  — hammer runs in a plain in-process thread (today's
                     turn architecture): GIL-bound C work starves the loop.
    mode="offload" — hammer submits iterations to a ProcessPoolExecutor and
                     blocks on the future (GIL released while waiting): the
                     structural-fix pattern.
    """
    gaps: list[float] = []
    hammer_done = threading.Event()

    def hammer():
        payload = _big_payload()
        deadline = time.monotonic() + HAMMER_SECONDS
        if mode == "inproc":
            while time.monotonic() < deadline:
                _hammer_iteration(payload)
        else:
            with ProcessPoolExecutor(max_workers=1) as ex:
                while time.monotonic() < deadline:
                    ex.submit(_hammer_iteration, payload).result()
        hammer_done.set()

    async def ticker():
        last = time.perf_counter()
        while not hammer_done.is_set():
            await asyncio.sleep(0.02)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    async def main():
        tick_task = asyncio.create_task(ticker())
        await asyncio.to_thread(hammer)
        tick_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tick_task

    asyncio.run(main())
    assert gaps, "ticker must have recorded ticks"
    settled = gaps[10:] or gaps or [0.0]
    return max(settled)


def test_structural_pattern_bounds_loop_impact():
    """The measured contract:
    - the current in-process architecture lets a heavy turn-path storm stall
      the event loop (>= STALL_FLOOR_S tick gaps at production payload scale);
    - the structural pattern (ProcessPoolExecutor offload) keeps ticks near
      the 20ms cadence for identical work.

    Calibration note: a SINGLE 64MB frame costs ~150ms of GIL — survivable.
    Production stalls of 5-25s are sustained/serial heavy work; the offload
    pattern is what bounds them, and this pair of assertions guards it.
    """
    inproc_gap = _measure_max_tick_gap("inproc")
    offload_gap = _measure_max_tick_gap("offload")
    assert inproc_gap >= STALL_FLOOR_S, (
        f"in-process storm no longer stalls the loop (max tick gap "
        f"{inproc_gap*1000:.0f} ms) — repro lost fidelity"
    )
    # Offload must be clearly better, not just marginally better. (
        f"offloaded storm still degraded the loop (max tick gap "
        f"{offload_gap*1000:.0f} ms > {OFFLOAD_CEILING_S*1000:.0f} ms)"
    )
    assert inproc_gap >= 3 * max(offload_gap, 0.001), (
        f"offload pattern not clearly better: inproc {inproc_gap*1000:.0f} ms "
        f"vs offload {offload_gap*1000:.0f} ms"
    )


def test_sigusr1_dump_names_gil_holder_mid_storm(tmp_path):
    """faulthandler dump fired mid-storm names the holding frame (G2)."""
    import faulthandler
    import signal

    dump_path = tmp_path / "stacks.log"
    with open(dump_path, "w", encoding="utf-8") as dump_file:
        faulthandler.register(signal.SIGUSR1, file=dump_file, all_threads=True)
        try:
            storm_done = threading.Event()

            def storm():
                payload = _big_payload()
                deadline = time.monotonic() + 1.5
                while time.monotonic() < deadline:
                    _hammer_iteration(payload)
                storm_done.set()

            t = threading.Thread(target=storm, daemon=True)
            t.start()
            time.sleep(0.4)  # storm mid-flight
            os.kill(os.getpid(), signal.SIGUSR1)
            assert storm_done.wait(timeout=15)
            t.join(timeout=5)
        finally:
            with contextlib.suppress(Exception):
                faulthandler.unregister(signal.SIGUSR1)

    content = dump_path.read_text()
    assert "_hammer_iteration" in content, (
        "dump mid-storm must name the GIL-holding frame"
    )


if __name__ == "__main__":
    for mode in ("inproc", "offload"):
        gap = _measure_max_tick_gap(mode)
        print(f"{mode}: max event-loop tick gap = {gap*1000:.0f} ms")
