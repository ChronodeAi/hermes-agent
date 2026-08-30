"""Out-of-process offload for GIL-bound text work (structural fix).

Runs a module-level pure ``fn(text, **kwargs)`` in a bounded
ProcessPoolExecutor (one worker — heavy texts serialize one at a time,
never a burst) when the text exceeds a size threshold; the caller blocks
on the future with the GIL released, so the gateway's event loop keeps
ticking while the regex/C work runs in the worker.

Why: the gateway's session turns run as in-process threads, so CPU-bound
text processing in a turn holds the GIL and stalls the loop. Measured
2026-08-30 (gateway stall-storm investigation): redaction of URL-dense
text goes super-linear — 5.1s of GIL for a single 44MB call — and
redaction runs in-process on every tool result. Same class as the
2026-08-29 compression serialization stall (fixed separately).

Contract: ``offload_text_call`` returns ``None`` on any failure or
non-applicability — the caller falls back to inline execution (the
historical behavior; correct output, worst case restored). Children are
pinned with an env marker so a worker that reaches an offloading entry
point runs inline instead of spawning nested pools.
"""

import importlib
import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

# Above ~1M chars, inline regex work is already tens of milliseconds and
# the worst cases go super-linear. Below this, the subprocess round-trip
# costs more than the GIL time it can save.
DEFAULT_OFFLOAD_MIN_CHARS = 1_000_000

# Hard ceiling for one offloaded call; on expiry the caller falls back to
# inline execution — correct output, GIL-holding, i.e. the old behavior.
CALL_TIMEOUT_S = 120

_CHILD_ENV_MARKER = "_HERMES_GIL_OFFLOAD_CHILD"

_executor: ProcessPoolExecutor | None = None
_executor_lock = threading.Lock()


def _child_call(dotted_fn: str, text: str, kwargs: dict) -> str:
    """Runs in the worker: import the target by dotted path and apply it."""
    mod_name, fn_name = dotted_fn.rsplit(".", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, fn_name)(text, **kwargs)


def _child_init() -> None:
    # Pin the worker so an offloading entry point reached inside the child
    # (e.g. compression offload child calling redaction) runs inline rather
    # than spawning a nested pool.
    os.environ[_CHILD_ENV_MARKER] = "1"


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ProcessPoolExecutor(
                max_workers=1, initializer=_child_init
            )
        return _executor


def _discard_executor() -> None:
    global _executor
    with _executor_lock:
        if _executor is not None:
            try:
                _executor.shutdown(wait=False, cancel_futures=True)
            except Exception:  # pragma: no cover - shutdown best-effort
                pass
            _executor = None


def offload_text_call(dotted_fn: str, text: str, min_chars: int, **kwargs):
    """Run the dotted module-level ``fn(text, **kwargs)`` out-of-process.

    Returns the fn's result, or ``None`` when offload is not applicable or
    failed (caller must then run inline): text under ``min_chars``, already
    inside an offload child, pool unavailable, or the call failed/timed out.
    Never raises.
    """
    if not isinstance(text, str) or len(text) < min_chars:
        return None
    if os.environ.get(_CHILD_ENV_MARKER):
        return None
    try:
        executor = _get_executor()
        future = executor.submit(_child_call, dotted_fn, text, kwargs)
        return future.result(timeout=CALL_TIMEOUT_S)
    except Exception as exc:  # BrokenProcessPool, TimeoutError, spawn failure
        logger.debug(
            "offload of %s fell back inline: %s", dotted_fn, exc
        )
        _discard_executor()
        return None
