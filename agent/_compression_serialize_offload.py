# ═══════════════════════════════════════════════════════════════════════
# Out-of-process summary serialization (2026-08-29 gateway incident)
# ═══════════════════════════════════════════════════════════════════════
#
# ContextCompressor._serialize_for_summary() runs _redact_compaction_text
# (C-level regex) and strip_think_blocks over every middle-window message.
# On a ~900k-token session that is seconds of GIL-holding C work; on the
# gateway process it starves web_server.py's event loop ("event loop
# stalled 25.3s", WS heartbeat timeouts, reconnect storms). This module
# tail section offloads that serialization to a subprocess via a
# ProcessPoolExecutor: the pickled turns cross process boundaries, the
# regex/redaction work runs in a worker WITHOUT the gateway's GIL, and the
# caller blocks on future.result() (GIL released while waiting). Bounded
# to one worker so a huge session serializes one at a time, never a burst.
#
# The helper is module-level (this tail) so ProcessPoolExecutor can pickle
# it; the child re-imports the module functions it needs, so no live
# compressor object (DB handles, provider clients) is ever pickled.

# Below this rough token estimate the subprocess round-trip costs more
# than the GIL time it saves — small sessions serialize inline.
import logging
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from agent.context_compressor import logger  # noqa: E402 — shared module logger

_OUT_OF_PROCESS_SERIALIZATION_MIN_TOKENS = 150_000

# Hard ceiling on one serialization round-trip; past it the inline path
# runs as the fallback (compression degrades, never dies).
_SUMMARY_SERIALIZE_TIMEOUT_SECONDS = 300

_summary_serialize_executor = None
_summary_serialize_executor_lock = threading.Lock()


def _summary_serialize_child(compressor_state: dict, turns: list) -> str:
    """Subprocess entry: serialize turns exactly as
    ContextCompressor._serialize_for_summary does, from a plain knob dict.

    ``compressor_state`` carries only the knobs the serializer reads, so no
    live compressor object (DB handles, provider clients) is pickled.
    The redaction/think-strip helpers are re-imported from the live module
    inside the child, so behavior stays pinned to the real implementation.
    """
    from agent.context_compressor import _redact_compaction_text
    from agent.context_compressor import _MEDIA_DIRECTIVE_RE
    from agent.agent_runtime_helpers import strip_think_blocks

    parts: list = []
    content_head = int(compressor_state.get("content_head") or 6000)
    content_tail = int(compressor_state.get("content_tail") or 2000)
    content_max = int(compressor_state.get("content_max") or 12000)
    tool_args_head = int(compressor_state.get("tool_args_head") or 600)
    tool_args_max = int(compressor_state.get("tool_args_max") or 1200)
    for msg in turns:
        role = msg.get("role", "unknown")
        content = msg.get("content")
        if isinstance(content, list):
            text_parts: list = []
            for part in content:
                if isinstance(part, dict):
                    ptype = part.get("type")
                    if ptype == "text":
                        text_parts.append(part.get("text", ""))
                    elif ptype in {"image", "image_url", "input_image"}:
                        text_parts.append("[image]")
                    else:
                        text_parts.append(f"[{ptype or 'attachment'}]")
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)
        content = _redact_compaction_text(content or "")
        content = _MEDIA_DIRECTIVE_RE.sub("[media attachment]", content)
        if role == "assistant" and content:
            content = strip_think_blocks(None, content)

        if role == "tool":
            tool_id = msg.get("tool_call_id", "")
            if len(content) > content_max:
                content = (
                    content[:content_head] + "\n...[truncated]...\n"
                    + content[-content_tail:]
                )
            parts.append(f"[TOOL RESULT {tool_id}]: {content}")
            continue

        if role == "assistant":
            if len(content) > content_max:
                content = (
                    content[:content_head] + "\n...[truncated]...\n"
                    + content[-content_tail:]
                )
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "?")
                        args = _redact_compaction_text(fn.get("arguments", ""))
                        if len(args) > tool_args_max:
                            args = args[:tool_args_head] + "..."
                        tc_parts.append(f"  {name}({args})")
                    else:
                        fn = getattr(tc, "function", None)
                        name = getattr(fn, "name", "?") if fn else "?"
                        tc_parts.append(f"  {name}(...)")
                content += "\n[Tool calls:\n" + "\n".join(tc_parts) + "\n]"
            parts.append(f"[ASSISTANT]: {content}")
            continue

        if len(content) > content_max:
            content = (
                content[:content_head] + "\n...[truncated]...\n"
                + content[-content_tail:]
            )
        parts.append(f"[{role.upper()}]: {content}")

    return "\n\n".join(parts)


def _get_summary_serialize_executor():
    global _summary_serialize_executor
    with _summary_serialize_executor_lock:
        if _summary_serialize_executor is None:
            # Pin "spawn": fork in a heavily threaded gateway process is a
            # lock-held deadlock lottery (Linux default on py3.11); spawn
            # also avoids per-restart first-call re-import stalls.
            _summary_serialize_executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return _summary_serialize_executor


def _discard_summary_serialize_executor():
    """Tear down a broken/timed-out pool so the next call rebuilds it.

    Without this, one worker death (OOM while unpickling a giant payload)
    or one timeout left the pool permanently broken and every later
    attempt silently reverted to inline serialization — the pre-fix
    GIL-stall behavior."""
    global _summary_serialize_executor
    with _summary_serialize_executor_lock:
        if _summary_serialize_executor is not None:
            try:
                _summary_serialize_executor.shutdown(
                    wait=False, cancel_futures=True
                )
            except Exception:
                pass
            _summary_serialize_executor = None


def _serialize_for_summary_out_of_process(compressor, turns) -> str:
    """Serialize turns for the summarizer, off-process for big sessions.

    Small sessions (under the token floor) serialize inline — a subprocess
    round-trip costs more than the GIL time it saves. Large sessions go
    through the single-worker ProcessPoolExecutor; on any failure (pickling
    an unforkable payload, executor wedged, platform fork limits) the inline
    path runs as a fallback so compression degrades, never dies.
    """
    from agent.model_metadata import estimate_messages_tokens_rough

    try:
        est = estimate_messages_tokens_rough(turns)
    except Exception:
        est = 0
    if est < _OUT_OF_PROCESS_SERIALIZATION_MIN_TOKENS:
        return compressor._serialize_for_summary(turns)
    state = {
        "content_head": getattr(compressor, "_CONTENT_HEAD", 6000),
        "content_tail": getattr(compressor, "_CONTENT_TAIL", 2000),
        "content_max": getattr(compressor, "_CONTENT_MAX", 12000),
        "tool_args_head": getattr(compressor, "_TOOL_ARGS_HEAD", 600),
        "tool_args_max": getattr(compressor, "_TOOL_ARGS_MAX", 1200),
    }
    logger.info(
        "context compression: serializing ~%d tokens of middle window in a "
        "subprocess (GIL starvation guard)",
        est,
    )
    try:
        executor = _get_summary_serialize_executor()
        future = executor.submit(_summary_serialize_child, state, turns)
        return future.result(timeout=_SUMMARY_SERIALIZE_TIMEOUT_SECONDS)
    except Exception:
        # The pool may be broken (worker died) or the future timed out with
        # the worker still grinding — either way, rebuild the pool on the
        # next attempt instead of letting every future submit fail.
        _discard_summary_serialize_executor()
        logger.warning(
            "out-of-process summary serialization failed — falling back to "
            "inline serialization",
            exc_info=True,
        )
        return compressor._serialize_for_summary(turns)
