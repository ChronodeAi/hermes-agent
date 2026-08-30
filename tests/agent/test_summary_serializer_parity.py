"""Serialization parity: off-process child == real compressor serializer.

Adversarial review finding #5: the out-of-process child once diverged at
birth — it hardcoded "[image]" where the real path preserves image URLs
via _image_part_label. Both paths now call the ONE canonical
agent.context_compressor.serialize_turns_for_summary; this test drives
the real ContextCompressor object end-to-end (non-circular: the
reference is the real method, not the child itself).
"""

from agent._compression_serialize_offload import _summary_serialize_child
from agent.context_compressor import ContextCompressor


def _knobs() -> dict:
    return {
        "content_head": 4000,
        "content_tail": 1500,
        "content_max": 6000,
        "tool_args_head": 1200,
        "tool_args_max": 1500,
    }


def _sample_turns() -> list:
    return [
        {"role": "user", "content": "please analyze the chart"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "remote image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://pics.example.com/chart.png"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {"type": "screenshot", "path": "/tmp/x.png"},
            ],
        },
        {
            "role": "assistant",
            "content": "answer",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/etc/hosts"}',
                    }
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "file contents here"},
        {"role": "user", "content": "and the second question"},
    ]


def test_child_matches_real_compressor_serializer():
    """The child path must equal the REAL ContextCompressor._serialize_for_summary
    output byte-for-byte."""
    real = ContextCompressor.__new__(ContextCompressor)
    # The serializer only reads the class-level knob constants; skip the
    # heavy engine __init__ entirely.
    turns = _sample_turns()

    inproc = real._serialize_for_summary(turns)
    child = _summary_serialize_child(_knobs(), turns)
    assert child == inproc


def test_remote_image_urls_survive_serialization():
    """The birth defect: remote image labels must keep a referenceable URL
    (base64 data URLs correctly collapse)."""
    real = ContextCompressor.__new__(ContextCompressor)
    out = real._serialize_for_summary(_sample_turns())
    assert "[image: https://pics.example.com/chart.png]" in out
    assert "data:image/png;base64" not in out or "[image]" in out
    assert out != real._serialize_for_summary([])


def test_long_content_truncates_identically():
    long_text = "x" * 20_000
    turns = [{"role": "tool", "tool_call_id": "t9", "content": long_text}]
    real = ContextCompressor.__new__(ContextCompressor)
    inproc = real._serialize_for_summary(turns)
    child = _summary_serialize_child(_knobs(), turns)
    assert child == inproc
    assert "...[truncated]..." in child
