"""Tests for pcbworld.agent.backends.

Scoped to parse_tool_call_arguments()'s arithmetic-expression hint --
everything else in this file (ScriptedBackend, QwenBackend's model-loading
and generate() paths) either has no local test surface (needs a real GPU
and model weights, see scripts/run_qwen_agent_live.py for that) or is
already exercised indirectly through tests/test_agent_loop.py's use of
ScriptedBackend. Not attempting broader coverage here that nothing has
asked for yet.
"""

from __future__ import annotations

from pcbworld.agent.backends import parse_tool_call_arguments


def test_valid_json_parses_with_no_hint():
    result, err = parse_tool_call_arguments('{"x_mm": 29.823, "y_mm": 15.357}')
    assert result == {"x_mm": 29.823, "y_mm": 15.357}
    assert err is None


def test_arithmetic_expression_in_json_value_gets_a_correction_hint():
    """Live Colab run (Qwen3-4B): the model wrote {"x_mm": 21.823 + 8.0, ...}
    -- an unevaluated expression, invalid as a JSON value -- and made the
    BYTE-IDENTICAL mistake again 8 turns later in the same net's turn,
    with its own earlier correction still sitting in its own context. A
    generic "malformed JSON" message alone apparently isn't a strong
    enough signal for a model this size to reliably generalize from across
    that many turns; restating the fix inline, every time this exact
    pattern recurs, costs nothing and doesn't depend on model memory."""
    raw = '{"name": "route_to", "arguments": {"x_mm": 21.823 + 8.0, "y_mm": 15.357}}'
    result, err = parse_tool_call_arguments(raw)

    assert result is None
    assert err is not None
    assert "HINT" in err
    assert "plain numbers" in err
    assert "29.823" in err  # the corrected numeric value is shown as an example


def test_unrelated_malformed_json_gets_no_arithmetic_hint():
    result, err = parse_tool_call_arguments("not json at all {{{")
    assert result is None
    assert "HINT" not in err


def test_dict_passthrough_still_works():
    """Some callers already hand a parsed dict (not a JSON string) --
    confirm the hint logic (string-only) doesn't interfere with that path."""
    result, err = parse_tool_call_arguments({"x_mm": 1.0})
    assert result == {"x_mm": 1.0}
    assert err is None
