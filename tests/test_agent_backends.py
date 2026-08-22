"""Tests for pcbworld.agent.backends.

Covers everything in this file that has real, local test surface without a
GPU or model weights: parse_tool_call_arguments()'s arithmetic-expression
hint, _collapse_repetition(), and QwenBackend._fit_to_context()/
_render_prompt() (exercised against a lightweight FakeTokenizer, not a
real one -- the actual tokenizer/generate() calls need a real model, see
scripts/run_qwen_agent_live.py for that). ScriptedBackend is already
exercised indirectly through tests/test_agent_loop.py.
"""

from __future__ import annotations

from typing import Any

from pcbworld.agent.backends import QwenBackend, _collapse_repetition, parse_tool_call_arguments


class FakeTokenizer:
    """Enough of a HF tokenizer's surface for _render_prompt()/
    _fit_to_context() to exercise their real control flow -- word-count
    'tokenization', no real model needed."""

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools=None,
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        enable_thinking: bool | None = None,
    ) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text: str, return_tensors=None) -> dict[str, list[str]]:
        return {"input_ids": text.split()}


def _make_conversation(n_turns: int, words_per_turn: int = 8) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": "system prompt " * 5},
        {"role": "user", "content": "route net_2 " * 5},
    ]
    for i in range(n_turns):
        messages.append(
            {"role": "assistant", "content": f"turn {i} reasoning and a tool call " * words_per_turn}
        )
        messages.append(
            {"role": "tool", "content": f"turn {i} result: some tool output here " * words_per_turn}
        )
    return messages


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


# -- _collapse_repetition ---------------------------------------------------


def test_collapses_the_literal_live_failure_pattern():
    """Live Colab run (Qwen3-4B): once QwenBackend's context window
    overflowed (see _fit_to_context tests below for the fix), the model's
    next generation degenerated into "_up_up_up_up..." for the entire
    2048-token thinking budget, on every remaining turn of that net's
    step budget -- 789s for one net, the actual cause of a run that looked
    "stuck" rather than genuinely reasoning."""
    garbage = "_up" * 500 + "<|im_end|>"
    result = _collapse_repetition(garbage)

    assert len(result) < 100
    assert "_up_up_up" in result  # a few repeats kept, not zero
    assert "truncated" in result
    assert result.endswith("<|im_end|>")  # trailing real content preserved


def test_normal_text_is_untouched():
    text = "Okay, the head is at (24.064, 28.335) and the target is (41.325, 25.187)."
    assert _collapse_repetition(text) == text


def test_short_legitimate_repetition_is_not_collapsed():
    """A model legitimately repeating a short phrase a FEW times (e.g.
    "no no no, wait") should not be mangled -- the pattern only fires past
    9 consecutive repeats, well beyond anything a coherent response would
    produce."""
    text = "no no no wait let me reconsider"
    assert _collapse_repetition(text) == text


# -- QwenBackend._fit_to_context / _render_prompt ---------------------------


def _backend_with_fake_tokenizer(max_seq_length: int) -> QwenBackend:
    backend = QwenBackend(max_seq_length=max_seq_length)
    backend.tokenizer = FakeTokenizer()
    return backend


def test_short_conversation_is_never_trimmed():
    backend = _backend_with_fake_tokenizer(max_seq_length=2000)
    short = _make_conversation(1)
    result = backend._fit_to_context(short, tools=[], think=False, generation_budget=200)
    assert result == short


def test_long_conversation_is_trimmed_to_fit_the_real_budget():
    """Live Colab run: net_2's conversation grew past 4096 tokens across
    many collision-recovery turns before QwenBackend ever tried to fit it
    to the model's context window -- this is the fix, run against a
    conversation shaped the same way (many turns accumulating)."""
    backend = _backend_with_fake_tokenizer(max_seq_length=2000)
    long_conversation = _make_conversation(30)

    trimmed = backend._fit_to_context(long_conversation, tools=[], think=False, generation_budget=200)

    assert len(trimmed) < len(long_conversation)
    prompt = backend._render_prompt(trimmed, [], False)
    rendered_length = len(backend.tokenizer(prompt)["input_ids"])
    assert rendered_length <= 2000 - 200 - 128  # max_seq_length - generation_budget - safety_margin


def test_trimmed_conversation_keeps_system_prompt_and_initial_user_message():
    backend = _backend_with_fake_tokenizer(max_seq_length=2000)
    long_conversation = _make_conversation(30)
    trimmed = backend._fit_to_context(long_conversation, tools=[], think=False, generation_budget=200)

    assert trimmed[0] == long_conversation[0]  # system prompt
    assert trimmed[1] == long_conversation[1]  # initial "route net X" message
    assert trimmed[-1] == long_conversation[-1]  # most recent turn preserved


def test_trimmed_conversation_names_how_many_turns_were_dropped():
    backend = _backend_with_fake_tokenizer(max_seq_length=2000)
    long_conversation = _make_conversation(30)
    trimmed = backend._fit_to_context(long_conversation, tools=[], think=False, generation_budget=200)

    placeholder_texts = [m["content"] for m in trimmed if "trimmed to fit" in m.get("content", "")]
    assert len(placeholder_texts) == 1
    assert "earlier tool call" in placeholder_texts[0]


def test_pathologically_large_single_turn_degrades_without_crashing():
    """Even the single most recent message could, in principle, alone
    exceed the budget (a long thinking trace on its own). Must not crash
    or infinite-loop -- degrades to a truncated version of that message
    rather than raising."""
    backend = _backend_with_fake_tokenizer(max_seq_length=100)
    huge_conversation = _make_conversation(20, words_per_turn=50)

    result = backend._fit_to_context(huge_conversation, tools=[], think=False, generation_budget=10)

    assert result[0] == huge_conversation[0]
    assert result[-1]["content"].endswith("[...truncated, too long to fit context...]")
