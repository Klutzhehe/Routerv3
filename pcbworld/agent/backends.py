"""Model backend interface for the PCB routing agent.

Provides a unified interface `Backend` decoupling the agent loop from the
underlying LLM inference engine.

Implementations:
- `ScriptedBackend`: Deterministic, canned sequence of replies for offline testing
  without GPU or KiCad dependencies.
- `QwenBackend`: T4-safe 4-bit fp16 backend for Qwen3's dense small tier (4B by
  default, 8B fits the same 16GB budget at Q4) with thinking trace extraction and
  robust tool argument parsing.

Model choice, decided outside this file, not to be silently swapped: Qwen3, not
Qwen2.5 or any other family. It is the only local-sized option with all three
properties this agent needs -- native tool calling, an explicit <think> trace, and
a per-turn thinking toggle -- and the toggle is what keeps a board's wall clock in
minutes rather than the ~70 it costs to think on every one of ~70 net-level tool
calls. See `chat()`'s `think` handling below for how the toggle is actually applied
(Qwen3's `enable_thinking` chat-template kwarg, not a hand-appended string -- an
appended "/no_think" is Qwen3's own documented FALLBACK for suppressing thinking
mid-conversation when a caller cannot pass the kwarg, not the primary mechanism,
and it does nothing at all on a non-Qwen3 model).

Colab Setup for QwenBackend:
    !pip install -q unsloth transformers accelerate bitsandbytes
    from pcbworld.agent.backends import QwenBackend
    backend = QwenBackend(model_name="Qwen/Qwen3-4B", load_in_4bit=True)
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Callable, Protocol, Sequence


@dataclasses.dataclass
class ToolCall:
    """A tool call requested by the model."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None


@dataclasses.dataclass
class BackendReply:
    """Standard reply from a Backend."""

    text: str = ""
    reasoning: str | None = None
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    raw: Any = None


class Backend(Protocol):
    """Protocol for model backends."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        think: bool,
    ) -> BackendReply:
        """Generates a reply given message history, available tools, and think toggle."""
        ...


def extract_reasoning(text: str) -> tuple[str | None, str]:
    """Extracts <think>...</think> content and returns (reasoning, clean_text)."""
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        clean_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return reasoning, clean_text
    return None, text.strip()


# A short substring (1-20 chars) immediately repeated 10+ times in a row --
# the live failure signature this exists for: once QwenBackend's context
# window overflowed, the model's next generation degenerated into
# "_up_up_up_up..." for the entire 2048-token budget, on every remaining
# turn of that net's step budget (789s for one net). Non-greedy so the
# SHORTEST repeating unit is found first (matches "_up" rather than some
# longer, coincidentally-also-repeating superstring).
_REPETITION_PATTERN = re.compile(r"(.{1,20}?)\1{9,}", re.DOTALL)


def _collapse_repetition(text: str) -> str:
    """Collapses a detected repetition run down to a few repeats plus a
    note, instead of passing hundreds-to-thousands of tokens of garbage
    through as if it were real content. Matters beyond readability: the
    result becomes BackendReply.text, which the agent loop appends back
    into conversation history every turn -- left uncollapsed, it would
    feed its own garbage back in and make the NEXT turn's prompt even
    longer, compounding the exact context-overflow problem that produced
    it in the first place."""

    def _replace(match: "re.Match[str]") -> str:
        unit = match.group(1)
        return f"{unit}{unit}{unit} [...repeated pattern truncated...]"

    return _REPETITION_PATTERN.sub(_replace, text)


# Matches "21.823 + 8.0" / "29.5 - 3" style unevaluated arithmetic inside a
# JSON value -- number, operator, number. Deliberately loose (a JSON key
# that happens to contain digits and a hyphen could false-positive) since
# a false positive here just adds an irrelevant-but-harmless hint to an
# already-malformed-JSON error; the cost of missing a real occurrence is
# higher than the cost of an occasional unnecessary hint.
_ARITHMETIC_IN_JSON_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?")


def parse_tool_call_arguments(raw_args: str | dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Safely parses tool call arguments.

    Returns (parsed_dict, None) on success, or (None, error_description) on failure.
    """
    if isinstance(raw_args, dict):
        return raw_args, None

    if not isinstance(raw_args, str):
        return None, f"Expected JSON string or dict for tool arguments, got {type(raw_args).__name__}"

    raw_str = raw_args.strip()
    if not raw_str:
        return {}, None

    try:
        parsed = json.loads(raw_str)
        if isinstance(parsed, dict):
            return parsed, None
        return None, f"Parsed JSON argument is not an object/dict: {parsed!r}"
    except json.JSONDecodeError as err:
        # Attempt simple repair for common LLM single quote / markdown wrap issues
        cleaned = raw_str
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception:
            pass

        hint = ""
        if _ARITHMETIC_IN_JSON_PATTERN.search(raw_str):
            # Measured live (Qwen3-4B, two Colab runs): the model wrote an
            # unevaluated expression as a JSON value -- {"x_mm": 21.823 +
            # 8.0} -- and made the BYTE-IDENTICAL mistake again 8 turns
            # later in the same net's turn, with its own earlier correction
            # still sitting in context. A generic parse-error message
            # apparently isn't a strong enough correction signal on its own
            # for a model this size to reliably generalize from; restating
            # the fix explicitly, every single time this exact pattern
            # recurs, costs nothing and doesn't depend on the model
            # remembering a correction from many turns back.
            hint = (
                " HINT: JSON values must be plain numbers, not expressions -- "
                "you wrote something like 'a + b'. Compute the result "
                "yourself first, then pass only the final number, e.g. "
                '{"x_mm": 29.823} not {"x_mm": 21.823 + 8.0}.'
            )

        return None, f"Malformed JSON arguments: {err} in text: {raw_str!r}{hint}"


class ScriptedBackend:
    """Deterministic backend returning a pre-programmed sequence of replies or tool calls.

    Essential for testing the agent loop without GPU or real router dependencies.
    """

    def __init__(
        self,
        script: Sequence[BackendReply | list[ToolCall] | ToolCall | str | Callable[..., BackendReply]] | None = None,
    ) -> None:
        self.script: list[Any] = list(script or [])
        self.call_history: list[dict[str, Any]] = []
        self._index = 0

    def add_step(self, item: BackendReply | list[ToolCall] | ToolCall | str | Callable[..., BackendReply]) -> None:
        self.script.append(item)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        think: bool,
    ) -> BackendReply:
        self.call_history.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "think": think,
                "step_index": self._index,
            }
        )

        if self._index >= len(self.script):
            return BackendReply(
                text="[ScriptedBackend exhausted]",
                reasoning="No more scripted actions available.",
                tool_calls=[],
            )

        item = self.script[self._index]
        self._index += 1

        if callable(item):
            return item(messages, tools, think)

        if isinstance(item, BackendReply):
            return item

        if isinstance(item, ToolCall):
            return BackendReply(
                text=f"Calling {item.name}",
                reasoning="Scripted step reasoning" if think else None,
                tool_calls=[item],
            )

        if isinstance(item, list):
            # List of ToolCall objects
            return BackendReply(
                text="Calling scripted tools",
                reasoning="Scripted multi-tool step" if think else None,
                tool_calls=item,
            )

        if isinstance(item, str):
            reasoning, text = extract_reasoning(item)
            return BackendReply(text=text, reasoning=reasoning, tool_calls=[])

        raise TypeError(f"Unsupported script item type: {type(item)}")


class QwenBackend:
    """T4-friendly backend for Qwen open-weights models (fp16, 4-bit quantized).

    Designed for Google Colab T4 GPUs (Turing SM 7.5).
    - Uses 4-bit quantization to fit comfortably in 16GB VRAM alongside PNS bridge.
    - Uses fp16 (no bf16 or FlashAttention-2 which requires SM 8.0+).
    - Extracts reasoning from <think> tags.
    - Robustly parses tool call arguments with error reporting.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        load_in_4bit: bool = True,
        device: str = "cuda",
        max_new_tokens: int = 512,
        max_new_tokens_thinking: int = 2048,
        temperature: float = 0.2,
        max_seq_length: int = 16384,
    ) -> None:
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = device
        # Two budgets, not one -- a `think=True` call has to fit a whole
        # <think>...</think> trace (500-2000 tokens on its own, per
        # docs/AI_ARCHITECTURE.md's own estimate for this model class)
        # AND the tool call that follows it. A single fixed budget sized
        # for the common no_think case truncates the rarer thinking case
        # mid-thought, before any tool call is ever emitted -- the loop
        # would see zero tool_calls and burn a step for nothing, on
        # exactly the turns (post-error) where progress matters most.
        self.max_new_tokens = max_new_tokens
        self.max_new_tokens_thinking = max_new_tokens_thinking
        self.temperature = temperature
        # Live Colab run (Qwen3-4B): a net stuck in repeated collision
        # recovery accumulated conversation history past this value at its
        # OLD default of 4096, and Unsloth's own automatic truncation
        # ("we shall truncate it ourselves") on overflow is not
        # context-aware -- the model's next response came back as a
        # fragment ("_up") and then degenerated into a pure repetition
        # loop for the rest of that net's budget (789s for one net, the
        # actual cause of "extremely long", not slow reasoning). Qwen3
        # natively supports far more than 4096; this alone buys real
        # headroom, though chat()'s own history-trimming (below) is the
        # structural fix -- this is a safety margin under it, not a
        # substitute for it.
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Unsloth fast path if installed, otherwise standard transformers
        try:
            from unsloth import FastLanguageModel

            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
                dtype=torch.float16,
            )
            FastLanguageModel.for_inference(self.model)
        except ImportError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )

        self._initialized = True

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        think: bool,
    ) -> BackendReply:
        self._lazy_init()
        import torch

        # Qwen3's documented, primary thinking toggle is the chat template's own
        # `enable_thinking` kwarg -- it inserts (or pre-fills empty) the
        # <think>...</think> scaffold before generation starts, which is a
        # stronger signal than asking the model to comply with an instruction.
        # The appended "/think" / "/no_think" text is Qwen3's own FALLBACK for
        # suppressing/enabling thinking on a template that does not expose the
        # kwarg (e.g. an older tokenizer_config.json) -- kept as a second layer,
        # not the primary mechanism.
        formatted_messages = list(messages)
        marker = "/think" if think else "/no_think"
        if formatted_messages and formatted_messages[0]["role"] == "system":
            orig_content = formatted_messages[0]["content"]
            if marker not in orig_content:
                formatted_messages[0] = {
                    "role": "system",
                    "content": orig_content + f"\n{marker}",
                }

        # Reserve room for the generation itself, then trim conversation
        # HISTORY (not the system prompt or the most recent turns) to fit
        # what's left. Structural fix, not just a bigger ceiling: a long
        # collision-recovery saga can grow without bound turn over turn,
        # and letting the tokenizer/Unsloth silently truncate on overflow
        # is what produced this exact failure live -- the model's next
        # response came back as a fragment, then degenerated into a pure
        # repetition loop for the rest of that net's step budget (789s for
        # one net). Trimming ourselves, before that point, keeps the
        # prompt coherent -- the model is told what was dropped, rather
        # than silently losing the start or end of its own conversation.
        generation_budget = self.max_new_tokens_thinking if think else self.max_new_tokens
        formatted_messages = self._fit_to_context(formatted_messages, tools, think, generation_budget)

        prompt_text = self._render_prompt(formatted_messages, tools, think)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=generation_budget,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)

        # Collapse degenerate repetition (the live failure mode above,
        # once context breaks) BEFORE it becomes reply.text -- loop.py
        # appends reply.text back into conversation history every turn, so
        # an uncollapsed repetition loop would feed its own garbage back
        # in and make the NEXT turn's context even longer, compounding
        # exactly the problem that caused it.
        output_text = _collapse_repetition(output_text)

        # Parse output for reasoning and tool calls
        reasoning, clean_text = extract_reasoning(output_text)
        tool_calls = self._parse_tool_calls_from_text(clean_text)

        return BackendReply(
            text=clean_text,
            reasoning=reasoning,
            tool_calls=tool_calls,
            raw=output_text,
        )

    def _render_prompt(self, formatted_messages: list[dict[str, Any]], tools, think: bool) -> str:
        """The chat-template application, factored out so _fit_to_context()
        can measure a candidate message list's real token length the same
        way chat() renders it for real -- these must stay in exact sync,
        or the trim loop measures something different than what actually
        gets sent to generate()."""
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return "\n".join(f"{m['role']}: {m['content']}" for m in formatted_messages)

        try:
            return self.tokenizer.apply_chat_template(
                formatted_messages,
                tools=tools if tools else None,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=think,
            )
        except TypeError:
            # This tokenizer's template predates enable_thinking -- the
            # appended /think or /no_think marker in chat() is what is
            # left doing the work; not a hard failure.
            try:
                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tools=tools if tools else None,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
        except Exception:
            # tools argument not accepted by this template at all
            return self.tokenizer.apply_chat_template(
                formatted_messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=think,
            )

    def _fit_to_context(
        self,
        formatted_messages: list[dict[str, Any]],
        tools,
        think: bool,
        generation_budget: int,
        keep_recent: int = 6,
    ) -> list[dict[str, Any]]:
        """Drops the OLDEST middle of the conversation -- keeping the
        system prompt, the net's initial "route net X" user message, and
        the most recent turns -- until the rendered prompt fits under
        max_seq_length, leaving room for generation_budget tokens of
        output. The dropped middle is replaced with one placeholder
        message so the model knows context was trimmed rather than
        silently missing pieces of its own history.

        Recent turns are what a routing decision actually depends on (the
        current head position, the last few tool results); the early
        history of a long recovery saga is not load-bearing the same way.

        Iterative, not a single fixed-size tail: `keep_recent` messages
        are not a fixed token cost -- several could each carry a full
        thinking trace (up to max_new_tokens_thinking on their own), so a
        naive single trim to a FIXED message count could still overflow.
        Shrinks the kept tail one turn at a time until it actually fits,
        down to a floor of the single most recent message (the one thing
        that must never be dropped -- without it the model has no idea
        where the head currently is).
        """
        safety_margin = 128  # chat-template special tokens, rounding
        token_budget = self.max_seq_length - generation_budget - safety_margin

        def _measure(msgs: list[dict[str, Any]]) -> int:
            text = self._render_prompt(msgs, tools, think)
            return len(self.tokenizer(text)["input_ids"])

        if _measure(formatted_messages) <= token_budget or len(formatted_messages) <= 3:
            return formatted_messages

        head = formatted_messages[:2]  # system prompt, initial user message

        for tail_size in range(min(keep_recent, len(formatted_messages) - 2), 0, -1):
            tail = formatted_messages[-tail_size:]
            dropped = len(formatted_messages) - len(head) - len(tail)
            if dropped <= 0:
                continue
            placeholder = {
                "role": "user",
                "content": (
                    f"[{dropped} earlier tool call(s)/result(s) in this net's attempt "
                    f"were trimmed to fit the model's context window and are no longer "
                    f"visible. Continue from the most recent state shown below.]"
                ),
            }
            candidate = head + [placeholder] + tail
            if _measure(candidate) <= token_budget:
                return candidate

        # Even head + placeholder + the single most recent message doesn't
        # fit -- that one message's own content is pathologically large.
        # Truncate its content directly rather than dropping it outright;
        # the model still needs to know it exists and roughly what it said.
        last = formatted_messages[-1]
        truncated_last = {
            **last,
            "content": str(last.get("content", ""))[:2000] + " [...truncated, too long to fit context...]",
        }
        dropped = len(formatted_messages) - len(head) - 1
        placeholder = {
            "role": "user",
            "content": (
                f"[{dropped} earlier tool call(s)/result(s) were trimmed to fit the "
                f"model's context window.]"
            ),
        }
        return head + [placeholder, truncated_last]

    def _parse_tool_calls_from_text(self, text: str) -> list[ToolCall]:
        """Parses tool calls from model output text using standard Qwen tool format or JSON blocks."""
        tool_calls: list[ToolCall] = []

        # 1. Match <tool_call> ... </tool_call> tags
        tc_pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        for match in re.finditer(tc_pattern, text, re.DOTALL):
            block = match.group(1).strip()
            args_dict, err = parse_tool_call_arguments(block)
            if args_dict and "name" in args_dict:
                name = args_dict["name"]
                args = args_dict.get("arguments", {})
                if isinstance(args, str):
                    sub_args, sub_err = parse_tool_call_arguments(args)
                    args = sub_args if sub_args is not None else {"_parse_error": sub_err, "raw": args}
                tool_calls.append(ToolCall(name=name, arguments=args))
            elif args_dict:
                # Raw dict without name wrapper
                name = args_dict.get("tool", args_dict.get("function", "unknown"))
                tool_calls.append(ToolCall(name=name, arguments=args_dict))
            else:
                tool_calls.append(ToolCall(name="_parse_error", arguments={"error": err, "raw": block}))

        if tool_calls:
            return tool_calls

        # 2. Match markdown code blocks ```json { "name": ..., "arguments": ... } ```
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        for match in re.finditer(json_pattern, text, re.DOTALL):
            block = match.group(1).strip()
            args_dict, err = parse_tool_call_arguments(block)
            if args_dict and "name" in args_dict:
                name = args_dict["name"]
                args = args_dict.get("arguments", args_dict)
                if isinstance(args, str):
                    sub_args, sub_err = parse_tool_call_arguments(args)
                    args = sub_args if sub_args is not None else {"_parse_error": sub_err, "raw": args}
                tool_calls.append(ToolCall(name=name, arguments=args))

        return tool_calls
