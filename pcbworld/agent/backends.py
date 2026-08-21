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

        return None, f"Malformed JSON arguments: {err} in text: {raw_str!r}"


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
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
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
                max_seq_length=4096,
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

        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tools=tools if tools else None,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=think,
                )
            except TypeError:
                # This tokenizer's template predates enable_thinking -- the
                # appended /think or /no_think marker above is what is left
                # doing the work; not a hard failure.
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        formatted_messages,
                        tools=tools if tools else None,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except Exception:
                    prompt_text = self.tokenizer.apply_chat_template(
                        formatted_messages,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
            except Exception:
                # tools argument not accepted by this template at all
                prompt_text = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=think,
                )
        else:
            prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in formatted_messages)

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)

        # Parse output for reasoning and tool calls
        reasoning, clean_text = extract_reasoning(output_text)
        tool_calls = self._parse_tool_calls_from_text(clean_text)

        return BackendReply(
            text=clean_text,
            reasoning=reasoning,
            tool_calls=tool_calls,
            raw=output_text,
        )

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
