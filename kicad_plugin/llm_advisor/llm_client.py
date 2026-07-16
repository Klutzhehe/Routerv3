"""Minimal, dependency-free client for OpenAI / Anthropic / DeepSeek chat APIs.

Deliberately uses only the Python standard library (urllib), not the
`openai`/`anthropic` SDKs or `requests` -- KiCad ships its own bundled
Python interpreter, and `pip install`-ing third-party packages into it is
its own can of worms (different per OS, sometimes a different Python
version than your system one). Stdlib-only means this just works once
dropped into KiCad's plugins folder.
"""

import json
import os
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


def load_dotenv(path):
    """Tiny .env loader -- KEY=VALUE per line, '#' comments, no quoting
    edge cases handled beyond stripping surrounding quotes. Doesn't
    overwrite variables already set in the real environment.
    """
    if not os.path.isfile(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _post_json(url, headers, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise LLMError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"could not reach {url}: {e.reason}") from e


def _call_openai_compatible(base_url, api_key, model, prompt, timeout):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = _post_json(base_url, headers, payload, timeout)

    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"unexpected response shape: {resp}") from e


def _call_anthropic(api_key, model, prompt, timeout):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = _post_json("https://api.anthropic.com/v1/messages", headers, payload, timeout)

    try:
        return resp["content"][0]["text"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"unexpected response shape: {resp}") from e


# provider -> (env var for key, default model, call fn)
_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022"),
    "deepseek": ("DEEPSEEK_API_KEY", "deepseek-chat"),
}


def query_llm(prompt, provider=None, model=None, timeout=30):
    """Send `prompt` to the configured LLM provider, return the text reply.

    Provider/model/key are read from the environment (see .env.example):
      LLM_PROVIDER   openai | anthropic | deepseek   (default: anthropic)
      LLM_MODEL      overrides the provider's default model
      OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY
    """
    provider = (provider or os.environ.get("LLM_PROVIDER", "anthropic")).lower()

    if provider not in _PROVIDERS:
        raise LLMError(f"unknown LLM_PROVIDER '{provider}', expected one of {list(_PROVIDERS)}")

    key_env, default_model = _PROVIDERS[provider]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise LLMError(f"{key_env} is not set (check your .env file / environment)")

    model = model or os.environ.get("LLM_MODEL", default_model)

    if provider == "openai":
        return _call_openai_compatible(
            "https://api.openai.com/v1/chat/completions", api_key, model, prompt, timeout
        )
    if provider == "deepseek":
        return _call_openai_compatible(
            "https://api.deepseek.com/chat/completions", api_key, model, prompt, timeout
        )
    if provider == "anthropic":
        return _call_anthropic(api_key, model, prompt, timeout)

    raise AssertionError("unreachable")
