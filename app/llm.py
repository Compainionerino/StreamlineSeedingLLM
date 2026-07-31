from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

from .code_validation import validate_generated_code


class LLMError(RuntimeError):
    """Raised when code generation through LiteLLM fails."""


DEFAULT_MAX_TOKENS = 50000
MAX_TOKEN_LIMIT = 200000
DEFAULT_MAX_CONTINUATIONS = 2
DEFAULT_REPAIR_ATTEMPTS = 2
BLABLADOR_API_BASE = "https://api.blablador.fz-juelich.de/v1/"
PROVIDER_MODEL_PREFIXES = {
    "openai": "openai/",
    "anthropic": "anthropic/",
    "gemini": "gemini/",
    "blablador": "openai/",
}
PROVIDER_API_KEY_ENV_VARS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "blablador": ("BLABLADOR_API_KEY", "BLABLADOOR_API_KEY"),
}
PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
    "blablador": "Blablador",
}
PROVIDER_DEFAULT_API_BASES = {
    "blablador": BLABLADOR_API_BASE,
}
PROVIDER_ENV_API_KEY_FOR_COMPLETION = {
    "blablador",
}


def normalize_model_name(model: str, provider: str = "custom") -> str:
    cleaned = model.strip()
    prefix = PROVIDER_MODEL_PREFIXES.get(provider.strip().lower(), "")
    if not cleaned or not prefix or "/" in cleaned:
        return cleaned
    return prefix + cleaned


def provider_from_model_name(model: str) -> str | None:
    cleaned = model.strip().lower()
    for provider, prefix in PROVIDER_MODEL_PREFIXES.items():
        if cleaned.startswith(prefix):
            return provider
    return None


def _current_model_provider(settings: LLMSettings) -> str | None:
    selected_provider = settings.provider.strip().lower()
    if selected_provider == "blablador":
        return selected_provider
    resolved_model = normalize_model_name(settings.model, settings.provider)
    return provider_from_model_name(resolved_model)


def _windows_persistent_env_var(name: str) -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg
    except ImportError:
        return ""

    locations = (
        (winreg.HKEY_CURRENT_USER, r"Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for root, path in locations:
        try:
            with winreg.OpenKey(root, path) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if isinstance(value, str) and value:
            return value
    return ""


def _provider_env_api_key(provider: str) -> str:
    for name in PROVIDER_API_KEY_ENV_VARS.get(provider, ()):
        value = os.environ.get(name)
        if value:
            return value
        value = _windows_persistent_env_var(name)
        if value:
            return value
    return ""


def _has_provider_api_key(provider: str, explicit_api_key: str = "") -> bool:
    if explicit_api_key.strip():
        return True
    return bool(_provider_env_api_key(provider))


def _validate_api_key_for_current_model(settings: LLMSettings) -> None:
    provider = _current_model_provider(settings)
    if provider is None or _has_provider_api_key(provider, settings.api_key):
        return

    env_vars = PROVIDER_API_KEY_ENV_VARS.get(provider, ())
    label = PROVIDER_LABELS.get(provider, provider)
    resolved_model = normalize_model_name(settings.model, settings.provider)
    raise LLMError(
        f"{label} API key is required for the selected model {resolved_model!r}. "
        f"Enter an API key in the LLM panel or set {' or '.join(env_vars)}."
    )


@dataclass(frozen=True)
class LLMSettings:
    model: str
    provider: str = "custom"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.2
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS
    repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS


@dataclass(frozen=True)
class LLMResponse:
    content: str
    code: str
    model: str
    finish_reason: str | None = None
    continuation_count: int = 0
    repair_attempts: int = 0
    token_usage: dict[str, Any] | None = None
    validation_errors: list[str] | None = None
    raw: Any = None


@dataclass(frozen=True)
class LLMCallUsage:
    label: str
    model: str
    finish_reason: str | None
    input_tokens: int | float | None = None
    output_tokens: int | float | None = None
    total_tokens: int | float | None = None
    raw_usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "raw_usage": self.raw_usage,
        }


def extract_code_block(content: str) -> str:
    text = content.strip()
    if "```" not in text:
        return text

    blocks = text.split("```")
    for index in range(1, len(blocks), 2):
        block = blocks[index].strip()
        if block.startswith("python"):
            return block[len("python") :].strip()
    return blocks[1].strip()


def _message_content(response: Any) -> str:
    try:
        return response["choices"][0]["message"]["content"]
    except Exception:
        pass

    try:
        return response.choices[0].message.content
    except Exception as exc:
        raise LLMError("LiteLLM response did not contain message content.") from exc


def _finish_reason(response: Any) -> str | None:
    try:
        return response["choices"][0].get("finish_reason")
    except Exception:
        pass

    try:
        return response.choices[0].finish_reason
    except Exception:
        return None


def _is_truncated(response: Any) -> bool:
    finish_reason = _finish_reason(response)
    return finish_reason in {"length", "max_tokens", "content_filter_length"}


def _plain_data(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        try:
            return _plain_data(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _plain_data(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _plain_data(vars(value))
    return repr(value)


def _response_usage(response: Any) -> dict[str, Any] | None:
    try:
        usage = response["usage"]
    except Exception:
        usage = getattr(response, "usage", None)
    if usage is None:
        return None
    usage_payload = _plain_data(usage)
    return usage_payload if isinstance(usage_payload, dict) else {"usage": usage_payload}


def _usage_number(usage: dict[str, Any] | None, *keys: str) -> int | float | None:
    if usage is None:
        return None
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _call_usage(call_label: str, model: str, response: Any) -> LLMCallUsage:
    raw_usage = _response_usage(response)
    input_tokens = _usage_number(raw_usage, "prompt_tokens", "input_tokens", "prompt_token_count", "input_token_count")
    output_tokens = _usage_number(
        raw_usage,
        "completion_tokens",
        "output_tokens",
        "completion_token_count",
        "output_token_count",
        "candidates_token_count",
    )
    total_tokens = _usage_number(raw_usage, "total_tokens", "total_token_count")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return LLMCallUsage(
        label=call_label,
        model=model,
        finish_reason=_finish_reason(response),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        raw_usage=raw_usage,
    )


def _sum_optional(values: Sequence[int | float | None]) -> int | float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return sum(numeric)


def _format_token_value(value: Any) -> str:
    return str(value) if value is not None else "not reported"


def _aggregate_token_usage(call_usages: Sequence[LLMCallUsage]) -> dict[str, Any] | None:
    if not call_usages:
        return None
    return {
        "calls": len(call_usages),
        "total": {
            "input_tokens": _sum_optional([usage.input_tokens for usage in call_usages]),
            "output_tokens": _sum_optional([usage.output_tokens for usage in call_usages]),
            "total_tokens": _sum_optional([usage.total_tokens for usage in call_usages]),
        },
        "per_call": [usage.to_dict() for usage in call_usages],
    }


def _print_token_usage_summary(
    *,
    model: str,
    continuation_count: int,
    repair_attempts: int,
    call_usages: Sequence[LLMCallUsage],
) -> None:
    print("[streamline-rag] ===== LLM USAGE SUMMARY =====", flush=True)
    print(f"[streamline-rag] Model: {model}", flush=True)
    print(f"[streamline-rag] Calls: {len(call_usages)}", flush=True)
    print(f"[streamline-rag] Continuations: {continuation_count}", flush=True)
    print(f"[streamline-rag] Repairs: {repair_attempts}", flush=True)
    print("[streamline-rag]", flush=True)
    print("[streamline-rag] Response stack:", flush=True)
    if not call_usages:
        print("[streamline-rag] token usage: not reported by provider for any call", flush=True)
    for index, usage in enumerate(call_usages, start=1):
        print(f"[streamline-rag] {index}. {usage.label}", flush=True)
        print(f"[streamline-rag]    finish_reason: {_format_token_value(usage.finish_reason)}", flush=True)
        print(f"[streamline-rag]    input_tokens: {_format_token_value(usage.input_tokens)}", flush=True)
        print(f"[streamline-rag]    output_tokens: {_format_token_value(usage.output_tokens)}", flush=True)
        print(f"[streamline-rag]    total_tokens: {_format_token_value(usage.total_tokens)}", flush=True)
    totals = _aggregate_token_usage(call_usages)
    print("[streamline-rag]", flush=True)
    print("[streamline-rag] Total:", flush=True)
    if totals is None:
        print("[streamline-rag] input_tokens: not reported", flush=True)
        print("[streamline-rag] output_tokens: not reported", flush=True)
        print("[streamline-rag] total_tokens: not reported", flush=True)
    else:
        total = totals["total"]
        print(f"[streamline-rag] input_tokens: {_format_token_value(total['input_tokens'])}", flush=True)
        print(f"[streamline-rag] output_tokens: {_format_token_value(total['output_tokens'])}", flush=True)
        print(f"[streamline-rag] total_tokens: {_format_token_value(total['total_tokens'])}", flush=True)


def _completion_kwargs(settings: LLMSettings, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
    selected_provider = settings.provider.strip().lower()
    kwargs: dict[str, Any] = {
        "model": normalize_model_name(settings.model, settings.provider),
        "messages": list(messages),
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    elif selected_provider in PROVIDER_ENV_API_KEY_FOR_COMPLETION:
        env_api_key = _provider_env_api_key(selected_provider)
        if env_api_key:
            kwargs["api_key"] = env_api_key

    api_base = settings.api_base.strip() or PROVIDER_DEFAULT_API_BASES.get(selected_provider, "")
    if api_base:
        kwargs["api_base"] = api_base
    return kwargs


def _call_completion(
    settings: LLMSettings,
    messages: Sequence[dict[str, str]],
    *,
    call_label: str,
) -> tuple[Any, LLMCallUsage]:
    _validate_api_key_for_current_model(settings)
    try:
        from litellm import completion  # type: ignore
    except ImportError as exc:
        raise LLMError(
            "LiteLLM is required for LLM code generation. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    kwargs = _completion_kwargs(settings, messages)
    try:
        response = completion(**kwargs)
    except Exception as exc:
        raise LLMError(f"LiteLLM generation failed: {exc}") from exc
    return response, _call_usage(call_label, kwargs["model"], response)


def _initial_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You generate concise, executable pure Python VTK visualization code. "
                "Return one complete Python script and no explanatory prose."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def _continue_truncated_response(
    *,
    prompt: str,
    partial_content: str,
    settings: LLMSettings,
    continuation_index: int,
    request_label: str,
) -> tuple[Any, LLMCallUsage]:
    print(
        f"[streamline-rag] LLM output reached max_tokens; requesting continuation "
        f"{continuation_index}/{settings.max_continuations}.",
        flush=True,
    )
    return _call_completion(
        settings,
        [
            *_initial_messages(prompt),
            {"role": "assistant", "content": partial_content},
            {
                "role": "user",
                "content": (
                    "Continue exactly where the previous response stopped. "
                    "Output only the remaining Python code. Do not repeat earlier code, "
                    "do not add explanations, and close any open code block only if needed."
                ),
            },
        ],
        call_label=f"{request_label} continuation {continuation_index}",
    )


def _complete_with_continuations(
    prompt: str,
    settings: LLMSettings,
    *,
    request_label: str,
) -> tuple[str, Any, int, list[LLMCallUsage]]:
    response, usage = _call_completion(
        settings,
        _initial_messages(prompt),
        call_label=request_label,
    )
    call_usages = [usage]
    content = _message_content(response)
    continuation_count = 0

    while _is_truncated(response) and continuation_count < settings.max_continuations:
        continuation_count += 1
        response, usage = _continue_truncated_response(
            prompt=prompt,
            partial_content=content,
            settings=settings,
            continuation_index=continuation_count,
            request_label=request_label,
        )
        call_usages.append(usage)
        content += "\n" + _message_content(response)

    if _is_truncated(response):
        print(
            f"[streamline-rag] WARNING: LLM output was still truncated after "
            f"{settings.max_continuations} continuation request(s).",
            flush=True,
        )

    return content, response, continuation_count, call_usages


def _repair_prompt(original_prompt: str, code: str, errors: Sequence[str]) -> str:
    return f"""The previous generated VTK script failed validation.

Return a complete corrected Python script, not a patch and not an explanation.

Validation errors:
{chr(10).join(f"- {error}" for error in errors)}

Previous generated code:
```python
{code}
```

Original task prompt:
{original_prompt}
"""


def generate_code(prompt: str, settings: LLMSettings) -> LLMResponse:
    content, raw_response, continuation_count, call_usages = _complete_with_continuations(
        prompt,
        settings,
        request_label="initial generation",
    )
    code = extract_code_block(content)
    validation = validate_generated_code(code)
    repair_attempts = 0

    while not validation.ok and repair_attempts < settings.repair_attempts:
        repair_attempts += 1
        print(
            f"[streamline-rag] Generated code failed validation; requesting repair "
            f"{repair_attempts}/{settings.repair_attempts}: {'; '.join(validation.errors)}",
            flush=True,
        )
        repair_content, raw_response, repair_continuations, repair_usages = _complete_with_continuations(
            _repair_prompt(prompt, code, validation.errors),
            settings,
            request_label=f"validation repair {repair_attempts}",
        )
        call_usages.extend(repair_usages)
        continuation_count += repair_continuations
        content = repair_content
        code = extract_code_block(content)
        validation = validate_generated_code(code)

    if not validation.ok:
        print(
            "[streamline-rag] Generated code still failed validation after repair attempts: "
            + "; ".join(validation.errors),
            flush=True,
        )

    token_usage = _aggregate_token_usage(call_usages)
    _print_token_usage_summary(
        model=normalize_model_name(settings.model, settings.provider),
        continuation_count=continuation_count,
        repair_attempts=repair_attempts,
        call_usages=call_usages,
    )

    return LLMResponse(
        content=content,
        code=code,
        model=normalize_model_name(settings.model, settings.provider),
        finish_reason=_finish_reason(raw_response),
        continuation_count=continuation_count,
        repair_attempts=repair_attempts,
        token_usage=token_usage,
        validation_errors=None if validation.ok else validation.errors,
        raw=raw_response,
    )
