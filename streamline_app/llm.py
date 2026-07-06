from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .code_validation import validate_generated_code


class LLMError(RuntimeError):
    """Raised when code generation through LiteLLM fails."""


DEFAULT_MAX_TOKENS = 50000
MAX_TOKEN_LIMIT = 200000
DEFAULT_MAX_CONTINUATIONS = 2
DEFAULT_REPAIR_ATTEMPTS = 2


@dataclass(frozen=True)
class LLMSettings:
    model: str
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
    validation_errors: list[str] | None = None
    raw: Any = None


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


def _completion_kwargs(settings: LLMSettings, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": list(messages),
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    if settings.api_base:
        kwargs["api_base"] = settings.api_base
    return kwargs


def _call_completion(settings: LLMSettings, messages: Sequence[dict[str, str]]) -> Any:
    try:
        from litellm import completion  # type: ignore
    except ImportError as exc:
        raise LLMError(
            "LiteLLM is required for LLM code generation. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        return completion(**_completion_kwargs(settings, messages))
    except Exception as exc:
        raise LLMError(f"LiteLLM generation failed: {exc}") from exc


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
) -> Any:
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
    )


def _complete_with_continuations(prompt: str, settings: LLMSettings) -> tuple[str, Any, int]:
    response = _call_completion(settings, _initial_messages(prompt))
    content = _message_content(response)
    continuation_count = 0

    while _is_truncated(response) and continuation_count < settings.max_continuations:
        continuation_count += 1
        response = _continue_truncated_response(
            prompt=prompt,
            partial_content=content,
            settings=settings,
            continuation_index=continuation_count,
        )
        content += "\n" + _message_content(response)

    if _is_truncated(response):
        print(
            f"[streamline-rag] WARNING: LLM output was still truncated after "
            f"{settings.max_continuations} continuation request(s).",
            flush=True,
        )

    return content, response, continuation_count


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
    content, raw_response, continuation_count = _complete_with_continuations(prompt, settings)
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
        repair_content, raw_response, repair_continuations = _complete_with_continuations(
            _repair_prompt(prompt, code, validation.errors),
            settings,
        )
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

    return LLMResponse(
        content=content,
        code=code,
        model=settings.model,
        finish_reason=_finish_reason(raw_response),
        continuation_count=continuation_count,
        repair_attempts=repair_attempts,
        validation_errors=None if validation.ok else validation.errors,
        raw=raw_response,
    )
