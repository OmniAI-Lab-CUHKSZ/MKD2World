from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: float = 60.0
    retry_attempts: int = 1
    enable_thinking: bool | None = False
    reasoning_effort: str | None = None
    thinking_token_budget: int | None = None
    response_format: Optional[dict] = None


def _openai_content_block(block: dict) -> dict:
    if block.get("type") == "image" and isinstance(block.get("source"), dict):
        source = block["source"]
        if source.get("type") != "base64":
            raise ValueError(f"Unsupported image source type for OpenAI: {source.get('type')}")
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
    return block


def _openai_messages(messages: list[dict]) -> list[dict]:
    converted = []
    for message in messages:
        item = dict(message)
        content = item.get("content")
        if isinstance(content, list):
            item["content"] = [_openai_content_block(block) if isinstance(block, dict) else block for block in content]
        converted.append(item)
    return converted


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text")
    return str(content)


def _sdk_object_to_dict(value: Any) -> dict[str, Any] | None:
    """Normalize response metadata across OpenAI/Anthropic SDK versions."""
    if value is None:
        return None
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            return method()
    if isinstance(value, dict):
        return dict(value)
    return None


def _anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    system_text = ""
    user_messages = []
    for message in messages:
        if message["role"] == "system":
            system_text = _content_text(message.get("content", ""))
        else:
            user_messages.append(dict(message))
    return system_text, user_messages


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._telemetry: dict[str, Any] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "finish_reasons": {},
            "output_limit_hits": 0,
            "configured_max_tokens": int(config.max_tokens),
        }

    def telemetry_snapshot(self) -> dict[str, Any]:
        return {
            **self._telemetry,
            "finish_reasons": dict(self._telemetry["finish_reasons"]),
        }

    def reset_telemetry(self) -> None:
        self._telemetry.update(
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "finish_reasons": {},
                "output_limit_hits": 0,
                "configured_max_tokens": int(self.config.max_tokens),
            }
        )

    def _record_telemetry(self, result: dict[str, Any]) -> None:
        self._telemetry["calls"] += 1
        usage = result.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        self._telemetry["prompt_tokens"] += prompt_tokens
        self._telemetry["completion_tokens"] += completion_tokens
        self._telemetry["total_tokens"] += total_tokens
        finish_reason = str(result.get("finish_reason") or "unknown")
        reasons = self._telemetry["finish_reasons"]
        reasons[finish_reason] = int(reasons.get(finish_reason, 0)) + 1
        if finish_reason in {"length", "max_tokens"} or completion_tokens >= int(self.config.max_tokens):
            self._telemetry["output_limit_hits"] += 1

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.config.provider in {"openai", "openai_compatible"}:
            import openai

            api_key_env = self.config.api_key_env or "OPENAI_API_KEY"
            kwargs = {
                "api_key": self.config.api_key or os.environ.get(api_key_env, "none"),
                "timeout": self.config.timeout_seconds,
            }
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
                parsed = urlparse(self.config.base_url)
                if parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0"}:
                    import httpx

                    kwargs["http_client"] = httpx.Client(trust_env=False)
            self._client = openai.OpenAI(**kwargs)
            return self._client
        if self.config.provider == "anthropic":
            import anthropic

            api_key_env = self.config.api_key_env or "ANTHROPIC_API_KEY"
            api_key = self.config.api_key or os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"{api_key_env} is not set")
            self._client = anthropic.Anthropic(api_key=api_key)
            return self._client
        raise ValueError(f"Unknown LLM provider: {self.config.provider}")

    def chat_with_metadata(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        attempts = max(4, int(self.config.retry_attempts))
        last_error = None
        for _attempt in range(attempts):
            try:
                result = self._chat_with_metadata_once(messages, temperature=temperature, max_tokens=max_tokens)
                self._record_telemetry(result)
                return result
            except Exception as exc:
                last_error = exc
                if _attempt + 1 < attempts:
                    # Retry transient connection failures with backoff.
                    time.sleep(min(30.0, 5.0 * (2 ** _attempt)))
        assert last_error is not None
        raise last_error

    def _chat_with_metadata_once(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        client = self._get_client()
        temp = self.config.temperature if temperature is None else temperature
        tokens = self.config.max_tokens if max_tokens is None else max_tokens

        if self.config.provider in {"openai", "openai_compatible"}:
            request = {
                "model": self.config.model,
                "messages": _openai_messages(messages),
                "temperature": temp,
                "max_tokens": tokens,
            }
            if self.config.response_format is not None:
                request["response_format"] = self.config.response_format
            if self.config.provider == "openai_compatible" and self.config.enable_thinking is not None:
                extra_body: dict[str, Any] = {
                    "chat_template_kwargs": {"enable_thinking": bool(self.config.enable_thinking)}
                }
                if self.config.thinking_token_budget is not None:
                    extra_body["thinking_token_budget"] = int(self.config.thinking_token_budget)
                if self.config.reasoning_effort is not None:
                    # extra_body works with both older and newer OpenAI SDKs.
                    extra_body["reasoning_effort"] = self.config.reasoning_effort
                request["extra_body"] = extra_body
            response = client.chat.completions.create(**request)
            choice = response.choices[0]
            usage = _sdk_object_to_dict(getattr(response, "usage", None))
            return {
                "content": choice.message.content or "",
                "finish_reason": getattr(choice, "finish_reason", None),
                "usage": usage,
            }

        system_text, user_messages = _anthropic_messages(messages)
        response = client.messages.create(
            model=self.config.model,
            system=system_text,
            messages=user_messages,
            temperature=temp,
            max_tokens=tokens,
        )
        usage = _sdk_object_to_dict(getattr(response, "usage", None))
        return {
            "content": response.content[0].text,
            "finish_reason": getattr(response, "stop_reason", None),
            "usage": usage,
        }

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.chat_with_metadata(messages, temperature=temperature, max_tokens=max_tokens)["content"]
