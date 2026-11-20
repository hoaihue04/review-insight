from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from app.assistant.config import settings


class LLMError(RuntimeError):
    pass


class ChatLLMClient:
    def __init__(self) -> None:
        provider = settings.llm_provider.strip().lower()
        if provider == "gemini":
            self.client = GeminiChatClient()
        elif provider == "groq":
            self.client = GroqChatClient()
        else:
            raise LLMError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

    @property
    def model(self) -> str:
        return self.client.model

    def is_configured(self) -> bool:
        return self.client.is_configured()

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        return self.client.generate_text(system_prompt, user_prompt)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return self.client.generate_json(system_prompt, user_prompt)


class GroqChatClient:
    def __init__(self) -> None:
        self.model = settings.groq_model
        self.fallback_model = settings.groq_fallback_model

    def is_configured(self) -> bool:
        return bool(settings.groq_api_key)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.is_configured():
            raise LLMError("GROQ_API_KEY is not configured")
        try:
            from groq import Groq
        except Exception as exc:
            raise LLMError("groq package is not installed") from exc

        client = Groq(api_key=settings.groq_api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            completion = client.chat.completions.create(
                model=self.model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_output_tokens,
                messages=messages,
            )
        except Exception as exc:
            if not self.fallback_model or self.fallback_model == self.model:
                raise LLMError(str(exc)) from exc
            completion = client.chat.completions.create(
                model=self.fallback_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_output_tokens,
                messages=messages,
            )
        return _strip_thinking(completion.choices[0].message.content or "")

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        text = self.generate_text(system_prompt, user_prompt)
        try:
            return json.loads(_extract_json_object(text))
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM did not return valid JSON: {text[:200]}") from exc


class GeminiChatClient:
    def __init__(self) -> None:
        self.model = settings.gemini_model

    def is_configured(self) -> bool:
        return bool(settings.gemini_api_key)

    def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if not self.is_configured():
            raise LLMError("GEMINI_API_KEY is not configured")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": settings.llm_max_output_tokens,
            },
        }
        if self.model.startswith("gemini-2.5") and settings.gemini_thinking_budget >= 0:
            payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": settings.gemini_thinking_budget}
        attempt = 0
        while True:
            try:
                response = requests.post(
                    url,
                    headers={
                        "x-goog-api-key": settings.gemini_api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60,
                )
                if response.status_code == 429 and attempt < settings.llm_retry_on_429:
                    wait = _retry_delay_seconds(response)
                    # Chờ quá lâu nghĩa là hết hạn mức trong ngày: thử lại cũng vô ích.
                    if wait is not None and wait <= MAX_RETRY_WAIT_SECONDS:
                        attempt += 1
                        print(f"[LLM] Gemini 429, chờ {wait:.0f}s rồi thử lại ({attempt}/{settings.llm_retry_on_429})")
                        time.sleep(wait)
                        continue
                response.raise_for_status()
                break
            except Exception as exc:
                raise LLMError(f"Gemini request failed: {exc}") from exc

        data = response.json()
        try:
            candidate = data["candidates"][0]
            parts = candidate["content"]["parts"]
            text = "".join(str(part.get("text") or "") for part in parts)
        except Exception as exc:
            raise LLMError(f"Gemini response did not contain text: {str(data)[:200]}") from exc
        if candidate.get("finishReason") == "MAX_TOKENS":
            usage = data.get("usageMetadata") or {}
            raise LLMError(
                "Gemini bị cắt vì hết token (LLM_MAX_OUTPUT_TOKENS="
                f"{settings.llm_max_output_tokens}, suy nghĩ={usage.get('thoughtsTokenCount')}, "
                f"trả lời={usage.get('candidatesTokenCount')})"
            )
        return _strip_thinking(text)

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        text = self.generate_text(system_prompt, user_prompt)
        try:
            return json.loads(_extract_json_object(text))
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM did not return valid JSON: {text[:200]}") from exc


GroqLlamaClient = GroqChatClient

MAX_RETRY_WAIT_SECONDS = 60


def _retry_delay_seconds(response: requests.Response) -> float | None:
    """Thời gian Gemini yêu cầu chờ sau lỗi 429 (header Retry-After hoặc RetryInfo trong body)."""
    header = response.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        for detail in response.json().get("error", {}).get("details", []):
            delay = str(detail.get("retryDelay") or "")
            if delay.endswith("s"):
                return float(delay[:-1]) + 1
    except Exception:
        return None
    return 15.0  # 429 không kèm thời gian chờ: thường là giới hạn theo phút


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    start = text.lower().find("<think>")
    if start >= 0:
        marker = "</think>"
        end = text.lower().find(marker, start)
        if end >= 0:
            text = (text[:start] + text[end + len(marker):]).strip()
        else:
            text = text[:start].strip()
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    return text


def _extract_json_object(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text
