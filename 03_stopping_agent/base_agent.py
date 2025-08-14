from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Dict, Tuple
from openai import OpenAI


@dataclass
class UsageInfo:
    model: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    max_tokens: int


class BaseAgent:
    def __init__(self, client: OpenAI, model: str = "deepseek-chat", temperature: float = 0.6) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def chat_completion(self, messages: List[Dict[str, str]], *, max_tokens: int) -> Tuple[str, UsageInfo]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        content = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        model_used = getattr(response, "model", self.model)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)
        return content, UsageInfo(
            model=model_used,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            max_tokens=max_tokens,
        )


