from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable
from openai import OpenAI
from base_agent import BaseAgent, UsageInfo


class SummarizerAgent(BaseAgent):
    def __init__(self, client: OpenAI, model: str = "deepseek-chat", temperature: float = 0.2) -> None:
        super().__init__(client, model, temperature)

    @dataclass
    class Summary:
        text: str
        model: str
        total_tokens: int
        prompt_tokens: int
        completion_tokens: int
        max_tokens: int

    def summarize_history(self, history: Iterable[dict[str, str]], *, max_tokens: int = 4096) -> "SummarizerAgent.Summary":
        system = (
            "Ты — ассистент, который кратко конспектирует диалог между тренером и пользователем. "
            "Сделай лаконичное резюме в 3-7 пунктов: цель, ограничения/аллергии, предпочтения, ориентиры по калориям/макросам, бюджет/время/оборудование."
        )
        messages = [{"role": "system", "content": system}] + list(history)
        content, usage = self.chat_completion(messages, max_tokens=max_tokens)
        return self.Summary(
            text=content,
            model=usage.model,
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            max_tokens=usage.max_tokens,
        )

    def humanize_json_menu(self, json_text: str, *, max_tokens: int = 4096) -> "SummarizerAgent.Summary":
        system = (
            "Ты — ассистент, который превращает JSON-меню в краткий человекочитаемый план. "
            "Сводка: цель, длительность, приёмы пищи/день, калорийность/макросы (если есть), основные блюда по дням, список покупок (кратко)."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"JSON-меню:\n```json\n{json_text}\n```"},
        ]
        content, usage = self.chat_completion(messages, max_tokens=max_tokens)
        return self.Summary(
            text=content,
            model=usage.model,
            total_tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            max_tokens=usage.max_tokens,
        )


