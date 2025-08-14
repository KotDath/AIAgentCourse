from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
from openai import OpenAI


class BaseAgent:
    def __init__(
        self,
        client: OpenAI,
        model: str = "deepseek-chat",
        temperature: float = 0.4,
        *,
        show_request: bool = False,
        show_answers: bool = False,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.show_request = show_request
        self.show_answers = show_answers

    def _print_request(self, header: str, messages: List[Dict[str, str]]) -> None:
        if not self.show_request:
            return
        try:
            system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            dialog = "\n".join(f"{m['role']}: {m['content']}" for m in messages[1:])
            print(f"{header}\nSYSTEM:\n{system}\nDIALOG:\n{dialog}")
        except Exception:
            pass

    def _print_answer(self, header: str, content: str) -> None:
        if not self.show_answers:
            return
        try:
            print(f"{header} {content}")
        except Exception:
            pass

    def chat(self, messages: List[Dict[str, str]]) -> str:
        self._print_request("[LLM PROMPT]", messages)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = (response.choices[0].message.content or "").strip()
        self._print_answer("[LLM ANSWER]", content)
        return content


