from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
from base_agent import BaseAgent


@dataclass
class AgentResult:
    content: str
    is_final: bool = False


class CommitGeneratorAgent(BaseAgent):
    def step(self, history: List[Dict[str, str]]) -> AgentResult:
        system = (
            "Ты — помощник, который пишет сообщения коммитов. Формат строгий и один‑строчный: "
            "Политика: 1 сообщение — 1 уточняющий вопрос. Когда данных достаточно — верни ТОЛЬКО commit message без пояснений."
        )
        messages = [{"role": "system", "content": system}] + history
        content = self.chat(messages)
        return AgentResult(content=content, is_final=False)


