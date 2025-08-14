from __future__ import annotations

from dataclasses import dataclass
from base_agent import BaseAgent


@dataclass
class AgentResult:
    content: str
    is_final: bool = False


class CommitValidatorAgent(BaseAgent):
    def validate(self, commit_message: str) -> AgentResult:
        system = (
            "Ты — строгий валидатор сообщений коммитов. Валидируй ТОЧНО по правилам:\n"
            "1) Сообщение ДОЛЖНО начинаться с одного из тегов: [Project] | [Bugfix] | [Structure] и пробела.\n"
            "2) После тега первое слово ДОЛЖНО начинаться с заглавной буквы.\n"
            "3) Сообщение ДОЛЖНО быть однострочным (без переводов строк).\n"
            "4) Сообщение ДОЛЖНО оканчиваться точкой.\n"
            "5) Сообщение ДОЛЖНО быть только на английском (ASCII буквы/цифры/знаки препинания).\n"
            "Семантика: [Project] = новая фича, [Bugfix] = исправление бага, [Structure] = рефакторинг. Если семантика явно противоречит тегу по формулировке субъекта — считай это нарушением; иначе пропусти семантическую проверку.\n\n"
            "ФОРМАТ ВЫВОДА (строго один из):\n"
            "- OK_AGENT1\n"
            "- Issues:\n- <короткое нарушение 1>\n- <короткое нарушение 2>\n\nProposed:\n<однострочное исправленное сообщение>\n"
            "Не добавляй никакие дополнительные комментарии, кавычки или код‑блоки."
        )
        user = f"Проверь это сообщение коммита:\n{commit_message.strip()}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        content = self.chat(messages)
        return AgentResult(content=content, is_final=(content.strip() == "OK_AGENT1"))


