from __future__ import annotations

from dataclasses import dataclass
import json
import os
from openai import OpenAI


class FormattingAgent:
    def __init__(self, client: OpenAI, model: str = "deepseek-chat", temperature: float = 0.6) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self._template: dict | None = None
        self._load_template()

    def _load_template(self) -> None:
        template_path = os.path.join(os.path.dirname(__file__), "template.json")
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                self._template = json.load(f)
        except Exception:
            self._template = None

    def _build_system_prompt(self) -> str:
        base = (
            "Ты — историк-агент. Твоя задача — отвечать на вопросы о биографии людей. "
            "Ты должен возвращать ответ строго в формате и схеме, заданных во внешнем шаблоне (template.json). "
            "Если вопрос не о конкретном человеке, ответь обычным текстом: 'У меня нет компетенций отвечать на это. Уточните имя человека.'"
        )

        if self._template:
            try:
                template_str = json.dumps(self._template, ensure_ascii=False)
                return base + "\nТребования формата (JSON):\n" + template_str
            except Exception:
                return base
        return base

    def _detect_lang(self, text: str) -> str:
        lower = text.lower()
        if lower.startswith("```"):
            return "fenced"  # already fenced
        if lower.startswith("{") or lower.startswith("["):
            return "json"
        if lower.startswith("<"):
            return "xml"
        lines = [ln for ln in text.splitlines() if ln.strip()]
        colon_lines = sum(1 for ln in lines[:10] if ":" in ln and not ln.strip().startswith("#"))
        return "yaml" if colon_lines >= 2 else "text"

    @dataclass
    class ReplyPayload:
        text: str
        use_markdown: bool

    def reply_payload(self, user_text: str) -> "FormattingAgent.ReplyPayload":
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": (
                    "Ответь на вопрос о биографии человека, используя строго схему из template.json и формат desired_format. "
                    "Если вопрос не о человеке — ответь обычным текстом о нехватке компетенций.\n"
                    f"Вопрос: {user_text}"
                ),
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        content = (response.choices[0].message.content or "").strip()
        lang = self._detect_lang(content)

        if lang == "fenced":
            return self.ReplyPayload(text=content, use_markdown=True)
        if lang == "text":
            return self.ReplyPayload(text=content, use_markdown=False)

        # json/yaml/xml -> fence for stable rendering
        fenced = f"```{lang}\n{content}\n```"
        return self.ReplyPayload(text=fenced, use_markdown=True)

    def reply(self, user_text: str) -> str:
        payload = self.reply_payload(user_text)
        return payload.text


