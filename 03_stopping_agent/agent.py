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
            "Ты — фитнес-тренер-агент по питанию. Формируй меню/план питания по запросу пользователя. "
            "Ты должен возвращать ответ строго в формате и схеме, заданных во внешнем шаблоне (template.json). "
            "Если запрос не о меню/плане питания, ответь обычным текстом: 'У меня нет компетенций отвечать на это. Уточните запрос по меню/питанию.'"
            "Все текстовые значения на русском (ru). Ключи строго как в menu_schema."
            "Будь краток и фактичен. Пиши пунктирно-короткими фразами."
            "Если информация неизвестна — ставь null для одиночных полей или пустые массивы для списков."
            "Никаких комментариев вокруг финального вывода. Возвращай ТОЛЬКО JSON по схеме."
            "При задавании уточнений — не более 1 вопроса за раз."
            "Режим: если данных недостаточно для корректного заполнения ключей из menu_schema без домыслов — не возвращай JSON в этом сообщении и задай 1–3 целевых вопроса."
            "Возвращай JSON только когда данных достаточно."
            "Обязательно требуй узнать количество приемов пищи и аллегрию. 1 СООБЩЕНИЕ - 1 ВОПРОС. Пример диалога (для референса): '1. Сколько приемов пищи в день?' потом '2. Три в день на 30 дней' потом '2. На что у тебя есть аллергия', узнаёшь что-то дополинтельное и пишешь json с готовым ежедневным рационом по шаблону"
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
            return "fenced"
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

    def reply_payload_from_history(self, conversation_history: list[dict[str, str]]) -> "FormattingAgent.ReplyPayload":
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(conversation_history)

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

        fenced = f"```{lang}\n{content}\n```"
        return self.ReplyPayload(text=fenced, use_markdown=True)

    def reply_payload(self, user_text: str) -> "FormattingAgent.ReplyPayload":
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": (
                    "Сформируй меню/план питания, используя строго схему из template.json и формат desired_format. "
                    "Если запрос не о питании — ответь обычным текстом о нехватке компетенций.\n"
                    f"Запрос: {user_text}"
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

        fenced = f"```{lang}\n{content}\n```"
        return self.ReplyPayload(text=fenced, use_markdown=True)

    def reply(self, user_text: str) -> str:
        payload = self.reply_payload(user_text)
        return payload.text


