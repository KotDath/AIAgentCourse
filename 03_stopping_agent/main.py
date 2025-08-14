import asyncio
import os
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.enums import ParseMode, ChatAction
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from agent import FormattingAgent


# Load environment variables
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please set it in your environment or in a .env file."
    )

if not TELEGRAM_API_KEY:
    raise RuntimeError(
        "TELEGRAM_API_KEY is not set. Please set it in your environment or in a .env file."
    )


# Initialize OpenAI client
openai_client = (
    OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    if OPENAI_BASE_URL
    else OpenAI(api_key=OPENAI_API_KEY)
)

router = Router()


# In-memory sessions: chat_id -> list of messages
SESSIONS: dict[int, list[dict[str, str]]] = {}

# Agent instance
AGENT = FormattingAgent(openai_client)


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    SESSIONS[message.chat.id] = []
    await message.answer(
        "Привет! Я — ваш фитнес-тренер и нутрициолог. Опишите кратко цель (снижение жира/набор/поддержание) и предпочтения — задам уточняющие вопросы и соберу меню.",
        parse_mode=None,
    )


@router.message()
async def on_message(message: Message) -> None:
    chat_id = message.chat.id
    user_text = (message.text or "").strip()
    if not user_text:
        await message.answer("Пожалуйста, отправьте текст.", parse_mode=None)
        return

    history = SESSIONS.setdefault(chat_id, [])

    try:
        # Build conversation history with the new user message
        history.append({"role": "user", "content": user_text})

        # Append user's answer
        history.append({"role": "user", "content": user_text})

        stop_event = asyncio.Event()

        async def keep_typing() -> None:
            try:
                while not stop_event.is_set():
                    await message.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                    await asyncio.wait_for(stop_event.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                await keep_typing()

        typing_task = asyncio.create_task(keep_typing())

        # Call agent with full history
        payload = await asyncio.to_thread(AGENT.reply_payload_from_history, history)
        stop_event.set()
        await typing_task

        # Send answer and keep session open
        if payload.use_markdown:
            await message.answer(payload.text)
        else:
            await message.answer(payload.text, parse_mode=None)
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"Произошла ошибка при обращении к LLM: {exc}", parse_mode=None)


async def main() -> None:
    bot = Bot(token=TELEGRAM_API_KEY, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2))
    dp = Dispatcher()
    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass


