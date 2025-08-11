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


# Initialize OpenAI client (mirrors 01_hello_world behavior)
openai_client = (
    OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    if OPENAI_BASE_URL
    else OpenAI(api_key=OPENAI_API_KEY)
)

router = Router()


# Single agent instance
AGENT = FormattingAgent(openai_client)


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer("Привет, я бот помидорка. Чем могу помочь?", parse_mode=None)

@router.message()
async def on_message(message: Message) -> None:
    user_text = (message.text or "").strip()
    if not user_text:
        await message.answer("Пожалуйста, отправьте текст запроса.", parse_mode=None)
        return

    try:
        # Show typing while processing
        stop_event = asyncio.Event()

        async def keep_typing() -> None:
            try:
                while not stop_event.is_set():
                    await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
                    await asyncio.wait_for(stop_event.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                # loop again to keep typing
                await keep_typing()

        typing_task = asyncio.create_task(keep_typing())

        # Run blocking call in a thread to avoid blocking the event loop
        payload = await asyncio.to_thread(AGENT.reply_payload, user_text)
        stop_event.set()
        await typing_task
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


