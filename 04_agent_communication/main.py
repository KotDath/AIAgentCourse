import asyncio
import os
from aiogram import Bot, Dispatcher, Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.enums import ParseMode, ChatAction
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from agent_generator import CommitGeneratorAgent
from agent_validator import CommitValidatorAgent


# Load environment variables
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
TELEGRAM_API_KEY = os.getenv("TELEGRAM_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not TELEGRAM_API_KEY:
    raise RuntimeError("TELEGRAM_API_KEY is not set")


openai_client = (
    OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    if OPENAI_BASE_URL
    else OpenAI(api_key=OPENAI_API_KEY)
)

router = Router()

# Sessions: chat_id -> history
SESSIONS: dict[int, list[dict[str, str]]] = {}

GEN = CommitGeneratorAgent(openai_client)
VAL = CommitValidatorAgent(openai_client)


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    SESSIONS[message.chat.id] = []
    await message.answer(
        "Опиши кратко, какое изменение ты внес. Я помогу оформить commit message, а проверяющий агент валидирует формат.",
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
    history.append({"role": "user", "content": user_text})

    try:
        stop_event = asyncio.Event()

        async def keep_typing() -> None:
            try:
                while not stop_event.is_set():
                    await message.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                    await asyncio.wait_for(stop_event.wait(), timeout=4.0)
            except asyncio.TimeoutError:
                await keep_typing()

        typing_task = asyncio.create_task(keep_typing())

        # Step 1: генератор формирует кандидат
        gen_res = await asyncio.to_thread(GEN.step, history)
        try:
            print(f"[Agent1] {gen_res.content}")
        except Exception:
            pass
        history.append({"role": "assistant", "content": gen_res.content})

        # Кандидат от агента 1
        candidate = gen_res.content

        # Валидация и общение между агентами до статуса OK_AGENT1
        while True:
            val_res = await asyncio.to_thread(VAL.validate, candidate)
            try:
                print(f"[Agent2] {val_res.content}")
            except Exception:
                pass
            if val_res.is_final and val_res.content == "OK_AGENT1":
                await message.answer(f"{candidate}", parse_mode=None)
                SESSIONS[chat_id] = []
                break
            # Передаём подсказки валидатора снова генератору как обычный юзер‑ввод
            history.append({"role": "user", "content": val_res.content})
            gen_fix = await asyncio.to_thread(GEN.step, history)
            try:
                print(f"[Agent1] {gen_fix.content}")
            except Exception:
                pass
            candidate = gen_fix.content

        stop_event.set()
        await typing_task
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"Произошла ошибка: {exc}", parse_mode=None)


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


