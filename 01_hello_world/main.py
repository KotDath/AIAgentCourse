import os
import threading
import queue
import tkinter as tk
from tkinter import scrolledtext, messagebox
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in your environment or in a .env file.")

client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


class ChatApplication:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Киберленинка")

        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED, width=80, height=24)
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.user_input = tk.Text(self.root, height=3, width=70)
        self.user_input.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.send_button = tk.Button(self.root, text="Отправить", command=self.on_send)
        self.send_button.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="e")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.user_input.bind("<Return>", self._handle_enter)
        self.user_input.bind("<Shift-Return>", lambda e: None)

        self.conversation_messages = [
            {"role": "system", "content": "Ты являешься профессионалом в теории Марксизма и знаешь наизусть все собрания сочинений Ленина."}
        ]

        self.stream_queue: "queue.Queue[str | tuple]" = queue.Queue()
        self.streaming_in_progress = False

        self.root.after(50, self._poll_stream_queue)

    def _handle_enter(self, event: tk.Event) -> str:
        self.on_send()
        return "break"

    def append_to_chat(self, text: str) -> None:
        self.chat_display.configure(state=tk.NORMAL)
        self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)
        self.chat_display.configure(state=tk.DISABLED)

    def on_send(self) -> None:
        if self.streaming_in_progress:
            return

        user_text = self.user_input.get("1.0", tk.END).strip()
        if not user_text:
            return

        self.user_input.delete("1.0", tk.END)

        self.append_to_chat(f"Вы: {user_text}\n")

        self.conversation_messages.append({"role": "user", "content": user_text})

        self.streaming_in_progress = True
        self.send_button.configure(state=tk.DISABLED)
        threading.Thread(target=self._stream_assistant_reply, daemon=True).start()

    def _stream_assistant_reply(self) -> None:
        try:
            self.stream_queue.put(("prefix", "Ассистент: "))

            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=self.conversation_messages,
                stream=True,
                temperature=0.2,
            )

            accumulated_text_parts: list[str] = []
            for chunk in response:
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta:
                    self.stream_queue.put(("token", delta))
                    accumulated_text_parts.append(delta)

            assistant_text = "".join(accumulated_text_parts)
            self.conversation_messages.append({"role": "assistant", "content": assistant_text})

            self.stream_queue.put(("done", "\n"))
        except Exception as e:
            self.stream_queue.put(("error", str(e)))

    def _poll_stream_queue(self) -> None:
        try:
            while True:
                item = self.stream_queue.get_nowait()
                kind, payload = item if isinstance(item, tuple) else ("token", item)

                if kind == "prefix":
                    self.append_to_chat(str(payload))
                elif kind == "token":
                    self.append_to_chat(str(payload))
                elif kind == "done":
                    self.append_to_chat(str(payload))
                    self.streaming_in_progress = False
                    self.send_button.configure(state=tk.NORMAL)
                elif kind == "error":
                    self.streaming_in_progress = False
                    self.send_button.configure(state=tk.NORMAL)
                    messagebox.showerror("Ошибка", f"Произошла ошибка: {payload}")
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self._poll_stream_queue)


def main() -> None:
    root = tk.Tk()
    app = ChatApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()