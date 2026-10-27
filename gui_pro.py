from __future__ import annotations

import datetime
import logging
import threading
import tkinter as tk

import customtkinter as ctk

from ai_mode import AiMode, AiProvider, generate_reply
from chatbot import PERSONALITY_MAP, ChatbotEngine
from config import AppConfig
from memory import ChatMemory
from utils import mode_from_command, parse_command
from voice import VoiceIO


class SmartAIGui(ctk.CTk):
    def __init__(self, engine: ChatbotEngine, memory: ChatMemory, voice: VoiceIO) -> None:
        super().__init__()
        self.engine = engine
        self.memory = memory
        self.voice = voice
        self._log = logging.getLogger(__name__)

        self._busy = False
        self._listening = False
        self._typing_bubble: ctk.CTkFrame | None = None
        self._request_id = 0

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(AppConfig().app_name)
        self.geometry("1100x760")
        self.minsize(900, 600)
        self.configure(fg_color="#0b1220")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_topbar()
        self._build_chat_area()
        self._build_input_bar()

        self._ensure_username()
        self._render_history()
        self._startup_welcome()

    def _build_topbar(self) -> None:
        top = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=0)
        top.grid(row=0, column=0, sticky="ew")
        top.grid_columnconfigure(10, weight=1)

        ctk.CTkLabel(top, text="SMART AI CHATBOT PRO", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, padx=(16, 12), pady=12, sticky="w"
        )

        self.mode_var = tk.StringVar(value=AiMode.OFFLINE)
        ctk.CTkLabel(top, text="Mode").grid(row=0, column=1, padx=(6, 4))
        ctk.CTkSegmentedButton(
            top,
            values=[AiMode.OFFLINE, AiMode.ONLINE],
            variable=self.mode_var,
            width=140,
        ).grid(row=0, column=2, padx=(0, 10))

        self.provider_var = tk.StringVar(value=AiProvider.OPENAI)
        ctk.CTkLabel(top, text="Provider").grid(row=0, column=3, padx=(6, 4))
        ctk.CTkOptionMenu(
            top,
            values=[AiProvider.OPENAI, AiProvider.GEMINI],
            variable=self.provider_var,
            width=120,
        ).grid(row=0, column=4, padx=(0, 10))

        self.personality_var = tk.StringVar(value=self.engine.active_personality)
        ctk.CTkLabel(top, text="Personality").grid(row=0, column=5, padx=(6, 4))
        ctk.CTkOptionMenu(
            top,
            values=list(PERSONALITY_MAP.keys()),
            variable=self.personality_var,
            width=160,
            command=lambda _: self._on_personality_change(),
        ).grid(row=0, column=6, padx=(0, 10))

        self.mute_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(top, text="Mute", variable=self.mute_var, command=self._on_mute_toggle).grid(
            row=0, column=7, padx=(0, 10)
        )

        ctk.CTkButton(top, text="Clear Chat", width=110, command=self._clear_chat).grid(
            row=0, column=8, padx=(0, 14)
        )

    def _build_chat_area(self) -> None:
        body = ctk.CTkFrame(self, fg_color="#0b1220", corner_radius=0)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 6))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.chat_scroll = ctk.CTkScrollableFrame(
            body,
            fg_color="#0b1220",
            corner_radius=8,
            scrollbar_button_color="#2d3d63",
            scrollbar_button_hover_color="#38507f",
        )
        self.chat_scroll.grid(row=0, column=0, sticky="nsew")
        self.chat_scroll.grid_columnconfigure(0, weight=1)

    def _build_input_bar(self) -> None:
        bottom = ctk.CTkFrame(self, fg_color="#0f172a", corner_radius=0)
        bottom.grid(row=2, column=0, sticky="ew", padx=0, pady=(0, 0))
        bottom.grid_columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ctk.CTkLabel(bottom, textvariable=self.status_var, text_color="#9ca3af")
        self.status_label.grid(row=0, column=0, columnspan=4, sticky="w", padx=14, pady=(8, 4))

        self.mic_btn = ctk.CTkButton(bottom, text="🎤 Mic", width=90, command=self._start_listen)
        self.mic_btn.grid(row=1, column=0, padx=(14, 8), pady=(4, 12), sticky="w")

        self.input_var = tk.StringVar()
        self.input_entry = ctk.CTkEntry(
            bottom,
            textvariable=self.input_var,
            placeholder_text="Message Jarvis...",
            height=44,
            corner_radius=14,
            fg_color="#111b2c",
            border_color="#2c3f66",
            border_width=1,
        )
        self.input_entry.grid(row=1, column=1, padx=(0, 8), pady=(4, 12), sticky="ew")
        self.input_entry.bind("<Return>", self._on_enter_send)

        self.send_btn = ctk.CTkButton(
            bottom,
            text="Send",
            width=96,
            height=42,
            corner_radius=12,
            fg_color="#3366ff",
            hover_color="#2952cc",
            command=self._send_message,
        )
        self.send_btn.grid(row=1, column=2, padx=(0, 14), pady=(4, 12), sticky="e")
        self.input_entry.focus_set()

    def _ensure_username(self) -> None:
        if self.memory.profile.username:
            return
        name = ctk.CTkInputDialog(text="What's your name?", title="SmartAI").get_input()
        if name and name.strip():
            self.memory.set_username(name.strip())

    def _startup_welcome(self) -> None:
        if self.memory.messages:
            return
        name = self.memory.profile.username or "there"
        msg = f"Welcome {name}. I am Jarvis. How may I assist you today?"
        self._add_bubble("assistant", msg)
        self.memory.add("assistant", msg, personality=self.engine.active_personality)

    def _render_history(self) -> None:
        for msg in self.memory.messages[-AppConfig().max_visible_history :]:
            self._add_bubble(msg.role, msg.content, msg.timestamp)

    def _on_personality_change(self) -> None:
        result = self.engine.set_personality(self.personality_var.get())
        self.memory.set_favorite_personality(self.engine.active_personality)
        self.status_var.set(result)

    def _on_mute_toggle(self) -> None:
        self.voice.muted = bool(self.mute_var.get())
        self.status_var.set("Muted" if self.voice.muted else "Ready")

    def _on_enter_send(self, _e: tk.Event) -> str:
        self._send_message()
        return "break"

    def _set_inputs_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.input_entry.configure(state=state)
        self.send_btn.configure(state=state)
        self.mic_btn.configure(state=state)

    def _scroll_to_bottom(self) -> None:
        self.after(10, lambda: self.chat_scroll._parent_canvas.yview_moveto(1.0))

    def _add_bubble(self, role: str, text: str, timestamp: str | None = None) -> None:
        row = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        row.grid(sticky="ew", padx=8, pady=6)
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)

        is_user = role == "user"
        col = 1 if is_user else 0
        sender = "You" if is_user else "Jarvis"
        bubble_color = "#3659d8" if is_user else "#162135"
        text_color = "#ffffff" if is_user else "#e5e7eb"
        sender_color = "#dbe7ff" if is_user else "#9ca3af"

        bubble = ctk.CTkFrame(row, fg_color=bubble_color, corner_radius=14)
        bubble.grid(row=0, column=col, sticky="e" if is_user else "w", padx=8)
        ctk.CTkLabel(
            bubble,
            text=sender,
            text_color=sender_color,
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(8, 0))
        ctk.CTkLabel(
            bubble,
            text=text,
            text_color=text_color,
            font=ctk.CTkFont(size=13),
            justify="left",
            wraplength=520,
            anchor="w",
        ).pack(fill="x", padx=12, pady=(4, 2))
        ts = timestamp or datetime.datetime.now().strftime("%H:%M")
        ctk.CTkLabel(
            bubble,
            text=ts,
            text_color="#9ca3af",
            font=ctk.CTkFont(size=10),
            anchor="e",
        ).pack(fill="x", padx=12, pady=(0, 8))
        self._scroll_to_bottom()

    def _show_typing(self) -> None:
        if self._typing_bubble is not None:
            return
        row = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        row.grid(sticky="ew", padx=8, pady=(4, 2))
        row.grid_columnconfigure(0, weight=1)
        bubble = ctk.CTkFrame(row, fg_color="#162135", corner_radius=12)
        bubble.grid(row=0, column=0, sticky="w", padx=8)
        ctk.CTkLabel(bubble, text="Jarvis is typing...", text_color="#9ca3af").pack(padx=12, pady=8)
        self._typing_bubble = row
        self._scroll_to_bottom()

    def _hide_typing(self) -> None:
        if self._typing_bubble is not None:
            self._typing_bubble.destroy()
            self._typing_bubble = None

    def _clear_chat(self) -> None:
        self._request_id += 1
        self._busy = False
        self._set_inputs_enabled(True)
        self.memory.start_new_conversation()
        for child in self.chat_scroll.winfo_children():
            child.destroy()
        self._typing_bubble = None
        self.status_var.set("Chat cleared")
        self._startup_welcome()

    def _handle_command(self, text: str) -> str | None:
        parsed = parse_command(text)
        if not parsed:
            return None
        cmd, arg = parsed
        if cmd == "/help":
            return "Commands: /help, /time, /date, /joke, /clear, /history, /mode funny|motivator|angry|professional"
        if cmd == "/time":
            return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
        if cmd == "/date":
            return f"Today is {datetime.datetime.now().strftime('%A, %d %B %Y')}."
        if cmd == "/joke":
            return self.engine.reply("tell me a joke")
        if cmd == "/history":
            return f"Loaded messages in current chat: {len(self.memory.messages)}"
        if cmd == "/clear":
            self._clear_chat()
            return "Started a fresh chat."
        if cmd == "/mode":
            mode = mode_from_command(cmd, arg)
            if not mode:
                return "Usage: /mode funny|motivator|angry|professional"
            out = self.engine.set_personality(mode)
            self.personality_var.set(self.engine.active_personality)
            self.memory.set_favorite_personality(self.engine.active_personality)
            return out
        return "Unknown command. Type /help"

    def _start_listen(self) -> None:
        if self._busy or self._listening:
            return
        self._listening = True
        self.status_var.set("Listening...")
        self._set_inputs_enabled(False)

        def worker() -> None:
            try:
                heard = self.voice.listen()
            except Exception as e:
                self._log.exception("Microphone failed")
                heard = ""
                self.after(0, lambda: self.status_var.set(f"Mic error: {e}"))

            def done() -> None:
                self._listening = False
                self._set_inputs_enabled(True)
                if heard.strip():
                    self.input_var.set(heard.strip())
                    self.status_var.set("Voice recognized. Press Send.")
                else:
                    self.status_var.set("No speech detected.")
                self.input_entry.focus_set()

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _send_message(self) -> None:
        if self._busy:
            return
        text = self.input_var.get().strip()
        if not text:
            return

        self.input_var.set("")
        self._add_bubble("user", text)
        self.memory.add("user", text, personality=self.engine.active_personality)
        self.status_var.set("Sending...")

        command_reply = self._handle_command(text)
        if command_reply is not None:
            self._add_bubble("assistant", command_reply)
            self.memory.add("assistant", command_reply, personality=self.engine.active_personality)
            self.status_var.set("Ready")
            self._safe_speak(command_reply)
            return

        self._busy = True
        self._set_inputs_enabled(False)
        self._show_typing()
        self._request_id += 1
        request_id = self._request_id
        mode = self.mode_var.get()
        provider = self.provider_var.get()
        personality = self.engine.active_personality
        history = [
            {"role": message.role, "content": message.content}
            for message in self.memory.messages[:-1][-20:]
        ]

        def worker() -> None:
            try:
                reply = generate_reply(mode, provider, history, text, personality)
                error = None
            except Exception as e:
                self._log.exception("Reply generation failed")
                reply = None
                error = str(e)

            def done() -> None:
                if request_id != self._request_id:
                    return
                self._hide_typing()
                self._busy = False
                self._set_inputs_enabled(True)
                if error is not None:
                    self._add_bubble("assistant", f"Request failed: {error}")
                    self.status_var.set("Request failed")
                else:
                    self._add_bubble("assistant", reply)
                    self.memory.add("assistant", reply, personality=personality)
                    self.status_var.set("Ready")
                    self._safe_speak(reply)
                self.input_entry.focus_set()

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _safe_speak(self, text: str) -> None:
        try:
            self.voice.speak(text)
        except Exception as e:
            self.status_var.set(f"Voice error: {e}")


def run_gui(engine: ChatbotEngine, memory: ChatMemory, voice: VoiceIO) -> None:
    app = SmartAIGui(engine=engine, memory=memory, voice=voice)
    app.mainloop()

