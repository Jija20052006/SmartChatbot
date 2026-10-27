from __future__ import annotations

import argparse
import datetime
import logging

from chatbot import ChatbotEngine, detect_name
from config import AppConfig, ensure_project_dirs
from database import ChatDatabase
from gui_pro import run_gui
from memory import ChatMemory, JsonMemoryStore
from utils import mode_from_command, now_ts, parse_command
from voice import VoiceIO


_PERSONALITY_OPTIONS: list[tuple[str, str, str]] = [
    ("1", "funny", "😂  Funny        – Jokes, sarcasm & bot humour"),
    ("2", "motivator", "💪  Motivator    – Uplifting, energetic & inspiring"),
    ("3", "angry", "😤  Angry        – Short-tempered & dramatic"),
    ("4", "professional", "💼  Professional – Formal, courteous & corporate"),
]


def get_timestamp() -> str:
    return now_ts()


def handle_cli_command(command_text: str, engine: ChatbotEngine, memory: ChatMemory) -> str | None:
    parsed = parse_command(command_text)
    if not parsed:
        return None
    cmd, arg = parsed

    if cmd == "/help":
        return (
            "Commands: /help, /time, /date, /joke, /clear, /history, "
            "/mode funny|motivator|angry|professional"
        )
    if cmd == "/time":
        return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
    if cmd == "/date":
        return f"Today is {datetime.datetime.now().strftime('%A, %d %B %Y')}."
    if cmd == "/joke":
        return engine.reply("tell me a joke")
    if cmd == "/history":
        return f"Loaded messages in current chat: {len(memory.messages)}"
    if cmd == "/clear":
        memory.start_new_conversation()
        return "Chat cleared and a new conversation started."
    if cmd == "/mode":
        mode = mode_from_command(cmd, arg)
        if not mode:
            return "Usage: /mode funny|motivator|angry|professional"
        out = engine.set_personality(mode)
        memory.set_favorite_personality(engine.active_personality)
        return out
    return "Unknown command. Type /help"


def choose_personality(engine: ChatbotEngine, memory: ChatMemory, voice: VoiceIO) -> None:
    banner = (
        "\n"
        "╔══════════════════════════════════════════════╗\n"
        "║        🤖  SmartAI Chatbot  🤖               ║\n"
        "║   Choose your chatbot personality to begin   ║\n"
        "╚══════════════════════════════════════════════╝\n"
    )
    print(banner)

    for num, _, label in _PERSONALITY_OPTIONS:
        print(f"  [{num}] {label}")

    print("\n  You can also type the name directly (e.g. 'funny')")
    fav = (memory.profile.favorite_personality or "").strip().lower() or None
    if fav:
        print(f"  Press Enter to use your saved favorite: '{fav}'.\n")
    else:
        print("  Type 'default' or press Enter for the neutral personality.\n")

    valid: dict[str, str] = {"default": "default"}
    for num, key, _ in _PERSONALITY_OPTIONS:
        valid[num] = key
        valid[key] = key
    valid[""] = fav or "default"

    while True:
        choice = input("  ➤  Your choice: ").strip().lower()
        if choice in valid:
            personality = valid[choice]
            break
        print(f"  ⚠  Invalid choice '{choice}'. Please enter a number (1-4) or a name.\n")

    confirmation = engine.set_personality(personality)
    memory.set_favorite_personality(personality)
    print(f"\n  ✅  {confirmation}")
    print("─" * 50)

    greeting_responses = {
        "funny": "Well, well, well… look who finally decided to talk to a robot! 😄 Type 'quit' to exit.",
        "motivator": "Hello, champion! Today is YOUR day — let's make it amazing! 🏆 Type 'quit' to exit.",
        "angry": "FINE. You're here. Let's just get this over with. 😤 Type 'quit' to exit.",
        "professional": "Good day. I am ready to assist you. Please proceed with your inquiry. Type 'quit' to exit.",
        "default": "Hello! SmartAI is ready. How can I help you today? Type 'quit' to exit.",
    }
    greeting = greeting_responses.get(personality, greeting_responses["default"])
    print(f"\n  🤖  {greeting}\n")
    voice.speak(greeting)


def chat_loop(engine: ChatbotEngine, memory: ChatMemory, voice: VoiceIO) -> None:
    print("─" * 50)
    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  🤖  Goodbye! See you next time.\n")
            voice.speak("Goodbye! See you next time.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            farewell = engine.get_personality_response("bye")
            print(f"  🤖  {farewell}\n")
            voice.speak(farewell)
            break

        if user_input.lower().startswith("switch personality"):
            parts = user_input.lower().split()
            new_p = parts[-1] if len(parts) > 2 else ""
            msg = engine.set_personality(new_p)
            memory.set_favorite_personality(engine.active_personality)
            print(f"  🤖  {msg}\n")
            continue

        cmd_reply = handle_cli_command(user_input, engine=engine, memory=memory)
        if cmd_reply is not None:
            print(f"  🤖  {cmd_reply}\n")
            voice.speak(cmd_reply)
            continue

        # detect if user just told their name and persist it
        name = detect_name(user_input)
        if name:
            memory.set_username(name)
            memory.add("user", user_input, personality=engine.active_personality)
            ack = f"Nice to meet you, {name}. I'll remember that."
            memory.add("assistant", ack, personality=engine.active_personality)
            print(f"  🤖  {ack}\n")
            voice.speak(ack)
            continue

        memory.add("user", user_input, personality=engine.active_personality)
        response = engine.reply(user_input)
        memory.add("assistant", response, personality=engine.active_personality)

        print(f"  🤖  {response}\n")
        voice.speak(response)


def build_app_state(db_path: str) -> tuple[ChatbotEngine, ChatMemory, VoiceIO]:
    engine = ChatbotEngine()
    store = JsonMemoryStore(AppConfig().memory_file)
    # Keep SQLite available (optional), but JSON is the primary source of truth for user+history.
    db = ChatDatabase(db_path=db_path)
    memory = ChatMemory(store=store, db=db)

    # Apply saved favorite personality automatically
    if memory.profile.favorite_personality:
        engine.set_personality(memory.profile.favorite_personality)

    # Ensure we have an active conversation
    if not memory.messages:
        memory.start_new_conversation()
    voice = VoiceIO(enabled=True)
    return engine, memory, voice


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SmartAIChatbot (CLI + GUI)")
    p.add_argument("--gui", action="store_true", help="Launch the Tkinter desktop UI")
    p.add_argument("--db", default=AppConfig().database_file, help="SQLite DB path (default: data/smartai.db)")
    p.add_argument("--no-voice", action="store_true", help="Disable text-to-speech")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    ensure_project_dirs()
    args = parse_args()
    engine, memory, voice = build_app_state(db_path=args.db)
    if args.no_voice:
        voice.enabled = False

    print(f"[{get_timestamp()}] {AppConfig().app_name} started.")

    if args.gui:
        run_gui(engine=engine, memory=memory, voice=voice)
        return

    # Username memory (CLI)
    if not memory.profile.username:
        name = input("Enter your name: ").strip()
        if name:
            memory.set_username(name)
    else:
        welcome = f"Welcome back {memory.profile.username}"
        print(f"\n  🤖  {welcome}\n")
        voice.speak(welcome)
        # update last_login in SQLite
        if memory.db is not None and memory.user_id is not None:
            memory.db.update_last_login(memory.user_id, get_timestamp())

    choose_personality(engine=engine, memory=memory, voice=voice)
    chat_loop(engine=engine, memory=memory, voice=voice)


if __name__ == "__main__":
    main()

