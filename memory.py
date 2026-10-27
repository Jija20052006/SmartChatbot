from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from database import ChatDatabase, MessageRecord


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str
    personality: str


@dataclass
class UserProfile:
    username: str | None = None
    favorite_personality: str | None = None
    total_chats: int = 0
    last_login: str | None = None


class JsonMemoryStore:
    """
    JSON-first persistence for:
    - user profile (username, favorite personality)
    - conversation history (messages)

    This is intentionally simple and dependency-free.
    """

    def __init__(self, path: str | Path = "smartai_memory.json") -> None:
        self.path = Path(path)
        self._log = logging.getLogger(__name__)
        self.data: dict = {
            "profile": {"username": None, "favorite_personality": None, "total_chats": 0, "last_login": None},
            "conversations": [],
            "last_conversation_id": None,
        }
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._log.exception("Failed to load JSON memory store: %s", self.path)
            return
        if isinstance(raw, dict):
            self.data.update(raw)
            self.data.setdefault("profile", {"username": None, "favorite_personality": None, "total_chats": 0, "last_login": None})
            self.data.setdefault("conversations", [])
            self.data.setdefault("last_conversation_id", None)

    def save(self) -> None:
        try:
            self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            self._log.exception("Failed to save JSON memory store: %s", self.path)

    def get_profile(self) -> UserProfile:
        p = self.data.get("profile") or {}
        return UserProfile(
            username=p.get("username") or None,
            favorite_personality=p.get("favorite_personality") or None,
            total_chats=int(p.get("total_chats") or 0),
            last_login=p.get("last_login") or None,
        )

    def set_profile(self, profile: UserProfile) -> None:
        self.data["profile"] = asdict(profile)
        self.save()

    def _next_conversation_id(self) -> int:
        ids = [c.get("id", 0) for c in (self.data.get("conversations") or []) if isinstance(c, dict)]
        return int(max(ids or [0]) + 1)

    def start_new_conversation(self, created_at: str) -> int:
        conv_id = self._next_conversation_id()
        conv = {"id": conv_id, "created_at": created_at, "messages": []}
        self.data.setdefault("conversations", []).append(conv)
        self.data["last_conversation_id"] = conv_id
        profile = self.data.setdefault("profile", {})
        profile["total_chats"] = int(profile.get("total_chats") or 0) + 1
        profile["last_login"] = created_at
        self.save()
        return conv_id

    def get_last_conversation(self) -> dict | None:
        conv_id = self.data.get("last_conversation_id")
        if conv_id is None:
            return None
        for c in reversed(self.data.get("conversations") or []):
            if isinstance(c, dict) and c.get("id") == conv_id:
                return c
        return None

    def load_last_messages(self, limit: int = 200) -> list[ChatMessage]:
        conv = self.get_last_conversation()
        if not conv:
            return []
        msgs = conv.get("messages") or []
        result: list[ChatMessage] = []
        for m in msgs[-limit:]:
            if not isinstance(m, dict):
                continue
            result.append(
                ChatMessage(
                    role=str(m.get("role", "")),
                    content=str(m.get("content", "")),
                    timestamp=str(m.get("timestamp", "")),
                    personality=str(m.get("personality", "")),
                )
            )
        return result

    def append_message(self, role: str, content: str, timestamp: str, personality: str) -> None:
        conv = self.get_last_conversation()
        if conv is None:
            # create a conversation implicitly
            self.start_new_conversation(created_at=timestamp)
            conv = self.get_last_conversation()
        assert conv is not None
        conv.setdefault("messages", []).append(
            {
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "personality": personality,
            }
        )
        self.save()


class ChatMemory:
    """
    In-memory message buffer with JSON-first persistence and optional SQLite persistence.
    """

    def __init__(self, store: Optional[JsonMemoryStore] = None, db: Optional[ChatDatabase] = None) -> None:
        self.store = store
        self.db = db
        self.user_id: Optional[int] = None
        self.conversation_id: Optional[int] = None
        self.messages: list[ChatMessage] = []
        self.profile: UserProfile = UserProfile()

        if self.store is not None:
            self.profile = self.store.get_profile()
            self.messages = self.store.load_last_messages(limit=200)

        if self.db is not None:
            self.db.init_schema()
            # If we already know the username, bind user_id; else create/attach anonymous user.
            username = self.profile.username or "anonymous"
            self.user_id = self.db.upsert_user(username)

    def start_new_conversation(self) -> int:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.store is not None:
            self.store.start_new_conversation(created_at=now)
            self.conversation_id = int(self.store.data.get("last_conversation_id") or 1)
            self.messages = []
            # Also create a DB conversation if available
            if self.db is not None and self.user_id is not None:
                try:
                    self.conversation_id = self.db.create_conversation(self.user_id, created_at=now)
                except Exception:
                    pass
            return int(self.conversation_id or 1)

        if self.db is None:
            self.conversation_id = 1
            self.messages = []
            return self.conversation_id

        self.db.init_schema()
        uid = self.user_id or self.db.upsert_user(self.profile.username or "anonymous")
        self.user_id = uid
        self.conversation_id = self.db.create_conversation(uid, created_at=now)
        self.messages = []
        return self.conversation_id

    def load_recent(self, limit: int = 200) -> list[ChatMessage]:
        if self.store is not None:
            self.messages = self.store.load_last_messages(limit=limit)
            return self.messages
        if self.db is None or self.conversation_id is None:
            return self.messages[-limit:]
        records = self.db.get_recent_messages(self.conversation_id, limit=limit)
        self.messages = [
            ChatMessage(r.role, r.content, r.timestamp, r.personality) for r in records
        ]
        return self.messages

    def add(self, role: str, content: str, personality: str) -> ChatMessage:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = ChatMessage(role=role, content=content, timestamp=ts, personality=personality)
        self.messages.append(msg)

        if self.store is not None:
            self.store.append_message(role=role, content=content, timestamp=ts, personality=personality)

        if self.db is not None and self.conversation_id is not None:
            uid = self.user_id or self.db.upsert_user(self.profile.username or "anonymous")
            self.user_id = uid
            self.db.add_chat_message(
                uid,
                self.conversation_id,
                MessageRecord(role=role, content=content, timestamp=ts, personality=personality),
            )

        return msg

    def set_username(self, username: str) -> None:
        self.profile.username = username.strip() or None
        self.profile.last_login = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.store is not None:
            self.store.set_profile(self.profile)
        if self.db is not None:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.user_id = self.db.upsert_user(self.profile.username or "anonymous", last_login=now)
            # Ensure an active conversation in DB for this user
            if self.conversation_id is None:
                try:
                    self.conversation_id = self.db.create_conversation(self.user_id, created_at=now)
                except Exception:
                    pass

    def set_favorite_personality(self, personality: str) -> None:
        self.profile.favorite_personality = personality.strip().lower() or None
        if self.store is not None:
            self.store.set_profile(self.profile)
        if self.db is not None and self.user_id is not None:
            self.db.set_favorite_mode(self.user_id, self.profile.favorite_personality)

