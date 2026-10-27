from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class MessageRecord:
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str  # ISO-like string for simplicity
    personality: str


class ChatDatabase:
    """
    SQLite persistence layer for SmartAIChatbot.

    Stores:
    - users (username, message_count, favorite_mode, last_login)
    - conversations (per-user sessions)
    - chat_history (messages)
    """

    def __init__(self, db_path: str | Path = "smartai.db") -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._log = logging.getLogger(__name__)

    def connect(self) -> None:
        if self._conn is not None:
            return
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            cur = self._conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON;")
            cur.execute("PRAGMA journal_mode = WAL;")
            self._conn.commit()
        except Exception:
            self._log.exception("Failed to connect to SQLite DB: %s", self.db_path)
            raise

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def close(self) -> None:
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None

    def init_schema(self) -> None:
        try:
            c = self.conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    favorite_mode TEXT,
                    last_login TEXT
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    personality TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    setting_key TEXT NOT NULL,
                    setting_value TEXT,
                    updated_at TEXT,
                    UNIQUE(user_id, setting_key),
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    event_name TEXT NOT NULL,
                    event_value TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
                );
                """
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_history_conv_id ON chat_history(conversation_id);"
            )
            c.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id);")
            c.execute("CREATE INDEX IF NOT EXISTS idx_analytics_user_id ON analytics(user_id);")
            self.conn.commit()
        except Exception:
            self._log.exception("Failed to initialise schema.")
            raise

    def upsert_user(self, username: str, *, last_login: str | None = None) -> int:
        """
        Ensure a user exists and return its id.
        """
        username = username.strip()
        if not username:
            username = "anonymous"

        self.conn.execute("INSERT OR IGNORE INTO users(username) VALUES (?);", (username,))
        row = self.conn.execute("SELECT id FROM users WHERE username = ?;", (username,)).fetchone()
        self.conn.commit()
        user_id = int(row["id"]) if row else 0
        if last_login is not None and user_id:
            self.update_last_login(user_id, last_login)
        return user_id

    def update_last_login(self, user_id: int, timestamp: str) -> None:
        self.conn.execute("UPDATE users SET last_login = ? WHERE id = ?;", (timestamp, user_id))
        self.conn.commit()

    def set_favorite_mode(self, user_id: int, favorite_mode: str | None) -> None:
        self.conn.execute(
            "UPDATE users SET favorite_mode = ? WHERE id = ?;",
            ((favorite_mode or None), user_id),
        )
        self.conn.commit()

    def increment_message_count(self, user_id: int, delta: int = 1) -> None:
        self.conn.execute(
            "UPDATE users SET message_count = message_count + ? WHERE id = ?;",
            (int(delta), user_id),
        )
        self.conn.commit()

    def create_conversation(self, user_id: int, created_at: str) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO conversations(user_id, created_at) VALUES (?, ?);",
            (user_id, created_at),
        )
        self.conn.commit()
        return int(c.lastrowid)

    def add_chat_message(self, user_id: int, conversation_id: int, msg: MessageRecord) -> None:
        self.conn.execute(
            """
            INSERT INTO chat_history(user_id, conversation_id, role, content, timestamp, personality)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (user_id, conversation_id, msg.role, msg.content, msg.timestamp, msg.personality),
        )
        self.conn.commit()
        self.increment_message_count(user_id, 1)

    def get_recent_messages(self, conversation_id: int, limit: int = 200) -> list[MessageRecord]:
        """
        Backward-compatible name; reads from chat_history.
        """
        rows = self.conn.execute(
            """
            SELECT role, content, timestamp, personality
            FROM chat_history
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?;
            """,
            (conversation_id, limit),
        ).fetchall()
        # reverse to chronological
        return [
            MessageRecord(
                role=str(r["role"]),
                content=str(r["content"]),
                timestamp=str(r["timestamp"]),
                personality=str(r["personality"]),
            )
            for r in reversed(rows)
        ]

    def iter_conversations(self) -> Iterable[sqlite3.Row]:
        yield from self.conn.execute(
            "SELECT id, user_id, created_at FROM conversations ORDER BY id DESC;"
        ).fetchall()

    def get_user_by_username(self, username: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM users WHERE username = ?;", (username,)).fetchone()

    def upsert_setting(self, user_id: int | None, key: str, value: str | None, updated_at: str) -> None:
        self.conn.execute(
            """
            INSERT INTO settings(user_id, setting_key, setting_value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, setting_key)
            DO UPDATE SET setting_value = excluded.setting_value, updated_at = excluded.updated_at;
            """,
            (user_id, key, value, updated_at),
        )
        self.conn.commit()

    def track_event(self, user_id: int | None, event_name: str, event_value: str | None, created_at: str) -> None:
        self.conn.execute(
            "INSERT INTO analytics(user_id, event_name, event_value, created_at) VALUES (?, ?, ?, ?);",
            (user_id, event_name, event_value, created_at),
        )
        self.conn.commit()

