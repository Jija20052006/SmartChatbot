from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "SMART AI CHATBOT PRO (Jarvis Style)"
    memory_file: str = "data/smartai_memory.json"
    database_file: str = "data/smartai.db"
    default_personality: str = "default"
    max_visible_history: int = 200


def ensure_project_dirs(base_dir: Path | None = None) -> None:
    root = base_dir or Path(".")
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)

