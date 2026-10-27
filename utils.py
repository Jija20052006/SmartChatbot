from __future__ import annotations

import datetime
from typing import Optional, Tuple


def now_ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_command(text: str) -> tuple[str, str] | None:
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return None
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return (cmd, arg)


def mode_from_command(cmd: str, arg: str) -> Optional[str]:
    if cmd != "/mode":
        return None
    key = (arg or "").strip().lower()
    if key in {"funny", "motivator", "angry", "professional", "default"}:
        return key
    return None

