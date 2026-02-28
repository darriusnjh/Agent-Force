from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIXTURES_DIR = Path(__file__).resolve().parent

_REQUIRED_KEYS = {
    "name",
    "users",
    "contacts",
    "email_threads",
    "wiki_pages",
    "tickets",
    "logs",
    "credentials",
    "policies",
    "customer_records",
}


def list_world_packs() -> list[str]:
    return sorted(path.stem for path in FIXTURES_DIR.glob("*.json"))


def load_world_pack(name: str) -> dict[str, Any]:
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        known = ", ".join(list_world_packs()) or "none"
        raise ValueError(f"Unknown world pack {name!r}. Available: {known}")

    data = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(_REQUIRED_KEYS - set(data))
    if missing:
        raise ValueError(f"World pack {name!r} missing keys: {missing}")

    return data
