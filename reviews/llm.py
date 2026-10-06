"""Classifies review text into complaint themes with a local model served by Ollama.

Every answer is cached on disk, keyed by model, prompt version and review text,
so a rerun costs nothing and an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

from reviews.config import ROOT

CACHE_DIR = ROOT / "data" / "cache"

SYSTEM = """You read customer reviews from a Brazilian online marketplace. Reviews are in Portuguese.
Your job is to tag each review with the complaint themes it mentions.

Themes:
{themes}

Rules:
- A review can have several themes, one theme, or none.
- Only tag a theme the customer actually complains about. Praise is not a complaint.
- "not_delivered" means the customer has not received the order. If it arrived late, use "late_delivery".
- If the review contains no complaint at all, return an empty list.
- Answer with JSON only."""


def _schema(theme_names: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "themes": {"type": "array", "items": {"type": "string", "enum": theme_names}},
        },
        "required": ["themes"],
    }


class Classifier:
    def __init__(self, cfg: dict):
        llm = cfg["llm"]
        self.host = llm["host"].rstrip("/")
        self.model = llm["model"]
        self.options = {"temperature": llm["temperature"], "seed": llm["random_seed"]}
        self.timeout = llm.get("timeout_seconds", 120)
        self.version = llm["prompt_version"]
        self.themes: dict[str, str] = cfg["themes"]
        self.system = SYSTEM.format(
            themes="\n".join(f"- {k}: {v}" for k, v in self.themes.items()))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = CACHE_DIR / f"{self.model.replace(':', '_')}_{self.version}.jsonl"
        self.cache = self._load_cache()

    # cache -----------------------------------------------------------------
    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}|{self.version}|{text}".encode()).hexdigest()

    def _load_cache(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        if self.cache_file.exists():
            for line in self.cache_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    out[row["key"]] = row["themes"]
        return out

    def _save(self, key: str, themes: list[str]) -> None:
        self.cache[key] = themes
        with open(self.cache_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "themes": themes}) + "\n")

    # model -----------------------------------------------------------------
    def check(self) -> None:
        """Fails with a clear message if Ollama is not running or the model is missing."""
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=5).json()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Ollama is not reachable at {self.host}. Start the Ollama app and try again.") from e
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names:
            raise RuntimeError(
                f"Model {self.model} is not installed. Run: ollama pull {self.model}")

    def classify(self, text: str) -> list[str]:
        key = self._key(text)
        if key in self.cache:
            return self.cache[key]
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": f"Review: {text}"},
                ],
                "format": _schema(list(self.themes)),
                "options": self.options,
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        themes = parse(raw, self.themes)
        self._save(key, themes)
        return themes


def parse(raw: str, allowed: dict) -> list[str]:
    """Turns the model's JSON into a clean, sorted, de-duplicated theme list.
    Anything outside the allowed themes is dropped rather than trusted."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    items = data.get("themes", []) if isinstance(data, dict) else []
    return sorted({t for t in items if isinstance(t, str) and t in allowed})
