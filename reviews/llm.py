"""Classifies review text into complaint themes with a local model served by Ollama.

Every answer is cached on disk, keyed by model, prompt version and review text,
so a rerun costs nothing and an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import requests

from reviews.config import ROOT

CACHE_DIR = ROOT / "data" / "cache"

PROMPTS = {
    # First version: theme list plus a few general rules.
    "v1": """You read customer reviews from a Brazilian online marketplace. Reviews are in Portuguese.
Your job is to tag each review with the complaint themes it mentions.

Themes:
{themes}

Rules:
- A review can have several themes, one theme, or none.
- Only tag a theme the customer actually complains about. Praise is not a complaint.
- "not_delivered" means the customer has not received the order. If it arrived late, use "late_delivery".
- If the review contains no complaint at all, return an empty list.
- Answer with JSON only.""",
    # Second version: adds the rules from docs/LABELLING_GUIDE.md for the cases
    # v1 left open, so the model follows the same definitions as the labels.
    "v2": """You read customer reviews from a Brazilian online marketplace. Reviews are in Portuguese.
Your job is to tag each review with the complaint themes it mentions.

Themes:
{themes}

General rules:
- A review can have several themes, one theme, or none. Return an empty list if there is no complaint.
- Only tag what the customer actually complains about. Praise or a neutral remark is not a complaint.
- A vague negative with no reason ("não recomendo", "muito ruim" about the store) gets no theme. If the product itself is called bad, use poor_quality.
- Wanting a refund or return is not refund_return by itself. Use refund_return only when the customer reports trouble getting one.

Delivery:
- Customer is still waiting and nothing has arrived, even past the deadline: not_delivered.
- Tracking says delivered but the customer has nothing: not_delivered.
- It arrived, but late or slowly: late_delivery.
- Part of the order arrived and some units or items are still missing: missing_items (not not_delivered).
- The order came in several shipments and nothing is missing: other_complaint.
- The customer had to collect the parcel at the post office (correios): other_complaint.

Product:
- A different product, colour, size, voltage or model was sent: wrong_item.
- Broken, scratched, torn, leaking or not working: damaged_defective.
- Differs from the listing, photo or specs, or is fake / "não é original": not_as_described.
- Correct product but flimsy, thin, cheap or badly finished: poor_quality.
- Correct product that the customer simply finds small or does not like: other_complaint.

Service and cost:
- No reply, no contact, no tracking information, support unreachable: seller_support.
- Badly packed, box opened or unsealed, no protection: packaging.
- Product or shipping (frete) too expensive, or shipping charged twice: price_shipping.

Vocabulary: "suporte" usually means a mount or stand, not customer support. "estorno" is a refund.

Answer with JSON only.""",
}


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
        self.system = PROMPTS[self.version].format(
            themes="\n".join(f"- {k}: {v}" for k, v in self.themes.items()))
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache_file = CACHE_DIR / f"{self.model.replace(':', '_')}_{self.version}.jsonl"
        self.cache = self._load_cache()
        self._lock = threading.Lock()

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
        # several worker threads write here, so one at a time
        with self._lock:
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
