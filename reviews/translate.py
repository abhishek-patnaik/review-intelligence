"""Offline Portuguese to English translation with an Argos Translate model.

The model package is downloaded once into data/cache/argos/ and run with
CTranslate2 and SentencePiece directly, which avoids the PyTorch dependency
the full argostranslate library pulls in.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import requests

from reviews.config import ROOT

CACHE = ROOT / "data" / "cache" / "argos"


def _download(url: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    existing = [p.parent for p in CACHE.rglob("model.bin")]
    if existing:
        return existing[0].parent if existing[0].name == "model" else existing[0]
    print(f"Downloading translation model from {url} (about 100 MB, once only)")
    resp = requests.get(url, timeout=600)
    resp.raise_for_status()
    zipfile.ZipFile(io.BytesIO(resp.content)).extractall(CACHE)
    found = [p.parent for p in CACHE.rglob("model.bin")]
    if not found:
        raise RuntimeError("Translation package downloaded but no model.bin inside it.")
    return found[0].parent if found[0].name == "model" else found[0]


class Translator:
    def __init__(self, cfg: dict):
        import ctranslate2
        import sentencepiece as spm

        root = _download(cfg["translate"]["package_url"])
        model_dir = next(p.parent for p in root.rglob("model.bin"))
        sp_file = next(root.rglob("sentencepiece.model"))
        self.sp = spm.SentencePieceProcessor(model_file=str(sp_file))
        self.model = ctranslate2.Translator(str(model_dir), device="cpu")

    def translate(self, texts: list[str], batch_size: int = 32) -> list[str]:
        """Each review is split into sentences, translated, then joined back."""
        pieces, owner = [], []
        for i, t in enumerate(texts):
            for s in split_sentences(t):
                pieces.append(s)
                owner.append(i)
        out_pieces: list[str] = []
        for start in range(0, len(pieces), batch_size):
            chunk = pieces[start:start + batch_size]
            tokens = [self.sp.encode(s, out_type=str) for s in chunk]
            results = self.model.translate_batch(tokens, beam_size=4, max_decoding_length=256)
            out_pieces += [detokenize(r.hypotheses[0]) for r in results]
        joined = [""] * len(texts)
        for i, s in zip(owner, out_pieces):
            joined[i] = (joined[i] + " " + s).strip()
        return joined


def detokenize(tokens: list[str]) -> str:
    """The model's output pieces mark word starts with U+2581; joining them
    and turning that mark into a space gives plain text."""
    return "".join(tokens).replace("\u2581", " ").strip()


def soften_caps(text: str) -> str:
    """Reviews typed in capitals translate badly (the model has seen few of
    them), so mostly upper case text is turned into sentence case first."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        text = text.lower()
        text = re.sub(r"(^|[.!?]\s+)([a-z\u00e0-\u00ff])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def split_sentences(text: str) -> list[str]:
    text = soften_caps(" ".join(text.split()))
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p] or [text]
