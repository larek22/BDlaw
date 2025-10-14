"""Tokenization helpers used by the vectorization pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:  # pragma: no cover - optional dependency
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore


@dataclass(slots=True)
class Tokenizer:
    name: str = "cl100k_base"

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        if tiktoken is None:
            return [ord(ch) for ch in text]
        try:
            encoding = tiktoken.get_encoding(self.name)
        except Exception:
            return [ord(ch) for ch in text]
        return encoding.encode(text, disallowed_special=())

    def decode(self, tokens: List[int]) -> str:
        if not tokens:
            return ""
        if tiktoken is None:
            return "".join(chr(token) for token in tokens)
        try:
            encoding = tiktoken.get_encoding(self.name)
        except Exception:
            return "".join(chr(token) for token in tokens)
        return encoding.decode(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


DEFAULT_TOKENIZER = Tokenizer()

__all__ = ["Tokenizer", "DEFAULT_TOKENIZER"]
