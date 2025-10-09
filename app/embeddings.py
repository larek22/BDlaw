from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from openai import OpenAI

from .settings import AppSettings, CONFIG_DIR

logger = logging.getLogger(__name__)

_CACHE_PATH = CONFIG_DIR / "embeddings_cache.sqlite"


class EmbeddingCache:
    def __init__(self, path: Path = _CACHE_PATH) -> None:
        self.path = path
        self._ensure_table()

    def _ensure_table(self) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    sha TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    PRIMARY KEY (sha, model)
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, sha: str, model: str) -> List[float] | None:
        conn = sqlite3.connect(self.path)
        try:
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE sha = ? AND model = ?",
                (sha, model),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
        finally:
            conn.close()

    def set(self, sha: str, model: str, embedding: Sequence[float]) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                "REPLACE INTO embeddings (sha, model, embedding) VALUES (?, ?, ?)",
                (sha, model, json.dumps(list(embedding))),
            )
            conn.commit()
        finally:
            conn.close()


@dataclass
class EmbeddingResult:
    sha: str
    vector: List[float]


class EmbeddingClient:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not set")
        self._client = OpenAI(api_key=api_key)
        self._cache = EmbeddingCache()

    def embed_texts(
        self,
        texts: Sequence[str],
        shas: Sequence[str],
        batch_size: int = 64,
        max_retries: int = 5,
    ) -> List[EmbeddingResult]:
        if len(texts) != len(shas):
            raise ValueError("texts and shas must have the same length")

        model = self.settings.openai_models.embedding
        vectors: List[EmbeddingResult | None] = [None] * len(texts)
        pending_indices: List[int] = []
        unique_pending: dict[str, List[int]] = {}
        unique_payload: dict[str, str] = {}

        for idx, (text, sha) in enumerate(zip(texts, shas)):
            cached = self._cache.get(sha, model)
            if cached is not None:
                vectors[idx] = EmbeddingResult(sha=sha, vector=cached)
                continue
            pending_indices.append(idx)
            if sha not in unique_pending:
                unique_pending[sha] = []
                unique_payload[sha] = text
            unique_pending[sha].append(idx)

        if unique_pending:
            pending_shas = list(unique_pending.keys())
            for start in range(0, len(pending_shas), batch_size):
                batch_shas = pending_shas[start : start + batch_size]
                payload = [unique_payload[sha] for sha in batch_shas]
                retry = 0
                while True:
                    try:
                        response = self._client.embeddings.create(
                            model=model,
                            input=payload,
                        )
                        break
                    except Exception as exc:  # pragma: no cover - network errors
                        retry += 1
                        if retry > max_retries:
                            raise
                        sleep_for = min(2 ** retry, 10)
                        logger.warning(
                            "Embedding request failed (attempt %s/%s): %s",
                            retry,
                            max_retries,
                            exc,
                        )
                        time.sleep(sleep_for)
                data = response.data
                for sha, item in zip(batch_shas, data):
                    vector = item.embedding
                    self._cache.set(sha, model, vector)
                    for idx in unique_pending[sha]:
                        vectors[idx] = EmbeddingResult(sha=sha, vector=vector)

        return [v for v in vectors if v is not None]

    def embed_query(self, text: str) -> List[float]:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._cache.get(sha, self.settings.openai_models.embedding)
        if cached is not None:
            return cached
        return self.embed_texts([text], [sha])[0].vector
