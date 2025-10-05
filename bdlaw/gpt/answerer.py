"""LLM-powered answer formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import re

from openai import OpenAI

SYSTEM_PROMPT = (
    "Ты — юридический ассистент. Отвечай ТОЛЬКО по предоставленным нормам.\n"
    "Формат обязателен:\n"
    "[{citation}] — краткое пояснение. Источник: {source_url}\n"
    "Если норм недостаточно — честно сообщи и не выдумывай."
)


@dataclass
class AnswerChunk:
    citation: str
    source_url: str
    clean_text: str


@dataclass
class Answer:
    summary: str
    citations: List[str]


class Answerer:
    def __init__(self, model: str):
        self.model = model
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            try:
                self._client = OpenAI()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Не удалось подключиться к OpenAI. Убедитесь, что установлен пакет 'openai' "
                    "и задан API-ключ (OPENAI_API_KEY или запись в keyring)."
                ) from exc
        return self._client

    def answer(self, question: str, chunks: List[AnswerChunk]) -> Answer:
        client = self._get_client()
        unique_chunks: Dict[str, AnswerChunk] = {}
        for chunk in chunks:
            unique_chunks.setdefault(chunk.citation, chunk)
        context_lines = [
            f"[{chunk.citation}]\n{chunk.clean_text}\nИсточник: {chunk.source_url}"
            for chunk in unique_chunks.values()
        ]
        context = "\n\n".join(context_lines)
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
            ],
        )
        message = response.output[0].content[0].text  # type: ignore[index]
        summary = message.strip()
        _validate_answer_format(summary)
        citations = [chunk.citation for chunk in unique_chunks.values()]
        return Answer(summary=summary, citations=citations)


FORMAT_RE = re.compile(r"^\[[^\]]+\]\s+—\s+.+?Источник:\s+\S+", re.MULTILINE)


def _validate_answer_format(answer: str) -> None:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        raise ValueError("LLM ответ пуст")
    for line in lines:
        if not FORMAT_RE.match(line):
            raise ValueError("Ответ не соответствует обязательному формату цитирования")


__all__ = ["Answer", "AnswerChunk", "Answerer"]
