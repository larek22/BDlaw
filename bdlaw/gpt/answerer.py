"""LLM-powered answer formatting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

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
        self.client = OpenAI()
        self.model = model

    def answer(self, question: str, chunks: List[AnswerChunk]) -> Answer:
        context_lines = [f"[{chunk.citation}]\n{chunk.clean_text}\nИсточник: {chunk.source_url}" for chunk in chunks]
        context = "\n\n".join(context_lines)
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"},
            ],
        )
        message = response.output[0].content[0].text  # type: ignore[index]
        citations = [chunk.citation for chunk in chunks]
        return Answer(summary=message.strip(), citations=citations)


__all__ = ["Answer", "AnswerChunk", "Answerer"]
