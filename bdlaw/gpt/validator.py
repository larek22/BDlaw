"""LLM-as-judge validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from openai import OpenAI

VALIDATOR_PROMPT = (
    "Проверь ответ на соответствие retrieved нормам."
    " Верни PASS, если каждая цитата соответствует одной из норм, иначе FAIL."
)


@dataclass
class ValidationResult:
    verdict: str
    explanation: str


class AnswerValidator:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    def validate(self, answer: str, citations: List[str], context_citations: List[str]) -> ValidationResult:
        input_payload = [
            {"role": "system", "content": VALIDATOR_PROMPT},
            {
                "role": "user",
                "content": (
                    "Цитаты ответа: "
                    + "; ".join(citations)
                    + "\nДоступные нормы: "
                    + "; ".join(context_citations)
                    + "\nОтвет:\n"
                    + answer
                ),
            },
        ]
        response = self.client.responses.create(model=self.model, input=input_payload)
        content = response.output[0].content[0].text  # type: ignore[index]
        verdict = "PASS" if "PASS" in content.upper() else "FAIL"
        return ValidationResult(verdict=verdict, explanation=content.strip())


__all__ = ["AnswerValidator", "ValidationResult"]
