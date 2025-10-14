"""Shared helpers for OpenAI model retries."""
from __future__ import annotations

import logging
from typing import Iterable, Tuple, Any

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency missing
    OpenAI = None  # type: ignore

LLM_MODEL_CANDIDATES: Tuple[str, ...] = ("gpt-4o-mini", "gpt-4.1")


def call_llm_with_retries(
    *,
    client: Any,
    system_prompt: str,
    user_prompt: str,
    logger: logging.Logger,
    candidates: Iterable[str] | None = None,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    use_responses_first: bool = True,
) -> tuple[str, str]:
    """Call an OpenAI client with retries across preferred models.

    Returns the text content and the model name that succeeded.
    """

    if client is None:
        raise RuntimeError("OpenAI client is not initialised")

    models = []
    for model in candidates or LLM_MODEL_CANDIDATES:
        if model and model not in models:
            models.append(model)

    if not models:
        raise RuntimeError("No candidate models provided for LLM call")

    user_content = [{"type": "text", "text": user_prompt}]
    last_err: Exception | None = None

    for model in models:
        if use_responses_first and hasattr(client, "responses"):
            try:
                kwargs = {
                    "model": model,
                    "instructions": system_prompt,
                    "input": [{"role": "user", "content": user_content}],
                    "temperature": temperature,
                }
                kwargs["max_output_tokens"] = max_output_tokens or 2000
                response = client.responses.create(**kwargs)
                text = getattr(response, "output_text", None)
                if text:
                    logger.info("LLM success via Responses API with model %s", model)
                    return text, model
                try:
                    text = response.output[0].content[0].text  # type: ignore[index]
                except Exception as exc:  # pragma: no cover - SDK quirks
                    logger.warning(
                        "Responses payload parse failed for %s: %r", model, exc
                    )
                else:
                    if text:
                        logger.info(
                            "LLM success via Responses API with model %s", model
                        )
                        return text, model
            except Exception as exc:
                logger.warning("Responses failed for %s: %r", model, exc)
                last_err = exc

        chat_api = getattr(getattr(client, "chat", None), "completions", None)
        if chat_api is None:
            continue
        try:
            chat_kwargs = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if max_output_tokens is not None:
                chat_kwargs["max_tokens"] = max_output_tokens
            response = chat_api.create(**chat_kwargs)
            text = response.choices[0].message.content  # type: ignore[index]
            if not text:
                raise ValueError("Chat completion returned empty content")
            logger.info("LLM success via Chat Completions with model %s", model)
            return text, model
        except Exception as exc:
            logger.warning("Chat failed for %s: %r", model, exc)
            last_err = exc

    if last_err is None:
        last_err = RuntimeError("no models available")
    raise RuntimeError(f"All planner models failed: {last_err}")


__all__ = ["LLM_MODEL_CANDIDATES", "call_llm_with_retries"]
