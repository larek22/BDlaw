"""GPT-powered structure planner for vectorization plans."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI


class GPTStructurePlanner:
    """Generate document-specific vectorization plans using GPT."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("GPT_MODEL", "gpt-4.1")

    def generate_plan(self, text: str, filename: str) -> Dict[str, Any]:
        sample = text[:8000] if text else ""
        prompt = f"""
You are a senior legal-document structuring expert.
Analyze the text and return the optimal JSON plan for vectorization of Russian legal acts.

Document file name: {filename}

SAMPLE TEXT (may be truncated):
{sample}

REQUIREMENTS:
- Detect heading hierarchy (e.g., ЧАСТЬ/РАЗДЕЛ/ГЛАВА/СТАТЬЯ for codes).
- Provide robust heading regex (support roman numerals and word ordinals, e.g. ПЕРВАЯ).
- Provide preamble rules (exclude “Оглавление”, parliamentary lines, and inline “(в ред. …)” guidance).
- Set chunking policy (by headings, sensible split_on_headings).
- Keep max_tokens ≈ 900 and overlap ≈ 100 (default), unless text density suggests otherwise.
- Include simple quality checks.

Return STRICT JSON in this exact shape (no commentary):
{{
  "doc_type": "rtf|pdf|docx|txt",
  "hierarchy_rules": {{
    "heading_regex": {{
      "part": "..."
      ,"section": "..."
      ,"chapter": "..."
      ,"article": "..."
    }},
    "version_regex": "..."
    ,"path_fields": ["part","section","chapter","article","version_date"]
  }},
  "preamble_rules": {{
    "drop_before_first_heading": null,
    "exclude_regexes": ["...","..."]
  }},
  "chunking_policy": {{
    "mode": "by_headings",
    "split_on_headings": ["chapter","article"],
    "max_tokens": 900,
    "overlap_tokens": 100,
    "keep_lists_intact": true,
    "keep_tables_intact": true,
    "merge_short_paragraphs_under_tokens": 50
  }},
  "quality_checks": {{
    "min_chunks": 1,
    "max_tokens_per_chunk": 900,
    "forbid_empty_text": true,
    "dedupe_near_duplicates": true
  }}
}}
"""

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
            max_output_tokens=2500,
            response_format={"type": "json"},
        )
        try:
            output_text = getattr(response, "output_text", None)
            if not output_text:
                # Fallback to explicit traversal for older SDKs
                parts = []
                for item in getattr(response, "output", []) or []:
                    for content in getattr(item, "content", []) or []:
                        value = getattr(content, "text", None)
                        if value:
                            parts.append(value)
                output_text = "".join(parts)
            if not output_text:
                raise ValueError("empty response from GPT")
            return json.loads(output_text)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"GPT returned invalid JSON plan: {exc}") from exc
