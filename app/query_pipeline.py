from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from openai import OpenAI, OpenAIError, PermissionDeniedError
from qdrant_client.http import models as rest

from .embeddings import EmbeddingClient
from .legal_types import ChunkRecord
from .qdrant_client import QdrantVectorStore
from .settings import AppSettings
from .rerankers import RerankCandidate, build_reranker

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency
    import pymorphy2  # type: ignore
except Exception:  # pragma: no cover - env without pymorphy2
    pymorphy2 = None  # type: ignore


_WORD_RE = re.compile(r"[\wЁё]+", re.UNICODE)
_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "по",
    "о",
    "об",
    "от",
    "до",
    "за",
    "из",
    "с",
    "со",
    "у",
    "к",
    "как",
    "а",
    "но",
    "что",
    "это",
    "же",
    "бы",
    "ли",
    "мы",
    "вы",
    "он",
    "она",
    "оно",
    "они",
}
_MORPH_ANALYZER = None


def _get_morph():
    global _MORPH_ANALYZER
    if pymorphy2 is None:
        _MORPH_ANALYZER = None
        return None
    if _MORPH_ANALYZER is None:
        try:
            _MORPH_ANALYZER = pymorphy2.MorphAnalyzer()
        except Exception:  # pragma: no cover - analyser init failures
            _MORPH_ANALYZER = None
    return _MORPH_ANALYZER


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            try:
                return int(text)
            except ValueError:
                return None
    return None


def _date_to_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if len(digits) >= 8:
            try:
                return int(digits[:8])
            except ValueError:
                return None
    return None


def _article_base_from_value(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        base = text.split(".", 1)[0]
        if base.isdigit():
            try:
                return int(base)
            except ValueError:
                return None
    return None


@dataclass
class SourceChunk:
    chunk: ChunkRecord
    score: float


@dataclass
class QueryResponse:
    answer: str
    sources: List[SourceChunk]


class QueryPipeline:
    def __init__(self, settings: AppSettings, embedding_client: EmbeddingClient, vector_store: QdrantVectorStore) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        api_key = settings.openai_api_key or None
        self._client = OpenAI(api_key=api_key)
        self._reranker = build_reranker(settings, self._client)
        self._chat_model: str | None = None

    def _format_context(self, sources: Sequence[SourceChunk]) -> str:
        lines = []
        for idx, source in enumerate(sources, start=1):
            chunk = source.chunk
            hierarchy = chunk.hierarchy or {}
            trail = []
            if hierarchy.get("part_no"):
                trail.append(f"Part {hierarchy['part_no']}")
            if hierarchy.get("article_no"):
                trail.append(f"Article {hierarchy['article_no']}")
            header = chunk.title_text or ", ".join(trail)
            lines.append(
                f"[{idx}] {header}\n{chunk.body_text}"
            )
        return "\n\n".join(lines)

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        top_k = top_k or self.settings.query.top_k
        sources = self._retrieve_sources(question, top_k)
        context = self._format_context(sources)
        prompt = (
            "You are a helpful assistant. Answer strictly using the provided context. "
            "If the answer is not in the context, say you don't know."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
        model = self._get_chat_model()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
        except PermissionDeniedError as exc:
            self._chat_model = None
            raise RuntimeError(
                "OpenAI denied access to the selected chat model. "
                "Please choose a model available to your account in the Settings tab."
            ) from exc
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI chat completion failed: {exc}") from exc

        answer = response.choices[0].message.content or "I don't know."
        return QueryResponse(answer=answer.strip(), sources=sources)

    def test_retrieval(self, question: str = "retrieval test", top_k: int = 3) -> List[SourceChunk]:
        """Perform a lightweight retrieval to validate search results."""

        limit = max(1, top_k or self.settings.query.top_k)
        return self._retrieve_sources(question, limit)

    def retrieve(self, question: str, top_k: int | None = None) -> List[SourceChunk]:
        top_k = top_k or self.settings.query.top_k
        return self._retrieve_sources(question, top_k)

    def _retrieve_sources(self, question: str, top_k: int) -> List[SourceChunk]:
        query_terms = self._tokenize_query(question)
        query_vector = self.embedding_client.embed_query(question)
        filter_ = self._build_combined_filter(question, top_k)
        body_results = self.vector_store.query(
            vector_name="body_vec",
            query_vector=query_vector,
            limit=max(top_k * 5, top_k + 40),
            filters=filter_,
        )
        title_results = self.vector_store.query(
            vector_name="title_vec",
            query_vector=query_vector,
            limit=max(top_k * 3, top_k + 20),
            filters=filter_,
        )
        fused_results = self._fuse_results(
            body_results,
            title_results,
            body_weight=self.settings.query.fusion_weight_body,
            title_weight=self.settings.query.fusion_weight_title,
            rrf_k=max(1, self.settings.query.fusion_rrf_k),
            query_terms=query_terms,
        )
        reranked = self._apply_reranker(question, query_terms, fused_results)
        return self._hydrate_sources(reranked[:top_k])

    def _build_combined_filter(self, question: str, top_k: int) -> rest.Filter | None:
        filters: List[rest.Filter] = []
        prefilter_limit = max(top_k * 5, self.settings.query.prefilter_limit)
        if prefilter_limit <= 0:
            prefilter = None
        else:
            doc_ids = self.vector_store.keyword_prefilter(question, prefilter_limit)
            prefilter = None
            if doc_ids:
                try:
                    prefilter = self.vector_store.build_doc_id_filter(doc_ids)
                except ValueError:
                    prefilter = None
        if prefilter:
            filters.append(prefilter)

        temporal_filter = self.vector_store.build_as_of_filter(
            status=self.settings.query.status_filter,
            as_of_start=self.settings.query.as_of_start_date,
            as_of_date=self.settings.query.as_of_date,
        )
        if temporal_filter:
            filters.append(temporal_filter)

        law_filter = self._build_law_filters()
        if law_filter:
            filters.append(law_filter)

        return self.vector_store.combine_filters(*filters)

    def _tokenize_query(self, query: str) -> List[str]:
        if not query:
            return []
        morph = _get_morph()
        tokens: List[str] = []
        for word in _WORD_RE.findall(query.lower()):
            if not word:
                continue
            lemma = word
            if morph is not None:
                try:
                    parsed = morph.parse(word)
                    if parsed:
                        lemma = parsed[0].normal_form
                except Exception:  # pragma: no cover - morphological failures
                    lemma = word
            if lemma in _STOPWORDS:
                continue
            if lemma:
                tokens.append(lemma)
        return tokens

    def _build_chapter_filter(self, chapter: str | None) -> rest.Filter | None:
        if chapter is None:
            return None
        chapter_text = chapter.strip().lower()
        if not chapter_text or chapter_text in {"all", "*"}:
            return None
        chapter_no = _coerce_int(chapter_text)
        if chapter_no is None:
            return None
        return rest.Filter(
            must=[
                rest.FieldCondition(
                    key="hierarchy.chapter_no",
                    match=rest.MatchValue(value=chapter_no),
                )
            ]
        )

    def _build_article_range_filter(
        self, start: str | None, end: str | None
    ) -> rest.Filter | None:
        if not hasattr(rest, "Range"):
            return None
        start_base = _article_base_from_value(start) if start else None
        end_base = _article_base_from_value(end) if end else None
        if start_base is None and end_base is None:
            return None
        range_kwargs: dict[str, int] = {}
        if start_base is not None:
            range_kwargs["gte"] = start_base
        if end_base is not None:
            range_kwargs["lte"] = end_base
        if not range_kwargs:
            return None
        condition = rest.FieldCondition(
            key="article_base",
            range=rest.Range(**range_kwargs),
        )
        return rest.Filter(must=[condition])

    def _build_law_filters(self) -> rest.Filter | None:
        if not getattr(self.settings.query, "enable_law_filters", True):
            return None
        must_conditions: List[rest.FieldCondition] = []
        status = (self.settings.query.status_filter or "").strip()
        if status:
            normalized = "in_force" if status.lower() in {"active", "in_force"} else status
            must_conditions.append(
                rest.FieldCondition(
                    key="law_meta.status",
                    match=rest.MatchValue(value=normalized),
                )
            )
        as_of_value = getattr(self.settings.query, "as_of_date", None)
        as_of_int = _date_to_int(as_of_value)
        if as_of_int is not None:
            if hasattr(rest, "Range"):
                try:
                    must_conditions.append(
                        rest.FieldCondition(
                            key="law_meta.date_from_int",
                            range=rest.Range(lte=as_of_int),
                        )
                    )
                    must_conditions.append(
                        rest.FieldCondition(
                            key="law_meta.date_to_int",
                            range=rest.Range(gte=as_of_int),
                        )
                    )
                except TypeError:
                    must_conditions.append(
                        rest.FieldCondition(
                            key="law_meta.date_from_int",
                            match=rest.MatchValue(value=as_of_int),
                        )
                    )
                    must_conditions.append(
                        rest.FieldCondition(
                            key="law_meta.date_to_int",
                            match=rest.MatchValue(value=as_of_int),
                        )
                    )
            else:
                must_conditions.append(
                    rest.FieldCondition(
                        key="law_meta.date_from_int",
                        match=rest.MatchValue(value=as_of_int),
                    )
                )
                must_conditions.append(
                    rest.FieldCondition(
                        key="law_meta.date_to_int",
                        match=rest.MatchValue(value=as_of_int),
                    )
                )
        part_filter = getattr(self.settings.query, "part_no", None)
        part_value = _coerce_int(str(part_filter)) if part_filter is not None else None
        if part_value is not None:
            must_conditions.append(
                rest.FieldCondition(
                    key="hierarchy.part_no",
                    match=rest.MatchValue(value=part_value),
                )
            )
        chapter_filter = getattr(self.settings.query, "chapter_no", None)
        chapter_value = _coerce_int(str(chapter_filter)) if chapter_filter is not None else None
        if chapter_value is not None:
            must_conditions.append(
                rest.FieldCondition(
                    key="hierarchy.chapter_no",
                    match=rest.MatchValue(value=chapter_value),
                )
            )
        article_filter = getattr(self.settings.query, "article_no_int", None)
        article_value = _coerce_int(str(article_filter)) if article_filter is not None else None
        if article_value is not None:
            must_conditions.append(
                rest.FieldCondition(
                    key="hierarchy.article_no_int",
                    match=rest.MatchValue(value=article_value),
                )
            )
        if not must_conditions:
            return None
        return rest.Filter(must=must_conditions)

    def _fuse_results(
        self,
        body_results: Sequence,
        title_results: Sequence,
        *,
        body_weight: float,
        title_weight: float,
        rrf_k: int,
        query_terms: Sequence[str] | None = None,
    ) -> List[object]:
        combined: dict[str, float] = {}
        registry: dict[str, object] = {}
        for weight, results in ((body_weight, body_results), (title_weight, title_results)):
            if weight <= 0:
                continue
            for rank, result in enumerate(results, start=1):
                point_id = str(getattr(result, "id", ""))
                if not point_id:
                    continue
                registry[point_id] = result
                combined[point_id] = combined.get(point_id, 0.0) + weight * (1.0 / (rrf_k + rank))
        boost_weight = getattr(self.settings.query, "keyword_boost_weight", 0.0)
        if query_terms and boost_weight > 0:
            lowered_terms = [term.lower() for term in query_terms if term]
            if lowered_terms:
                for point_id, result in registry.items():
                    payload = getattr(result, "payload", {}) or {}
                    snippets: List[str] = []
                    title = payload.get("title_text")
                    body = payload.get("body_text")
                    if isinstance(title, str):
                        snippets.append(title.lower())
                    if isinstance(body, str):
                        snippets.append(body.lower())
                    if not snippets:
                        continue
                    text_blob = " \\n".join(snippets)
                    hits = sum(1 for term in lowered_terms if term and term in text_blob)
                    if hits:
                        combined[point_id] = combined.get(point_id, 0.0) + boost_weight * hits

        sorted_ids = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        ordered_results: List[object] = []
        for point_id, score in sorted_ids:
            result = registry[point_id]
            setattr(result, "_combined_score", float(score))
            ordered_results.append(result)
        return ordered_results

    def _apply_reranker(
        self,
        question: str,
        query_terms: Sequence[str] | None,
        results: List[object],
    ) -> List[object]:
        if not results:
            return results
        rerank_limit = min(len(results), max(1, self.settings.query.reranker_top_n))
        working = results[:rerank_limit]
        base_scores = [float(getattr(result, "_combined_score", 0.0)) for result in working]
        max_base = max(base_scores) if base_scores else 1.0
        if max_base == 0:
            max_base = 1.0
        candidates: List[RerankCandidate] = []
        for result, base in zip(working, base_scores):
            payload = getattr(result, "payload", {}) or {}
            title = payload.get("title_text") if isinstance(payload.get("title_text"), str) else ""
            body = payload.get("body_text") if isinstance(payload.get("body_text"), str) else ""
            candidate_id = str(getattr(result, "id", len(candidates)))
            candidates.append(
                RerankCandidate(
                    candidate_id=candidate_id,
                    title=title or "",
                    body=body or "",
                    base_score=float(base / max_base),
                )
            )

        try:
            scores = self._reranker.rerank(question, candidates)
        except Exception as exc:  # pragma: no cover - reranker failures
            logger.warning("Reranker failed; using baseline order: %s", exc)
            return results
        if len(scores) != len(working):
            logger.debug(
                "Reranker returned mismatched score count (%s vs %s); using baseline order",
                len(scores),
                len(working),
            )
            return results

        scored = []
        for result, score in zip(working, scores):
            setattr(result, "_reranked_score", float(score))
            scored.append((result, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        reordered = [item[0] for item in scored]
        remainder = results[rerank_limit:]
        return reordered + list(remainder)

    def _hydrate_sources(self, results: Iterable[object]) -> List[SourceChunk]:
        sources: List[SourceChunk] = []
        for result in results:
            payload = result.payload or {}

            raw_title = payload.get("title_text")
            title_text = raw_title if isinstance(raw_title, str) else ""
            raw_body = payload.get("body_text")
            body_text = raw_body if isinstance(raw_body, str) else ""

            if not title_text and body_text:
                title_text = body_text[:120].strip()
            if not body_text and title_text:
                body_text = title_text

            chunk = ChunkRecord(
                doc_id=str(payload.get("doc_id") or ""),
                chunk_index=int(payload.get("chunk_index") or 0),
                chunk_id=str(payload.get("chunk_id") or ""),
                title_text=title_text,
                body_text=body_text,
                hierarchy=payload.get("hierarchy") or {},
                law_meta=payload.get("law_meta") or {},
                source=payload.get("source") or {},
                chunk_key=payload.get("chunk_key"),
                plan_version=payload.get("plan_version"),
                parser_version=payload.get("parser_version"),
                chunk_sha256=str(payload.get("chunk_sha256") or ""),
                title_sha256=str(payload.get("title_sha256") or ""),
                body_sha256=str(payload.get("body_sha256") or ""),
                law=str(payload.get("law") or ""),
                part=_coerce_int(payload.get("part")),
                article_str=str(payload.get("article_str") or ""),
                article_base=_coerce_int(payload.get("article_base")),
                article_suffix=_coerce_int(payload.get("article_suffix")),
                version_date=str(payload.get("version_date") or ""),
                content_sha256=str(payload.get("content_sha256") or ""),
                source_file=str(payload.get("source_file") or ""),
            )
            fused_score = getattr(result, "_reranked_score", None)
            if fused_score is None:
                fused_score = getattr(result, "_combined_score", None)
            if fused_score is None:
                fused_score = getattr(result, "score", 0.0)
            sources.append(SourceChunk(chunk=chunk, score=float(fused_score or 0.0)))
        return sources

    def _get_chat_model(self) -> str:
        if self._chat_model:
            return self._chat_model
        self._chat_model = self._resolve_chat_model()
        return self._chat_model

    def _resolve_chat_model(self) -> str:
        preferred = (self.settings.openai_models.chat or "").strip()
        candidates = []
        for candidate in [preferred, "gpt-4.1-mini", "gpt-4o-mini"]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        last_error: Exception | None = None
        probe_messages = [{"role": "user", "content": "ping"}]

        for model in candidates:
            try:
                self._client.chat.completions.create(
                    model=model,
                    messages=probe_messages,
                    max_tokens=1,
                    temperature=0,
                )
            except PermissionDeniedError as exc:
                logger.warning(
                    "Permission denied for OpenAI model %s during probe: %s", model, exc
                )
                last_error = exc
                continue
            except OpenAIError as exc:
                logger.error(
                    "OpenAI chat probe failed for model %s: %s", model, exc
                )
                raise RuntimeError(
                    f"OpenAI chat completion failed during model probe: {exc}"
                ) from exc
            else:
                logger.info("Using OpenAI chat model: %s", model)
                if model != preferred:
                    self.settings.openai_models.chat = model
                    try:
                        self.settings.save()
                    except Exception:
                        logger.exception("Failed to persist resolved chat model")
                return model

        raise RuntimeError(
            "No accessible OpenAI chat model found. Please update the Chat Model in Settings."
        ) from last_error
