from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .data_repository import DataRepository
from .id_utils import make_point_id
from .legal_types import (
    ArticleRecord,
    ChunkRecord,
    DEFAULT_PARSER_VERSION,
    HierarchyMetadata,
    LawMetadata,
)
from .readers.base import DocumentText
from .settings import AppSettings
from .structure_planner import StructurePlan, StructurePlanner

try:  # optional dependency for token-aware chunking
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency not available
    tiktoken = None  # type: ignore

logger = logging.getLogger(__name__)

_SECTION_RE = re.compile(r"^\s*РАЗДЕЛ\s+([IVXLC]+)\.?\s*(.*)$", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"^\s*ГЛАВА\s+(\d+)\.?\s*(.*)$", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^\s*СТАТЬЯ\s+((?:\d+(?:\.\d+)*))\.?\s*(.*)$", re.IGNORECASE)
_AMEND_RE = re.compile(r"ред\.?\s*от\s*(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE)
_LAW_NUMBER_RE = re.compile(r"(?:N|№)\s*([0-9\-А-Яа-яA-Za-z]+)")
_DATE_RE = re.compile(r"от\s*(\d{2}\.\d{2}\.\d{4})")
_PART_PATH_RE = re.compile(r"part[_\-\s]?(\d+)", re.IGNORECASE)
_PART_WORD_RE = re.compile(r"част[ьяи]\s+([а-я]+)", re.IGNORECASE)
_WORD_TO_NUM = {
    "первая": 1,
    "первой": 1,
    "вторая": 2,
    "второй": 2,
    "третья": 3,
    "третьей": 3,
    "четвертая": 4,
    "четвертой": 4,
    "пятая": 5,
    "пятой": 5,
}
def doc_prefix(doc_id: str) -> str:
    """Return the stable prefix for a legal document identifier."""

    if not doc_id:
        return ""
    if ":art" in doc_id:
        return doc_id.split(":art", 1)[0] + ":"
    if ":v" in doc_id:
        base = doc_id.split(":v", 1)[0]
        head, *_ = base.rsplit(":", 1)
        return f"{head}:"
    head, *_ = doc_id.rsplit(":", 1)
    return f"{head}:"


PARSER_VERSION = DEFAULT_PARSER_VERSION


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a deterministic identifier for a chunk."""

    return make_point_id(doc_id, chunk_index)


@dataclass
class ProcessedDocument:
    normalized_text_path: Path
    article_json_path: Path
    chunk_json_path: Path
    articles: List[ArticleRecord]
    chunks: List[ChunkRecord]
    doc_ids_changed: List[str]
    doc_ids_unchanged: List[str]


def normalize_text(raw: str) -> str:
    """Normalise raw document text prior to planning/parsing."""

    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")  # strip BOM
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"-\n", "", text)  # fix hyphenated breaks
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)  # drop standalone page numbers
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _iso_date(maybe_date: str | None) -> str | None:
    if not maybe_date:
        return None
    try:
        day, month, year = maybe_date.split(".")
        return f"{year}-{month}-{day}"
    except Exception:
        return None


def _infer_part_no(path: Path, text: str) -> int:
    match = _PART_PATH_RE.search(str(path))
    if match:
        return int(match.group(1))
    match = _PART_WORD_RE.search(path.name)
    if match:
        candidate = match.group(1).lower()
        return _WORD_TO_NUM.get(candidate, 0)
    match = _PART_WORD_RE.search(text)
    if match:
        candidate = match.group(1).lower()
        return _WORD_TO_NUM.get(candidate, 0)
    return 0


def _extract_law_number(path: Path, text: str) -> str | None:
    match = _LAW_NUMBER_RE.search(path.name)
    if match:
        return match.group(1)
    match = _LAW_NUMBER_RE.search(text)
    if match:
        return match.group(1)
    return None


def _extract_dates(text: str, path: Path) -> Tuple[str | None, str | None]:
    enact = None
    last_amend = None
    match = _DATE_RE.search(path.name)
    if match:
        enact = _iso_date(match.group(1))
    if not enact:
        match = _DATE_RE.search(text[:2000])
        if match:
            enact = _iso_date(match.group(1))
    amendments = _AMEND_RE.findall(text)
    if amendments:
        last_amend = _iso_date(amendments[-1])
    if not last_amend:
        last_amend = enact
    return enact, last_amend


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _article_doc_id(part_no: int, article_no: str, last_amend_date: str | None) -> str:
    version = last_amend_date or "undated"
    return f"gkrf:part{part_no}:art{article_no}:v{version}"


class LegalCorpusBuilder:
    def __init__(self, repository: DataRepository, settings: AppSettings | None = None) -> None:
        self.repository = repository
        self.settings = settings or AppSettings.load()
        self.planner = StructurePlanner(self.settings, repository)
        self.max_tokens = self.settings.chunking.max_tokens
        self.overlap_tokens = self.settings.chunking.overlap_tokens
        self.tokenizer = self._load_tokenizer(self.settings.chunking.tokenizer)

    def _load_tokenizer(self, name: str):
        if tiktoken is not None:
            try:
                return tiktoken.get_encoding(name)
            except Exception as exc:
                logger.warning(
                    "Tokenizer '%s' not available; using character-count fallback (%s)",
                    name,
                    exc,
                )
        else:
            logger.warning(
                "tiktoken not available; using character-count fallback",
                extra={"tokenizer": name},
            )

            class _CharacterTokenizer:
                def encode(self, text: str, disallowed_special=None):
                    return [ord(ch) for ch in text]

                def decode(self, tokens: List[int]) -> str:
                    return "".join(chr(token) for token in tokens)

            return _CharacterTokenizer()

    def process_document(
        self,
        document: DocumentText,
        *,
        normalized_text: str,
        chunk_size: int,
        overlap: int,
    ) -> ProcessedDocument:
        corpus_slug, part_slug = self.repository.derive_slugs(document.path)
        part_no = _infer_part_no(document.path, normalized_text)
        law_no = _extract_law_number(document.path, normalized_text)
        enact_date, last_amend = _extract_dates(normalized_text, document.path)

        law_meta = LawMetadata(
            corpus="ГК РФ",
            part_no=part_no,
            law_no=law_no,
            enact_date=enact_date,
            last_amend_date=last_amend,
            status="active",
            amendments=[],
        )

        staging_dir, chunk_dir = self.repository.ensure_subdirectories(corpus_slug, part_slug)
        normalized_path = self.repository.normalized_text_path(corpus_slug, part_slug)
        normalized_path.write_text(normalized_text, encoding="utf-8")

        plan = self.planner.plan_for_text(normalized_text)
        articles = self._extract_articles(
            normalized_text=normalized_text,
            law_meta=law_meta,
            source_path=document.path,
            part_no=part_no,
            plan=plan,
        )
        if not articles:
            logger.warning("Structure plan produced no articles; falling back to legacy parser")
            articles = self._extract_articles_legacy(
                normalized_text=normalized_text,
                law_meta=law_meta,
                source_path=document.path,
                part_no=part_no,
            )

        article_path = self.repository.article_json_path(corpus_slug, part_slug)
        self._write_jsonl(article_path, (self._article_to_json(article) for article in articles))

        chunk_records: List[ChunkRecord] = []
        for article in articles:
            chunk_records.extend(
                self._chunk_article(
                    article,
                    plan_version=plan.plan_version,
                    max_tokens=self.max_tokens,
                    overlap_tokens=self.overlap_tokens,
                )
            )

        chunk_path = self.repository.chunk_json_path(corpus_slug, part_slug)
        self._write_jsonl(chunk_path, (chunk.__dict__ for chunk in chunk_records))

        manifest_path = self.repository.manifest_path(corpus_slug, part_slug)
        existing_manifest = self._load_manifest(manifest_path)
        new_manifest, changed, unchanged = self._build_manifest(chunk_records, existing_manifest)
        self._write_manifest(manifest_path, new_manifest)

        return ProcessedDocument(
            normalized_text_path=normalized_path,
            article_json_path=article_path,
            chunk_json_path=chunk_path,
            articles=articles,
            chunks=chunk_records,
            doc_ids_changed=changed,
            doc_ids_unchanged=unchanged,
        )

    def _extract_articles(
        self,
        *,
        normalized_text: str,
        law_meta: LawMetadata,
        source_path: Path,
        part_no: int,
        plan: StructurePlan,
    ) -> List[ArticleRecord]:
        compiled: Dict[str, re.Pattern[str]] = {}
        level_order: List[str] = []
        for level in plan.levels:
            try:
                compiled[level.name] = re.compile(level.regex, re.IGNORECASE)
                level_order.append(level.name)
            except re.error as exc:
                logger.warning("Invalid regex for level %s: %s", level.name, exc)
        if plan.split.primary not in compiled:
            logger.warning("Primary split level '%s' missing from plan", plan.split.primary)
            return []

        name_to_index = {name: idx for idx, name in enumerate(level_order)}
        state: Dict[str, str | None] = {name: None for name in level_order}
        titles: Dict[str, str | None] = {name: None for name in level_order}

        current_article_no: str | None = None
        current_article_title: str = ""
        current_lines: List[str] = []
        articles: List[ArticleRecord] = []

        lines = normalized_text.splitlines()

        def reset_lower(level_name: str) -> None:
            level_index = name_to_index.get(level_name, -1)
            if level_index == -1:
                return
            for lower_name in level_order[level_index + 1 :]:
                state[lower_name] = None
                titles[lower_name] = None

        def finalize_article() -> None:
            nonlocal current_article_no, current_article_title, current_lines
            if current_article_no is None or not current_lines:
                return
            hierarchy = HierarchyMetadata(
                section_roman=state.get("section_roman"),
                section_title=titles.get("section_roman"),
                chapter_no=int(state["chapter_no"]) if state.get("chapter_no") else None,
                chapter_title=titles.get("chapter_no"),
                article_no=current_article_no,
                article_title=current_article_title or "",
            )
            articles.append(
                self._finalize_article(
                    part_no=part_no,
                    article_no=current_article_no,
                    article_title=current_article_title,
                    lines=current_lines,
                    hierarchy=hierarchy,
                    law_meta=law_meta,
                    source_path=source_path,
                )
            )
            current_article_no = None
            current_article_title = ""
            current_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    current_lines.append("")
                continue

            matched_level: str | None = None
            for level in plan.levels:
                pattern = compiled.get(level.name)
                if pattern is None:
                    continue
                match = pattern.match(stripped)
                if not match:
                    continue
                matched_level = level.name
                groupdict = match.groupdict()
                value = groupdict.get("num")
                if value:
                    state[level.name] = value.strip()
                remainder = groupdict.get("title")
                if remainder is None:
                    remainder = stripped[match.end():].strip()
                if remainder:
                    titles[level.name] = remainder.strip()
                reset_lower(level.name)
                if level.name == plan.split.primary:
                    finalize_article()
                    current_article_no = value.strip() if value else stripped
                    current_article_title = (remainder or "").strip(" .")
                    current_lines = [stripped]
                else:
                    if current_lines:
                        current_lines.append(stripped)
                break

            if matched_level is None and current_lines is not None:
                if current_article_no is not None:
                    current_lines.append(stripped)

        finalize_article()
        return articles

    def _extract_articles_legacy(
        self,
        normalized_text: str,
        law_meta: LawMetadata,
        source_path: Path,
        part_no: int,
    ) -> List[ArticleRecord]:
        section_roman: str | None = None
        section_title: str | None = None
        chapter_no: int | None = None
        chapter_title: str | None = None
        current_article_no: str | None = None
        current_article_title: str = ""
        current_lines: List[str] = []
        articles: List[ArticleRecord] = []

        lines = normalized_text.splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    current_lines.append("")
                continue
            section_match = _SECTION_RE.match(stripped)
            if section_match:
                section_roman = section_match.group(1)
                section_title = section_match.group(2).strip() or None
                continue
            chapter_match = _CHAPTER_RE.match(stripped)
            if chapter_match:
                chapter_no = int(chapter_match.group(1))
                chapter_title = chapter_match.group(2).strip() or None
                continue
            article_match = _ARTICLE_RE.match(stripped)
            if article_match:
                if current_article_no is not None:
                    articles.append(
                        self._finalize_article(
                            part_no=part_no,
                            article_no=current_article_no,
                            article_title=current_article_title,
                            lines=current_lines,
                            hierarchy=HierarchyMetadata(
                                section_roman=section_roman,
                                section_title=section_title,
                                chapter_no=chapter_no,
                                chapter_title=chapter_title,
                                article_no=current_article_no,
                                article_title=current_article_title,
                            ),
                            law_meta=law_meta,
                            source_path=source_path,
                        )
                    )
                    current_lines = []
                current_article_no = article_match.group(1)
                current_article_title = article_match.group(2).strip()
                heading = stripped
                current_lines = [heading]
                continue
            if current_article_no is not None:
                current_lines.append(stripped)

        if current_article_no is not None and current_lines:
            articles.append(
                self._finalize_article(
                    part_no=part_no,
                    article_no=current_article_no,
                    article_title=current_article_title,
                    lines=current_lines,
                    hierarchy=HierarchyMetadata(
                        section_roman=section_roman,
                        section_title=section_title,
                        chapter_no=chapter_no,
                        chapter_title=chapter_title,
                        article_no=current_article_no,
                        article_title=current_article_title,
                    ),
                    law_meta=law_meta,
                    source_path=source_path,
                )
            )

        return articles

    def _finalize_article(
        self,
        *,
        part_no: int,
        article_no: str,
        article_title: str,
        lines: Sequence[str],
        hierarchy: HierarchyMetadata,
        law_meta: LawMetadata,
        source_path: Path,
    ) -> ArticleRecord:
        text = "\n".join(line for line in lines if line is not None).strip()
        source_sha = _hash_text(text)
        doc_id = _article_doc_id(part_no, article_no, law_meta.last_amend_date)
        title_text = f"Статья {article_no}. {article_title}".strip()
        relative_path = self.repository.relative_to_raw(source_path)
        return ArticleRecord(
            doc_id=doc_id,
            title_text=title_text,
            article_text=text,
            hierarchy=hierarchy,
            law_meta=law_meta,
            source_path=source_path,
            source_file_name=source_path.name,
            source_relative_path=relative_path,
            source_sha256=source_sha,
        )

    def _chunk_article(
        self,
        article: ArticleRecord,
        *,
        plan_version: str,
        max_tokens: int,
        overlap_tokens: int,
    ) -> List[ChunkRecord]:
        paragraphs = [p.strip() for p in article.article_text.split("\n") if p.strip()]
        if not paragraphs:
            return []

        tokenised: List[tuple[str, List[int]]] = []
        for paragraph in paragraphs:
            tokens = self.tokenizer.encode(paragraph, disallowed_special=())
            if tokens and len(tokens) > max_tokens:
                for sentence in self._split_sentences(paragraph):
                    sentence_tokens = self.tokenizer.encode(sentence, disallowed_special=())
                    if sentence_tokens and len(sentence_tokens) > max_tokens:
                        for chunk_tokens in self._slice_tokens(sentence_tokens, max_tokens):
                            tokenised.append((self.tokenizer.decode(chunk_tokens).strip(), chunk_tokens))
                    else:
                        tokenised.append((sentence, sentence_tokens))
            else:
                tokenised.append((paragraph, tokens))

        chunks: List[ChunkRecord] = []
        window: List[tuple[str, List[int]]] = []
        token_total = 0
        chunk_index = 0

        for text_piece, tokens in tokenised:
            tokens = tokens or []
            if window and token_total + len(tokens) > max_tokens:
                chunks.append(
                    self._make_chunk(
                        article,
                        chunk_index,
                        window,
                        plan_version=plan_version,
                    )
                )
                chunk_index += 1
                if overlap_tokens > 0:
                    window, token_total = self._build_overlap(window, overlap_tokens)
                else:
                    window = []
                    token_total = 0
            window.append((text_piece, tokens))
            token_total += len(tokens)

        if window:
            chunks.append(
                self._make_chunk(
                    article,
                    chunk_index,
                    window,
                    plan_version=plan_version,
                )
            )
        return chunks

    def _build_overlap(
        self,
        window: List[tuple[str, List[int]]],
        overlap_tokens: int,
    ) -> tuple[List[tuple[str, List[int]]], int]:
        overlap: List[tuple[str, List[int]]] = []
        total = 0
        for text_piece, tokens in reversed(window):
            overlap.insert(0, (text_piece, tokens))
            total += len(tokens)
            if total >= overlap_tokens:
                break
        return overlap, total

    def _split_sentences(self, paragraph: str) -> List[str]:
        sentences = re.split(r"(?<=[\.!?])\s+", paragraph)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _slice_tokens(self, tokens: List[int], max_tokens: int) -> List[List[int]]:
        return [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]

    def _make_chunk(
        self,
        article: ArticleRecord,
        chunk_index: int,
        window: List[tuple[str, List[int]]],
        *,
        plan_version: str,
    ) -> ChunkRecord:
        body = "\n".join(part for part, _ in window).strip()
        body_sha = _hash_text(body)
        title_sha = _hash_text(article.title_text)
        chunk_id = make_chunk_id(article.doc_id, chunk_index)
        chunk_key = f"{article.doc_id}#c{chunk_index:04d}"
        source_path = article.source_relative_path or article.source_file_name
        return ChunkRecord(
            doc_id=article.doc_id,
            chunk_index=chunk_index,
            chunk_id=chunk_id,
            chunk_key=chunk_key,
            title_text=article.title_text,
            body_text=body,
            hierarchy={
                "corpus": article.law_meta.corpus,
                "part_no": article.law_meta.part_no,
                "section_roman": article.hierarchy.section_roman,
                "section_title": article.hierarchy.section_title,
                "chapter_no": article.hierarchy.chapter_no,
                "chapter_title": article.hierarchy.chapter_title,
                "article_no": article.hierarchy.article_no,
                "article_title": article.hierarchy.article_title,
                "clause_no": None,
            },
            law_meta={
                "law_no": article.law_meta.law_no,
                "enact_date": article.law_meta.enact_date,
                "last_amend_date": article.law_meta.last_amend_date,
                "status": article.law_meta.status,
                "amendments": list(article.law_meta.amendments),
            },
            source={
                "file_name": article.source_file_name,
                "relative_path": article.source_relative_path,
                "source_path": source_path,
                "source_sha256": article.source_sha256,
            },
            plan_version=plan_version,
            parser_version=PARSER_VERSION,
            chunk_sha256=body_sha,
            title_sha256=title_sha,
            body_sha256=body_sha,
        )

    def _write_jsonl(self, path: Path, rows: Iterable[Dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")

    def _article_to_json(self, article: ArticleRecord) -> Dict[str, object]:
        return {
            "doc_id": article.doc_id,
            "title_text": article.title_text,
            "hierarchy": {
                "corpus": article.law_meta.corpus,
                "part_no": article.law_meta.part_no,
                "section_roman": article.hierarchy.section_roman,
                "section_title": article.hierarchy.section_title,
                "chapter_no": article.hierarchy.chapter_no,
                "chapter_title": article.hierarchy.chapter_title,
                "article_no": article.hierarchy.article_no,
                "article_title": article.hierarchy.article_title,
            },
            "law_meta": {
                "law_no": article.law_meta.law_no,
                "enact_date": article.law_meta.enact_date,
                "last_amend_date": article.law_meta.last_amend_date,
                "status": article.law_meta.status,
                "amendments": list(article.law_meta.amendments),
            },
            "source_file_name": article.source_file_name,
            "source_relative_path": article.source_relative_path,
            "source_path": article.source_relative_path or article.source_file_name,
            "source_sha256": article.source_sha256,
            "article_text": article.article_text,
        }

    def _load_manifest(self, path: Path) -> Dict[str, Dict[str, object]]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            logger.exception("Failed to load manifest %s", path)
        return {}

    def _build_manifest(
        self,
        chunks: Sequence[ChunkRecord],
        existing: Dict[str, Dict[str, object]],
    ) -> Tuple[Dict[str, Dict[str, object]], List[str], List[str]]:
        grouped: Dict[str, List[ChunkRecord]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.doc_id, []).append(chunk)
        new_manifest: Dict[str, Dict[str, object]] = dict(existing)
        changed: List[str] = []
        unchanged: List[str] = []
        for doc_id, doc_chunks in grouped.items():
            chunk_shas = [chunk.chunk_sha256 for chunk in doc_chunks]
            entry = {
                "source_sha": doc_chunks[0].source.get("source_sha256"),
                "chunk_shas": chunk_shas,
            }
            if doc_id not in existing or existing[doc_id] != entry:
                new_manifest[doc_id] = entry
                changed.append(doc_id)
            else:
                unchanged.append(doc_id)
        return new_manifest, changed, unchanged

    def _write_manifest(self, path: Path, manifest: Dict[str, Dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

