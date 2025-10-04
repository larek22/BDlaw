"""High level orchestration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from bdlaw.index.embedder import OpenAIEmbedder
from bdlaw.index.pipeline import IndexingPipeline, IndexingStats
from bdlaw.index.vector_store import QdrantVectorStore
from bdlaw.ingest.base import Document, ImportPipeline
from bdlaw.ingest.docx import DOCXImporter
from bdlaw.ingest.html import HTMLImporter
from bdlaw.ingest.ocr import OCRImporter
from bdlaw.ingest.pdf import PDFImporter
from bdlaw.ingest.rtf import RTFImporter
from bdlaw.ingest.text import TextImporter
from bdlaw.index.schema import NormPayload
from bdlaw.normalize import TextNormalizer
from bdlaw.parser.chunker import chunk_norms
from bdlaw.parser.model import DocumentStructure
from bdlaw.parser.structure import ParsedDocument, ParsingContext, parse_document
from bdlaw.search.service import SearchService
from bdlaw.settings.config import AppConfig
from bdlaw.utils.files import discover_documents


ProgressCallback = Callable[[str], None]


@dataclass
class IngestionResult:
    """Container aggregating import and parsing outputs."""

    documents: List[Document]
    parsed: ParsedDocument

    @property
    def norms(self) -> List[NormPayload]:
        return self.parsed.norms


class ApplicationPipelines:
    """Facade that wires ingestion, parsing, chunking, and indexing."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.normalizer = TextNormalizer(replace_yo=config.normalization.replace_yo)
        self.import_pipeline = ImportPipeline(
            importers=[
                PDFImporter(),
                DOCXImporter(),
                RTFImporter(),
                HTMLImporter(),
                TextImporter(),
                OCRImporter(config.ocr),
            ],
            normalizers=[self.normalizer],
        )
        embedder = OpenAIEmbedder(config.embedding)
        store = QdrantVectorStore(config.qdrant)
        self.index_pipeline = IndexingPipeline(embedder, store, config.chunking)

    def ingest_paths(
        self,
        paths: Iterable[Path],
        context: ParsingContext,
        progress: Optional[ProgressCallback] = None,
    ) -> IngestionResult:
        """Import, normalize, parse, and chunk *paths* into norms."""

        documents: List[Document] = []
        norms = []
        structures: List[DocumentStructure] = []
        resolved_paths = list(discover_documents(paths))
        for idx, path in enumerate(resolved_paths, start=1):
            if progress:
                progress(f"Импорт ({idx}/{len(resolved_paths)}): {path.name}")
            document = self.import_pipeline.import_document(path)
            documents.append(document)
            if progress:
                progress(f"Парсинг: {path.name}")
            parsed = parse_document(document, context)
            structures.append(parsed.structure)
            chunked = chunk_norms(parsed.norms, self.config.chunking)
            norms.extend(chunked)
            if progress:
                progress(f"Получено норм: {len(chunked)}")
        combined_structure = DocumentStructure()
        for structure in structures:
            combined_structure.divisions.extend(structure.divisions)
            combined_structure.chapters.extend(structure.chapters)
            combined_structure.sections.extend(structure.sections)
            combined_structure.articles.extend(structure.articles)
        parsed_document = ParsedDocument(norms=norms, structure=combined_structure)
        return IngestionResult(documents=documents, parsed=parsed_document)

    def index_parsed(
        self,
        parsed: ParsedDocument,
        progress: Optional[ProgressCallback] = None,
    ) -> IndexingStats:
        """Index parsed norms in the configured vector store."""

        if progress:
            progress("Расчёт эмбеддингов и запись в Qdrant…")
        stats = self.index_pipeline.index(parsed.norms)
        if progress:
            progress(f"Индексировано норм: {stats.norms_indexed}")
        return stats

    def build_search_service(self) -> SearchService:
        """Create a :class:`SearchService` bound to the current pipelines."""

        return SearchService(self.index_pipeline.embedder, self.index_pipeline.store)


__all__ = ["ApplicationPipelines"]
