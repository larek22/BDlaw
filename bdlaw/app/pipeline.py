"""High level orchestration pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from bdlaw.index.embedder import OpenAIEmbedder
from bdlaw.index.pipeline import IndexingPipeline
from bdlaw.index.vector_store import QdrantVectorStore
from bdlaw.ingest.base import Document, ImportPipeline
from bdlaw.ingest.docx import DOCXImporter
from bdlaw.ingest.html import HTMLImporter
from bdlaw.ingest.ocr import OCRImporter
from bdlaw.ingest.pdf import PDFImporter
from bdlaw.ingest.rtf import RTFImporter
from bdlaw.ingest.text import TextImporter
from bdlaw.parser.chunker import chunk_norms
from bdlaw.parser.normalize import normalize_document
from bdlaw.parser.structure import ParsedDocument, ParsingContext, parse_document
from bdlaw.settings.config import AppConfig
from bdlaw.utils.files import discover_documents


class ApplicationPipelines:
    def __init__(self, config: AppConfig):
        self.config = config
        self.import_pipeline = ImportPipeline(
            importers=[
                PDFImporter(),
                DOCXImporter(),
                RTFImporter(),
                HTMLImporter(),
                TextImporter(),
                OCRImporter(config.ocr),
            ],
            normalizers=[normalize_document],
        )
        embedder = OpenAIEmbedder(config.embedding)
        store = QdrantVectorStore(config.qdrant)
        self.index_pipeline = IndexingPipeline(embedder, store, config.chunking)

    def ingest_paths(self, paths: Iterable[Path], context: ParsingContext) -> ParsedDocument:
        documents = [self.import_pipeline.import_document(path) for path in discover_documents(paths)]
        norms = []
        for document in documents:
            parsed = parse_document(document, context)
            norms.extend(chunk_norms(parsed.norms, self.config.chunking))
        return ParsedDocument(norms=norms)

    def index_parsed(self, parsed: ParsedDocument) -> None:
        self.index_pipeline.index(parsed.norms)


__all__ = ["ApplicationPipelines"]
