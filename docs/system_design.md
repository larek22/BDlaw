# BDlaw System Design Overview

## 1. Product Vision and Scope
- **Goal**: Desktop cross-platform (Windows priority) application for legal document ingestion, structured normalization, vector indexing, and GPT-backed retrieval with citations.
- **Primary users**:
  - *Quick user*: imports files, triggers indexing, runs search and consumes cited answers.
  - *Advanced user*: adjusts parsing/indexing settings, manages models, executes evaluation suites.
- **Out of scope for v1**: real-time collaboration, web deployment, complex ACL, managed cloud backend.

## 2. Architecture Summary
```
/app
  /ui
  /ingest
  /parser
  /index
  /search
  /gpt
  /testsuite
  /settings
  /export
  /utils
```
- **Language/UI**: Python + PySide6 (preferred) for fast iteration and native feel on Windows; packaged with PyInstaller.
- **Core services**: local Qdrant vector database (Docker or binary). Optional connectors to pgvector/Milvus/Weaviate in later releases.
- **Model configuration**: GPT-4.1 / GPT-4.1-mini for answer generation and automated evaluations. Default embeddings via `text-embedding-3-small` (1536-d), pluggable alternatives via sentence-transformers.

## 3. Processing Pipeline
1. **Import**: drag-and-drop queue with MIME detection.
2. **Extraction** by format:
   - PDF via PyMuPDF (primary) and pdfminer.six fallback.
   - DOCX via python-docx or mammoth (HTML conversion).
   - RTF via pypandoc/unrtf.
   - HTML via readability + BeautifulSoup.
   - Images/scanned PDFs via Tesseract OCR (rus+eng) with optional PaddleOCR.
3. **Normalization**: remove headers/footers, normalize Unicode, keep `page_spans` metadata.
4. **Structure parsing**: regex/spaCy-driven detection of `Статья/Часть/Пункт/Подпункт`; interactive UI for corrections.
5. **Metadata enrichment**: attach jurisdiction, act metadata, versioning fields, source URL.
6. **Indexing**: chunk by legal norm, embed, and batch upsert into Qdrant with deduplication by deterministic `gid`/`hash`.
7. **Retrieval & Answering**: vector search with filters (law, effective date, language), GPT synthesis with strict citations.
8. **Testing**: golden-set ingestion, compute Recall@k, Precision@k, MRR@k, Pass@k, LLM-as-judge validation; export CSV/JSON reports.
9. **Export**: JSONL dumps of norms, Qdrant snapshots, indexing logs.

## 4. Data Model
Each indexed norm captures legal granularity and versioning:
```json
{
  "gid": "sha256(jurisdiction|law_code|article|part|point|subpoint|version_id)",
  "jurisdiction": "ru",
  "act_type": "federal_law",
  "law_code": "ГК РФ",
  "law_full_title": "Гражданский кодекс РФ (часть первая)",
  "article": "395",
  "part": "1",
  "point": null,
  "subpoint": null,
  "version_id": "2024-09-01",
  "valid_from": "2024-09-01",
  "valid_to": null,
  "supersedes": "2023-03-01",
  "source_url": "https://publication.pravo.gov.ru/Document/View/...",
  "citation": "ГК РФ ст.395 ч.1 (ред. 2024-09-01)",
  "text": "За пользование чужими денежными средствами...",
  "hash": "sha256(нормализованный текст)",
  "tokens_est": 180,
  "ingested_at": "2025-10-04T10:00:00Z",
  "file_origin": "path/original.pdf",
  "page_spans": [[12, 14]],
  "lang": "ru"
}
```
- Deterministic identifiers enable stable references, deduplication, and caching.
- Version fields (`version_id`, `valid_from`, `valid_to`, `supersedes`) preserve historical revisions.
- `citation` is stored ready for presentation, ensuring consistent answers.

## 5. Security & Privacy
- Use OS keyring (Windows Credential Manager/macOS Keychain/Secret Service) for API keys; optional encrypted `.env` fallback with explicit warning.
- Mask keys in logs, avoid storing secrets in exports or code.
- Support offline mode (ingestion/search without GPT responses).

## 6. UX Highlights
- Dark theme first, accessible typography, clear states (idle/processing/success/warning/error).
- Home screen shortcuts: Import, Index, Search, Tests, Settings.
- Keyboard shortcuts: `Ctrl+I` (import), `Ctrl+F` (search), `Ctrl+Enter` (index), `F5` (tests).
- Progressive disclosure: simple defaults with expandable advanced settings.

## 7. Release Roadmap
- **v1 (4–6 weeks)**: ingestion (PDF/DOCX/RTF/TXT/HTML + OCR), norm parsing with manual edits, Qdrant indexing, GPT answers with citations, keyring-backed settings, minimal Recall@k tester.
- **v1.1**: expanded QA metrics, export flows, UX enhancements.
- **v2**: multi-store plugins, MCP server, hybrid search, automated version diffs.

## 8. Testing & Quality Gates
- Text extraction accuracy ≥95% on benchmark set.
- Structure detection ≥90% auto-recognition of headings, with efficient manual correction.
- Index integrity: no duplicates by `hash`/`gid`; indexing summary report.
- Retrieval latency <300 ms at k ≤10 on local hardware.
- GPT answers must include at least one verified citation and `source_url`.
- QA runner outputs metrics and exports reports (CSV/JSON).

## 9. Tooling & Dependencies
- Python packages: `pymupdf`, `pdfminer.six`, `python-docx`, `pypandoc`, `beautifulsoup4`, `readability-lxml`, `pytesseract`, `Pillow`, `spacy` (`ru_core_news_md`), `regex`, `openai>=1.0.0`, `qdrant-client`, `keyring`, `pydantic`, `tiktoken`, `rich`, `loguru`, `PySide6`.
- External services: Dockerized Qdrant (`docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant`).
- Build: PyInstaller for Windows `.exe`; optional Docker Compose for bundled Qdrant.

## 10. Documentation & Support
- Built-in help center: FAQ, quick-start videos, key handling guidance.
- Guides on preparing golden test sets, migrating storage backends, and securing API keys.

