# Vector KB Assistant

Vector KB Assistant is a desktop application built with PySide6 that lets you ingest local documents, store semantic embeddings in Qdrant, and ask grounded questions using OpenAI models. The application keeps all data-processing local except for calls to OpenAI for embeddings and language-model responses.

## Features

- Ingest PDF, DOCX, TXT, and RTF files.
- Normalise raw legal texts and plan their section/chapter/article hierarchy automatically (LLM-assisted with cached fallbacks).
- Chunk documents with configurable size/overlap **and** tokenizer-aware windows aligned to the embedding model.
- Cache embeddings locally to reduce repeated costs.
- Store chunk metadata and vectors in a Qdrant collection (local or Cloud) with deterministic UUIDv5 point IDs and per-document
  reconciliation to guarantee idempotent re-ingests.
- Parse Russian legal documents into Part/Chapter/Article structures with law-aware metadata and deterministic document versions.
- Emit dual named vectors (`title_vec`, `body_vec`) for hybrid retrieval, maintain payload indexes for law-specific filters, and
  persist chunk provenance including parser/plan versions and auditing keys.
- Ask natural-language questions with answers grounded in retrieved chunks.
- View citations and highlighted source snippets for transparency.
- Responsive PySide6 GUI with background threads for long-running tasks.
- Built-in health checks for OpenAI and Qdrant connectivity.

## Getting Started

### Prerequisites

- Python 3.11
- Access to OpenAI models `text-embedding-3-large` and a chat-capable model. The default is `gpt-4o-mini`, and the app will automatically fall back to it if another configured model is denied.
- A running Qdrant instance. For Qdrant Cloud, copy the HTTPS endpoint and API key from the Cloud console. For local testing you can use Docker: `docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant`.

### Installation

The project pins all runtime dependencies to reproducible versions. The most critical ones are:

| Package | Version |
| --- | --- |
| `qdrant-client` | 1.9.1 |
| `httpx` | 0.27.0 |
| `openai` | 1.30.1 |
| `pydantic` | 2.8.2 |

At runtime the app logs the detected Qdrant server version, the `qdrant-client` version above, and the active embedding model so the environment is always auditable.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenAI key in the environment or through the Settings tab:

```bash
export OPENAI_API_KEY=your_key_here
```

### Running the App

```bash
python main.py
```

The first launch creates a configuration directory at `~/.vector_kb` containing `config.json`, `app.log`, and an embeddings cache.

## Usage

1. Open the **Settings** tab to review model names, Qdrant settings, and chunking parameters. Save any changes and optionally run the OpenAI/Qdrant tests.
* When targeting **Qdrant Cloud**, set the Qdrant URL to the HTTPS endpoint (for example, `https://<cluster-id>.cloud.qdrant.io`) and paste the API key provided by Qdrant. The app enforces the API key for HTTPS endpoints, refuses to send it to `http://localhost`, and logs the exact host that will be used (`Using Qdrant endpoint: ...`).
   * The collection is automatically created (or recreated) with the correct vector dimension for the configured embedding model. If you switch models, re-run ingestion with **Rebuild Collection** enabled so the schema matches the new embedding size.
   * Query settings expose both the `Top K` result size and a `Prefilter limit`, which controls how many document ids are gathered via full-text search before the dual vector searches run.
2. Switch to the **Ingest** tab, add documents, choose whether to rebuild the collection, and click **Create Vector DB**. Progress appears in the log pane.
   * Ingestion normalises text into `data/staging/<corpus>/<part>/*.normalized.txt`, extracts article metadata into `*.articles.jsonl`, and writes chunk manifests under `data/chunks/.../*.chunks.jsonl` for deterministic refreshes.
   * Each article version receives a document id such as `gkrf:part3:art1110:v2024-08-08`; re-ingesting unchanged versions is idempotent, while changed versions are deleted and re-upserted by doc id.
   * After every batch upsert the log shows the updated Qdrant point count, the delta versus the previous run, and a verification search so you can confirm vectors are persisted.
   * Use **Reset DB** for a one-click drop-and-recreate of the collection before a clean rebuild. The app confirms the action, recreates the named-vector schema, and logs the reset so you start from a known-good baseline.
3. After ingestion, go to the **Ask** tab, enter a question, and click **Search & Answer**. Answers include citations, and sources display highlighted query terms.

## Configuration

The application persists configuration to `~/.vector_kb/config.json`. Any field can also be overridden via environment variables before launch.

| Setting | Environment variable | Description |
| --- | --- | --- |
| `openai_api_key` | `OPENAI_API_KEY` | API key used for embeddings and chat completions. |
| `openai_models.embedding` | `OPENAI_EMBEDDING_MODEL` | Embedding model (default `text-embedding-3-large`). |
| `openai_models.chat` | `OPENAI_CHAT_MODEL` | Preferred chat model; the pipeline falls back to `gpt-4.1-mini` then `gpt-4o-mini` if access is denied. |
| `qdrant.url` | `QDRANT_URL` | Qdrant endpoint. HTTPS endpoints require an API key. |
| `qdrant.api_key` | `QDRANT_API_KEY` | API key used when the URL is HTTPS or a remote HTTP host. |
| `qdrant.collection` | `QDRANT_COLLECTION` | Collection name (default `kb_docs_v1`). |
| `qdrant.upsert_batch_size` | `QDRANT_UPSERT_BATCH_SIZE` | Batch size for point upserts. |
| `qdrant.timeout_seconds` | `QDRANT_TIMEOUT_SECONDS` | HTTP timeout used for Qdrant requests. |
| `ingest.chunk_size_chars` | `CHUNK_SIZE` | Target chunk size. |
| `ingest.chunk_overlap_chars` | `CHUNK_OVERLAP` | Character overlap between chunks. |
| `query.top_k` | `QUERY_TOP_K` | Maximum number of chunks returned per query. |
| `query.prefilter_limit` | `QUERY_PREFILTER_LIMIT` | Candidate budget for keyword prefiltering. |
| `query.fusion_weight_title` | `FUSION_WEIGHT_TITLE` | Weight applied to title-vector rankings during fusion. |
| `query.fusion_weight_body` | `FUSION_WEIGHT_BODY` | Weight applied to body-vector rankings during fusion. |
| `query.fusion_rrf_k` | `FUSION_RRF_K` | `k` constant in reciprocal-rank fusion. |
| `query.as_of_start_date` | `QUERY_AS_OF_START` | Optional lower bound for temporal filtering (ISO date). |
| `query.as_of_date` | `QUERY_AS_OF_DATE` | Optional upper bound (“as-of” date) for temporal filtering. |
| `query.status_filter` | `QUERY_STATUS` | Status filter applied to `law_meta.status` (default `active`). |

## Troubleshooting

- **OpenAI model access errors**: On the first question the app probes the configured chat model and two fallbacks (`gpt-4.1-mini`, `gpt-4o-mini`). If none are accessible you will see a clear error asking you to change the Chat Model in Settings.
- **Qdrant collection stays empty**: Confirm the ingestion log includes `[UPSERT]` lines with the expected batch sizes and `dim`. Afterwards the log prints `[QDRANT] total points now: <count>`; if the count does not rise, re-check the configured endpoint and API key.
- **Re-ingesting adds duplicates**: Point IDs are deterministic. If counts climb unexpectedly, enable **Rebuild Collection** to clear stale data and rerun ingestion.

## Data pipeline and storage layout

```
data/
  raw/                # Original source files supplied by the user
  staging/            # Normalised text + per-article JSONL files
  chunks/             # Chunk JSONL + manifest files used for deterministic refresh
  snapshots/          # Exported Qdrant snapshot files
  logs/               # Verification transcripts and additional logs
```

Each staging JSON record preserves the legal hierarchy (Part/Chapter/Article) along with law numbers, enact dates, and amendment metadata. Chunk JSON lines store the chunk hash, deterministic UUID, and named-vector texts ready for embedding. Payload metadata stores the original file name and relative path within `data/raw` so the corpus can be moved between machines without leaking absolute host paths.

## Qdrant schema

The application creates (or recreates) the collection with two named vectors per point:

```
{
  "vectors": {
    "title_vec": {"size": 3072, "distance": "Cosine"},
    "body_vec":  {"size": 3072, "distance": "Cosine"}
  }
}
```

Payload indexes are installed for `doc_id`, the legal hierarchy fields, law status and amendment date, and for full-text search over `title_text` and `body_text`. Hybrid retrieval first applies a keyword filter, runs vector search on both named vectors, and fuses the results using reciprocal-rank fusion with configurable weights. Temporal filters derived from `QUERY_STATUS`, `QUERY_AS_OF_START`, and `QUERY_AS_OF_DATE` ensure answers respect “as-of” queries.

## Exporting the vector database

1. **Snapshot (recommended)**

   ```bash
   python snapshots.py create
   ```

   The script calls `create_snapshot` then downloads the file into `snapshots/` (or a path provided via `--output`). Restore with:

   ```bash
   python snapshots.py restore snapshots/kb_docs_v1-YYYY-MM-DD-HHMMSS.snapshot
   ```

   Restores automatically trigger the verification suite; the command fails if reference queries or self-hit checks regress.

2. **Physical copy** – stop Qdrant and copy `~/.qdrant/storage/collections/kb_docs_v1` to another machine with the same Qdrant version.

## Tests

Run the test suite with:

```bash
pytest
```

## Logging

Application logs are written to `~/.vector_kb/app.log` and streamed into the GUI log view during ingestion and queries.

## Verification & diagnostics helpers

Two helper scripts are provided in the project root:

- `python debug_qdrant.py` prints the configured collections and the current point count for `kb_docs_v1`. It honours the values saved in the app settings and optional `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` environment overrides.
- `python e2e_embed_upsert_test.py` recreates a scratch collection named `<collection>_debug`, uploads a single freshly generated embedding, verifies the count reaches 1, and performs a retrieval sanity check (limit 3). This is an easy way to confirm your OpenAI credentials, embedding dimension, and Qdrant endpoint all work together outside the GUI.
- `python search_api.py "your question" --pretty` runs the hybrid retrieval pipeline (keyword prefilter + dual named vectors) and prints the top-ranked chunks without invoking the chat model.
- `python verify.py` executes the end-to-end smoke checks described in the playbook: reference legal queries, self-hit tests for random articles, and empty-collection detection. The command exits with a non-zero status if any check fails.

All scripts log the endpoint in use so you can confirm that Cloud URLs are respected and that the pinned dependency versions are active.

## License

This project is provided as-is for demonstration purposes.
