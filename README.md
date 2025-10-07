# Vector KB Assistant

Vector KB Assistant is a desktop application built with PySide6 that lets you ingest local documents, store semantic embeddings in Qdrant, and ask grounded questions using OpenAI models. The application keeps all data-processing local except for calls to OpenAI for embeddings and language-model responses.

## Features

- Ingest PDF, DOCX, TXT, and RTF files.
- Chunk documents with configurable size and overlap.
- Cache embeddings locally to reduce repeated costs.
- Store chunk metadata and vectors in a Qdrant collection (local or Cloud) with deterministic point IDs.
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
   * When targeting **Qdrant Cloud**, set the Qdrant URL to the HTTPS endpoint (for example, `https://<cluster-id>.cloud.qdrant.io`) and paste the API key provided by Qdrant. The app enforces the API key for HTTPS endpoints, refuses to send it to `http://localhost`, and logs the exact host that will be used (`Using Qdrant endpoint: …`).
   * The collection is automatically created (or recreated) with the correct vector dimension for the configured embedding model. If you switch models, re-run ingestion with **Rebuild Collection** enabled so the schema matches the new embedding size.
2. Switch to the **Ingest** tab, add documents, choose whether to rebuild the collection, and click **Create Vector DB**. Progress appears in the log pane.
3. After ingestion, go to the **Ask** tab, enter a question, and click **Search & Answer**. Answers include citations, and sources display highlighted query terms.

## Troubleshooting

- **OpenAI model access errors**: On the first question the app probes the configured chat model and two fallbacks (`gpt-4.1-mini`, `gpt-4o-mini`). If none are accessible you will see a clear error asking you to change the Chat Model in Settings.
- **Qdrant collection stays empty**: Confirm the ingestion log includes `[UPSERT]` lines with the expected batch sizes and `dim`. Afterwards the log prints `[QDRANT] total points now: <count>`; if the count does not rise, re-check the configured endpoint and API key.
- **Re-ingesting adds duplicates**: Point IDs are deterministic. If counts climb unexpectedly, enable **Rebuild Collection** to clear stale data and rerun ingestion.

## Tests

Run the test suite with:

```bash
pytest
```

## Logging

Application logs are written to `~/.vector_kb/app.log` and streamed into the GUI log view during ingestion and queries.

## Qdrant verification helpers

Two helper scripts are provided in the project root:

- `python debug_qdrant.py` prints the configured collections and the current point count for `kb_docs_v1`. It honours the values saved in the app settings and optional `QDRANT_URL`, `QDRANT_API_KEY`, and `QDRANT_COLLECTION` environment overrides.
- `python e2e_embed_upsert_test.py` recreates a scratch collection named `<collection>_debug`, uploads a single freshly generated embedding, verifies the count reaches 1, and performs a retrieval sanity check (limit 3). This is an easy way to confirm your OpenAI credentials, embedding dimension, and Qdrant endpoint all work together outside the GUI.

Both scripts log the endpoint in use so you can confirm that Cloud URLs are respected.

## License

This project is provided as-is for demonstration purposes.
