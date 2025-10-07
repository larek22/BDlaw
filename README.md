# Vector KB Assistant

Vector KB Assistant is a desktop application built with PySide6 that lets you ingest local documents, store semantic embeddings in Qdrant, and ask grounded questions using OpenAI models. The application keeps all data-processing local except for calls to OpenAI for embeddings and language-model responses.

## Features

- Ingest PDF, DOCX, TXT, and RTF files.
- Chunk documents with configurable size and overlap.
- Cache embeddings locally to reduce repeated costs.
- Store chunk metadata and vectors in a local Qdrant collection.
- Ask natural-language questions with answers grounded in retrieved chunks.
- View citations and highlighted source snippets for transparency.
- Responsive PySide6 GUI with background threads for long-running tasks.
- Built-in health checks for OpenAI and Qdrant connectivity.

## Getting Started

### Prerequisites

- Python 3.11
- Access to OpenAI models `text-embedding-3-large` and a chat-capable model. The default is `gpt-4o-mini`, and the app will automatically fall back to it if another configured model is denied.
- A running Qdrant instance (e.g., via Docker: `docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant`).

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
   * When targeting **Qdrant Cloud**, set the Qdrant URL to the HTTPS endpoint (for example, `https://<cluster-id>.cloud.qdrant.io`) and paste the API key provided by Qdrant. The app will require the key for HTTPS endpoints and automatically log which host is used.
2. Switch to the **Ingest** tab, add documents, choose whether to rebuild the collection, and click **Create Vector DB**. Progress appears in the log pane.
3. After ingestion, go to the **Ask** tab, enter a question, and click **Search & Answer**. Answers include citations, and sources display highlighted query terms.

## Troubleshooting

- **OpenAI model access errors**: If you see a permission error when asking a question, the app falls back to `gpt-4o-mini`. If that also fails, choose a chat model available to your OpenAI account in the Settings tab and retry.

## Tests

Run the test suite with:

```bash
pytest
```

## Logging

Application logs are written to `~/.vector_kb/app.log` and streamed into the GUI log view during ingestion and queries.

## Qdrant connectivity check

To quickly verify the configured Qdrant endpoint and collection state outside of the GUI, run:

```bash
python debug_qdrant.py
```

The helper prints the available collections and the current point count for the configured collection, making it easy to confirm that cloud ingestions succeeded.

## License

This project is provided as-is for demonstration purposes.
