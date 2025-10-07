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
- Access to OpenAI models `text-embedding-3-large` and `gpt-4.1-nano`.
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
2. Switch to the **Ingest** tab, add documents, choose whether to rebuild the collection, and click **Create Vector DB**. Progress appears in the log pane.
3. After ingestion, go to the **Ask** tab, enter a question, and click **Search & Answer**. Answers include citations, and sources display highlighted query terms.

## Tests

Run the test suite with:

```bash
pytest
```

## Logging

Application logs are written to `~/.vector_kb/app.log` and streamed into the GUI log view during ingestion and queries.

## License

This project is provided as-is for demonstration purposes.
