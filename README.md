# BDlaw

This repository contains a minimal ingestion harness that focuses on safe, repeatable
writes to a Qdrant vector store. The design follows a blue/green workflow so new
collections are provisioned separately and only promoted after downstream checks.

## Key Features

* **Deterministic point identifiers** – chunk payloads are hashed and converted to
  UUIDv5 identifiers to guarantee idempotent re-ingestion.
* **Dynamic batch sizing** – upsert batches are bounded by an estimated payload size
  and the configured point limit to avoid `WriteTimeout` errors with large requests.
* **Configurable transport** – gRPC can be enabled for Qdrant Cloud, along with custom
  connection/read/write timeouts and retry budgets.
* **Do-no-harm aliasing** – helper methods create new physical collections and swap
  aliases atomically after validation, leaving the previous collection untouched for
  instant rollbacks.

## Configuration

Runtime configuration is read from environment variables. The most important ones are:

| Variable | Default | Description |
| --- | --- | --- |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint (HTTP or HTTPS). |
| `QDRANT_USE_GRPC` | `true` | Prefer gRPC if the server exposes it. |
| `QDRANT_GRPC_PORT` | `6334` | gRPC port. |
| `QDRANT_CONNECT_TIMEOUT` | `30` | Connection timeout (seconds). |
| `QDRANT_READ_TIMEOUT` | `120` | Read timeout (seconds). |
| `QDRANT_WRITE_TIMEOUT` | `120` | Write timeout (seconds). |
| `INGEST_TARGET_BATCH_BYTES` | `800000` | Target maximum batch payload size in bytes. |
| `BATCH_POINTS_MAX` | `32` | Maximum points per batch. |
| `INGEST_MAX_RETRIES` | `5` | Retry attempts for timeout failures. |
| `QDRANT_ALIAS` | `GK_RF2` | Stable alias to swap after validation. |

## Usage

The `IngestService` class accepts pre-computed chunk information and streams it to a new
Qdrant collection:

```python
from app.ingest import Chunk, IngestService

service = IngestService()
collection = service.ingest(
    vector_dim=3072,
    chunks=[
        Chunk(
            doc_id="gkrf:part1:art1:v2018-05-23",
            text="Статья 1. Основные начала ...",
            vector=[0.0] * 3072,
            metadata={
                "hierarchy": {"part_no": 1, "chapter_no": 1, "article_no_int": 1},
                "law_meta": {"status": "active", "enact_date_int": 20180523},
            },
        )
    ],
)
print("New collection:", collection)
```

After external verification succeeds, call `SafeQdrantClient.finalize_collection()` to
point the alias at the new collection. This keeps the previous collection available for
immediate rollback.

## Testing

The repository ships with unit tests that cover the batching logic and deterministic ID
helpers:

```bash
pip install -r requirements.txt
PYTHONPATH=. pytest -q
```

---

This minimal foundation can be extended with document parsing, GPT-based auto-structuring,
and snapshot tooling as required by production ingest pipelines.
