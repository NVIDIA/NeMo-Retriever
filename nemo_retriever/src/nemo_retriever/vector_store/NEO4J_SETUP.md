# Neo4j Setup Guide

This guide walks you through running Neo4j locally via Docker and ingesting embeddings into it using the `Neo4jVDB` operator.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Python 3.12+

---

## 1 — Clone this repo

```bash
git clone https://github.com/NVIDIA/NeMo-Retriever.git
cd NeMo-Retriever
```

---

## 2 — Configure credentials

Copy the example env file and set your values:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test
```

> **Note:** `.env` is gitignored — never commit it. `.env.example` is committed as a template.

> **Docker vs host:** Use `bolt://localhost:7687` when running Python on your host machine.
> Use `bolt://neo4j:7687` (Docker service name) when running inside the Docker network.

---

## 3 — Install the client package

```bash
uv venv --python 3.12
source .venv/bin/activate   # macOS / Linux

uv pip install -e client/ --no-deps
uv pip install "neo4j>=5.0"
```

---

## 4 — Start Neo4j

Docker Compose reads credentials from `.env` automatically:

```bash
docker compose --profile graph up -d neo4j
```

Wait ~30 seconds for the container to become healthy, then verify:

```bash
docker compose ps neo4j
```

You should see `healthy` in the status column.

### Access points

| Interface | URL |
|---|---|
| Browser UI | http://localhost:7474 |
| Bolt (Python) | `bolt://localhost:7687` |

Credentials come from your `.env` file (`NEO4J_USERNAME` / `NEO4J_PASSWORD`).

---

## 5 — Verify the connection

Open http://localhost:7474 in your browser, log in with the credentials from your `.env`, and run:

```cypher
RETURN 1
```

Or verify from Python (credentials are read from `NEO4J_*` env vars automatically):

```python
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
)
with driver.session() as session:
    result = session.run("RETURN 'connected' AS status")
    print(result.single()["status"])   # connected
driver.close()
```

---

## 6 — Ingest embeddings

`Neo4jVDB` reads `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` from the environment automatically — no need to pass credentials explicitly:

```python
from nv_ingest_client.util.vdb.neo4j import Neo4jVDB

# Credentials come from .env / environment variables
vdb = Neo4jVDB(
    index_name="nv-ingest",
    node_label="Document",
    dense_dim=2048,          # match your embedding model dimension
    similarity_function="cosine",
)

# Create the vector index (idempotent — safe to call multiple times)
vdb.create_index()

# Ingest NV-Ingest pipeline results
# `results` is the output of the NV-Ingest pipeline (list of record-sets)
vdb.run(results)
```

Or use the high-level store helper (reads `*.text_embeddings.json` files):

```python
from pathlib import Path
from nemo_retriever.vector_store.neo4j_store import Neo4jConfig, write_text_embeddings_dir_to_neo4j

# Neo4jConfig also reads from NEO4J_* env vars by default
cfg = Neo4jConfig(
    index_name="nv-ingest",
    dense_dim=2048,
)

info = write_text_embeddings_dir_to_neo4j(
    Path("./my_embeddings_dir"),
    cfg=cfg,
)
print(info)
```

---

## 6 — Query (vector similarity search)

```python
results = vdb.retrieval(
    queries=["What is NVIDIA NIM?"],
    top_k=5,
    embedding_endpoint="http://localhost:8012/v1",
    model_name="nvidia/llama-nemotron-embed-1b-v2",
)

for hit in results[0]:
    print(hit["entity"]["text"])
    print(hit["score"])
```

---

## Day-to-day workflow

```bash
# Start Neo4j
docker compose --profile graph up -d neo4j

# Stop Neo4j (data is preserved in the neo4j_data volume)
docker compose --profile graph down neo4j

# Wipe all data and start fresh
docker compose --profile graph down neo4j -v
```

---

## Updating the vector index

Re-running `vdb.run(results)` is safe — nodes are `MERGE`'d on `(source_id, page_number)` so existing nodes are updated rather than duplicated.

To drop and recreate the index from scratch:

```python
vdb.create_index(recreate=True)
vdb.write_to_index(results)
```

---

## Troubleshooting

**`docker compose ps neo4j` shows `unhealthy`**
Give it more time (up to 60s on first run while Neo4j initialises). Check logs:
```bash
docker compose logs neo4j
```

**`ServiceUnavailable: Failed to establish connection`**
Make sure the container is running and port 7687 is not blocked:
```bash
docker compose ps neo4j
```

**`neo4j` package not found**
```bash
uv pip install "neo4j>=5.0"
```

**Vector index creation fails**
Neo4j native vector indexes require **Neo4j 5.11+**. The Docker image used (`neo4j:5.26`) satisfies this. If you're connecting to an older instance, upgrade it.

**Password mismatch**
Credentials are set in `.env`. Make sure `NEO4J_PASSWORD` in `.env` matches what Docker Compose used when the container was first created. If you changed it, recreate the container:
```bash
docker compose --profile graph down neo4j -v
docker compose --profile graph up -d neo4j
```
