# Neo4j Setup Guide

> **Warning — local Docker developer tooling.** The Compose commands in this guide run **Neo4j locally** for development only. This is **not** a supported production deployment path. For NeMo Retriever / NIM deployment, use **[Helm](../../../../helm/README.md)** and the **[NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/)**.

This guide walks you through running Neo4j locally with the feature-owned Compose helper and using the relational_db Neo4j connection from `nemo_retriever.relational_db.neo4j_connection`.

---

## Prerequisites

- [Docker](https://www.docker.com/get-docker/) installed and running
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
> Use a container DNS name such as `bolt://neo4j:7687` only when your client runs in the same Docker network.

---

## 3 — Install dependencies

```bash
uv venv --python 3.12
source .venv/bin/activate   # macOS / Linux

uv pip install -e nemo_retriever/  # or your package path
uv pip install "neo4j>=5.0"
```

---

## 4 — Start Neo4j

Start Neo4j with the local development Compose helper:

```bash
docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml up -d neo4j
```

Wait ~30 seconds for the container to start accepting connections, then verify:

```bash
docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml ps neo4j
```

You should see the `neo4j` container running.

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

Or verify from Python using the relational_db Neo4j connection:

```python
from nemo_retriever.relational_db.neo4j_connection import get_neo4j_conn

conn = get_neo4j_conn()
conn.verify_connectivity()
```

---


## Day-to-day workflow

```bash
# Start Neo4j
docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml up -d neo4j

# Stop Neo4j (data is preserved in the neo4j_data volume)
docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml down

# Wipe all data and start fresh
docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml down -v
```

---

## Troubleshooting

**`docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml ps neo4j` does not show a running container**
Give it more time (up to 60s on first run). Check logs: `docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml logs neo4j`

**`ServiceUnavailable: Failed to establish connection`**  
Ensure the container is running and port 7687 is not blocked.

**`neo4j` package not found**  
`uv pip install "neo4j>=5.0"`

**Vector index creation fails**  
Neo4j native vector indexes require **Neo4j 5.11+**. The Docker image used (`neo4j:5.26`) satisfies this.

**Password mismatch**  
Recreate the container after changing `.env`: `docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml down -v && docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml up -d neo4j`.
