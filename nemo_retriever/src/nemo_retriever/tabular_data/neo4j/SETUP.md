# Neo4j Setup Guide

This guide walks you through running Neo4j locally via Docker and using the relational_db Neo4j connection from `nemo_retriever.relational_db.neo4j_connection`.

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
> Use `bolt://neo4j:7687` (Docker service name) when running inside the Docker network.

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
docker compose --profile graph up -d neo4j

# Stop Neo4j (data is preserved in the neo4j_data volume)
docker compose --profile graph down neo4j

# Wipe all data and start fresh
docker compose --profile graph down neo4j -v
```

---

## Troubleshooting

**`docker compose ps neo4j` shows `unhealthy`**  
Give it more time (up to 60s on first run). Check logs: `docker compose logs neo4j`

**`ServiceUnavailable: Failed to establish connection`**  
Ensure the container is running and port 7687 is not blocked.

**`neo4j` package not found**  
`uv pip install "neo4j>=5.0"`

**Vector index creation fails**  
Neo4j native vector indexes require **Neo4j 5.11+**. The Docker image used (`neo4j:5.26`) satisfies this.

**Password mismatch**  
Recreate the container after changing `.env`: `docker compose --profile graph down neo4j -v && docker compose --profile graph up -d neo4j`



# RUN WITH APOC!

docker run \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/liav_is_my_king \
  -e NEO4JLABS_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
  -e NEO4J_dbms_security_procedures_allowlist='apoc.*' \
  neo4j:5.26
