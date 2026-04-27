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

Run Neo4j in a container (credentials should match `.env`; example uses `neo4j` / `test`):

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test \
  -v neo4j_data:/data \
  neo4j:5.26
```

Wait ~30 seconds for the database to finish starting, then verify:

```bash
docker ps --filter name=neo4j
```

The container should stay in the `Up` state (Neo4j prints readiness in its logs).

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
# Stop Neo4j (data is preserved in the neo4j_data volume)
docker stop neo4j

# Start again
docker start neo4j

# Wipe all data and start fresh
docker rm -f neo4j
docker volume rm neo4j_data
# then re-run the docker run command from section 4
```

---

## Troubleshooting

**Neo4j is slow to accept connections on first start**  
Give it more time (up to 60s on first run). Check logs: `docker logs neo4j`

**`ServiceUnavailable: Failed to establish connection`**  
Ensure the container is running and port 7687 is not blocked.

**`neo4j` package not found**  
`uv pip install "neo4j>=5.0"`

**Vector index creation fails**  
Neo4j native vector indexes require **Neo4j 5.11+**. The Docker image used (`neo4j:5.26`) satisfies this.

**Password mismatch**  
Stop and remove the container and volume, then start again with `NEO4J_AUTH` matching your `.env` credentials.

### Optional: enable APOC

If you need APOC procedures, add the plugin and security flags to your `docker run` (see the [Neo4j Docker documentation](https://neo4j.com/docs/operations-manual/current/docker/introduction/)).
