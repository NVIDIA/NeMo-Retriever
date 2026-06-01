# Neo4j Setup Guide

> **Warning — local Docker developer tooling.** The Docker commands in this guide run **Neo4j locally** for development only. This is **not** a supported production deployment path. For NeMo Retriever / NIM deployment, use **[Helm](../../../../helm/README.md)** and the **[NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/)**.

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

Export the credentials from `.env`, then start Neo4j with Docker:

```bash
set -a
source .env
set +a

docker volume create neo4j_data
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH="${NEO4J_USERNAME:-neo4j}/${NEO4J_PASSWORD:-test}" \
  -v neo4j_data:/data \
  neo4j:5.26
```

Wait ~30 seconds for the container to start accepting connections, then verify:

```bash
docker ps --filter name=neo4j
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
docker start neo4j

# Stop Neo4j (data is preserved in the neo4j_data volume)
docker stop neo4j

# Wipe all data and start fresh
docker stop neo4j
docker rm neo4j
docker volume rm neo4j_data
```

---

## Troubleshooting

**`docker ps --filter name=neo4j` does not show a running container**
Give it more time (up to 60s on first run). Check logs: `docker logs neo4j`

**`ServiceUnavailable: Failed to establish connection`**  
Ensure the container is running and port 7687 is not blocked.

**`neo4j` package not found**  
`uv pip install "neo4j>=5.0"`

**Vector index creation fails**  
Neo4j native vector indexes require **Neo4j 5.11+**. The Docker image used (`neo4j:5.26`) satisfies this.

**Password mismatch**  
Recreate the container after changing `.env`: stop and remove the container, then rerun the `docker run` command above.

## Optional: run with APOC

```bash
set -a
source .env
set +a

docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH="${NEO4J_USERNAME:-neo4j}/${NEO4J_PASSWORD:-test}" \
  -e NEO4JLABS_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
  -e NEO4J_dbms_security_procedures_allowlist='apoc.*' \
  -v neo4j_data:/data \
  neo4j:5.26
```
