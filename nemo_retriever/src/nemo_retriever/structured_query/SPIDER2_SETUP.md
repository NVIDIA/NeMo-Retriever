# Spider2-lite + DuckDB Setup Guide

This guide walks you through loading the [Spider2-lite](https://github.com/xlang-ai/Spider2/tree/main/spider2-lite) benchmark databases into a local DuckDB file and running SQL queries against them.

Spider2-lite contains **30 databases** (Airlines, Baseball, Chinook, etc.), each stored as a folder of JSON files. The setup script loads them into a single `spider2.duckdb` file with **one schema per database**, so you query like:

```python
engine.execute("SELECT * FROM Airlines.flights LIMIT 5")
```

SQL generation (NL → SQL) is handled by your own LLM — this module provides the data loading and execution layer via `DuckDBEngine`.

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Git

### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 1 — Clone this repo

```bash
git clone https://github.com/NVIDIA/NeMo-Retriever.git
cd NeMo-Retriever
```

---

## 2 — Create a Python 3.12 environment and install

```bash
uv venv --python 3.12
source .venv/bin/activate   # macOS / Linux
```

Install the package without heavy ML dependencies (only `duckdb` is needed for setup):

```bash
uv pip install -e nemo_retriever/ --no-deps
uv pip install duckdb
```

---

## 3 — Run the one-time setup script

```bash
python3 nemo_retriever/src/nemo_retriever/structured_query/setup_spider2.py
```

This script will:
1. **Clone Spider2** from GitHub into `~/spider2` (shallow clone)
2. **Load all 30 databases** from `spider2-lite/resource/databases/sqlite/` into `spider2.duckdb`
3. **Print a summary** of every schema created

### Spider2 already cloned?

```bash
python3 nemo_retriever/src/nemo_retriever/structured_query/setup_spider2.py --skip-clone
```

### Custom paths

```bash
python3 nemo_retriever/src/nemo_retriever/structured_query/setup_spider2.py \
    --spider2-dir ~/projects/spider2 \
    --db ~/data/spider2.duckdb
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--spider2-dir` | `~/spider2` | Root of the Spider2 repository |
| `--db` | `./spider2.duckdb` | DuckDB database file to create or update |
| `--overwrite` | off | Drop and recreate schemas that already exist |
| `--skip-clone` | off | Skip the git clone step |

---

## 4 — Verify the data loaded

```bash
# List all schemas (one per Spider2-lite database)
retriever structured-query list-tables --database ./spider2.duckdb

# Show columns for a specific schema
retriever structured-query list-tables --database ./spider2.duckdb --schema
```

Or directly in Python:

```python
from nemo_retriever.structured_query.duckdb_engine import DuckDBEngine

engine = DuckDBEngine(database="./spider2.duckdb")
print(engine.list_schemas())          # ['Airlines', 'Baseball', 'chinook', ...]
print(engine.schema_tables("Airlines"))  # ['flights', 'airports_data', ...]
engine.close()
```

---

## 5 — Query the database

Each Spider2-lite database is a schema. Reference tables as `<Schema>.<table>`:

```python
from nemo_retriever.structured_query.duckdb_engine import DuckDBEngine

engine = DuckDBEngine(database="./spider2.duckdb")

# Direct SQL
rows = engine.execute("SELECT * FROM Airlines.flights LIMIT 5")
print(rows)

# With your own LLM generating the SQL
sql = your_llm_call(
    question="How many flights were delayed?",
    schema_context="Airlines database with tables: flights, airports_data, bookings, ..."
)
rows = engine.execute(sql)
print(rows)

engine.close()
```

### Available databases

| Schema name | Example tables |
|---|---|
| `Airlines` | `flights`, `airports_data`, `bookings`, `tickets` |
| `Baseball` | (baseball stats tables) |
| `chinook` | `albums`, `artists`, `customers`, `tracks` |
| `SQLITE_SAKILA` | `film`, `actor`, `rental`, `customer` |
| … 26 more | — |

---

## 6 — Run the Spider2-lite benchmark evaluation (optional)

Spider2-lite tasks are in `~/spider2/spider2-lite/spider2-lite.jsonl`. Each task has a `question` and `db` field:

```json
{"instance_id": "local001", "db": "Airlines", "question": "How many flights were delayed?"}
```

Use Spider2's evaluator to score predictions:

```bash
cd ~/spider2/spider2-lite/evaluation_suite
python3 evaluate.py --predictions ./results.json
```

---

## Day-to-day workflow (after first setup)

```bash
source .venv/bin/activate
```

Then query via Python using `DuckDBEngine` as shown in Step 5.

---

## Updating Spider2 data

```bash
cd ~/spider2 && git pull
python3 nemo_retriever/src/nemo_retriever/structured_query/setup_spider2.py \
    --skip-clone --overwrite
```

---

## Troubleshooting

**`zsh: command not found: python`**
Use `python3` on macOS.

**`Could not import nemo_retriever`**
```bash
uv pip install -e nemo_retriever/ --no-deps
uv pip install duckdb
```

**`Python>=3.12` error during install**
Create a 3.12 venv first:
```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

**`torch` platform error during `uv sync`**
Don't use `uv sync` for this setup — use `--no-deps` as shown in Step 2.

**`spider2-lite directory not found`**
Check the Spider2 repo structure:
```bash
ls ~/spider2/
```
The script expects `~/spider2/spider2-lite/resource/databases/sqlite/`. Pass `--spider2-dir` if Spider2 is elsewhere.
