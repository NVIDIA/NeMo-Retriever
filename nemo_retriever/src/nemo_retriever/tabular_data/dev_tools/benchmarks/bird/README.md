# BIRD × DuckDB Setup Guide

This guide walks you through downloading the [BIRD](https://bird-bench.github.io/) benchmark databases into a local DuckDB file and running SQL queries against them.

BIRD contains **95 databases** (california_schools, financial, etc.), each stored as a real SQLite file with **full data** (not samples). The setup script loads them into a single `bird.duckdb` file with **one schema per database**, so you query like:

```python
conn.execute("SELECT * FROM california_schools.schools LIMIT 5")
```

> Unlike Spider2-lite (which ships only 5 sample rows per table as JSON), BIRD ships complete `.sqlite` files — you get the full dataset.

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- ~5 GB disk space (mini-dev) or ~33 GB (full dev)

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

Install the package without heavy ML dependencies:

```bash
uv pip install -e nemo_retriever
uv pip install duckdb
```

---

## 3 — Run the one-time setup script

### Mini-dev (recommended for first-time use, ~780 instances, smaller download)

```bash
python nemo_retriever/tabular-dev-tools/benchmarks/bird/setup.py --split mini-dev
```

### Full dev set (~1,534 instances, ~33 GB)

```bash
python nemo_retriever/tabular-dev-tools/benchmarks/bird/setup.py --split dev
```

This script will:
1. **Download** the BIRD zip from `bird-bench.oss-cn-beijing.aliyuncs.com` into `~/bird/` — skipped automatically if the data is already present
2. **Extract** the zip and locate the `dev_databases/` folder
3. **Load all databases** from SQLite files into `bird.duckdb` with full data
4. **Print a summary** of every schema created

### Custom paths

```bash
python nemo_retriever/tabular-dev-tools/benchmarks/bird/setup.py \
    --split mini-dev \
    --bird-dir ~/projects/bird \
    --db ~/data/bird.duckdb
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--split` | `mini-dev` | Which split to download: `mini-dev` or `dev` |
| `--bird-dir` | `~/bird` | Directory where BIRD data is stored (or downloaded to) |
| `--db` | `./bird.duckdb` | DuckDB database file to create or update |
| `--overwrite` | off | Drop and recreate schemas that already exist |

---

## 4 — Verify the data loaded

```python
import duckdb

conn = duckdb.connect("./bird.duckdb", read_only=True)

# List all schemas (one per BIRD database)
schemas = conn.execute("""
    SELECT schema_name FROM information_schema.schemata
    WHERE schema_name NOT IN ('main', 'information_schema', 'pg_catalog')
    ORDER BY schema_name
""").fetchall()
print([s[0] for s in schemas])

# List tables in a schema
tables = conn.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'california_schools'
""").fetchall()
print([t[0] for t in tables])

conn.close()
```

---

## 5 — Query the database

Each BIRD database is a schema. Reference tables as `<schema>.<table>`:

```python
import duckdb

conn = duckdb.connect("./bird.duckdb", read_only=True)

# Direct SQL
rows = conn.execute("SELECT * FROM california_schools.schools LIMIT 5").fetchall()
print(rows)

# Count rows
count = conn.execute("SELECT COUNT(*) FROM financial.account").fetchone()[0]
print(f"financial.account has {count} rows")

conn.close()
```

### Available databases (mini-dev subset)

| Schema | Domain |
|---|---|
| `california_schools` | Education |
| `financial` | Finance |
| `card_games` | Entertainment |
| `european_football_2` | Sports |
| `formula_1` | Sports |
| `toxicology` | Science |
| … and more | — |

---

## 6 — Run the BIRD evaluation (optional)

BIRD tasks are stored in `mini_dev_sqlite.json`. Each entry has a `db_id`, `question`, `evidence`, and gold `SQL`:

```json
{
  "db_id": "california_schools",
  "question": "What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
  "evidence": "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`",
  "SQL": "SELECT ..."
}
```

Use the BIRD evaluator from the [mini_dev repo](https://github.com/bird-bench/mini_dev) to score predictions.

---

## Day-to-day workflow (after first setup)

```bash
source .venv/bin/activate
```

Then query via Python using DuckDB directly, as shown in Step 5.

---

## Re-loading / updating

If you want to refresh all schemas from the SQLite sources:

```bash
python nemo_retriever/tabular-dev-tools/benchmarks/bird/setup.py --overwrite
```

---

## Comparison with Spider2-lite

| | Spider2-lite | BIRD |
|---|---|---|
| Source format | JSON (sample rows) | SQLite files |
| Rows per table | 5 (samples only) | Full data |
| Download | `git clone` | Zip download |
| Schemas | 30 | 95 (dev) / ~11 (mini-dev) |
| Setup script | `benchmarks/spider2/setup.py` | `benchmarks/bird/setup.py` |
| DuckDB file | `spider2.duckdb` | `bird.duckdb` |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'duckdb'`**
```bash
uv pip install duckdb
```

**Download fails / times out**
The BIRD zip files are hosted on Alibaba Cloud OSS. If the download is slow, you can manually download from [bird-bench.github.io](https://bird-bench.github.io/) and place the zip in `~/bird/`, then re-run the script (it will find and extract the zip automatically).

**`Could not locate 'dev_databases/' after extraction`**
The zip layout may have changed. Unzip manually and pass `--bird-dir` pointing to the folder that contains `dev_databases/`:
```bash
python nemo_retriever/tabular-dev-tools/benchmarks/bird/setup.py --bird-dir ~/bird/mini_dev_data
```

**`Python>=3.12` error during install**
```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```
