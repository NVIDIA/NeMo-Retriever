# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite history database for tracking nemo_retriever harness benchmark results."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NEMO_RETRIEVER_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = NEMO_RETRIEVER_ROOT / "harness" / "history.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    git_commit TEXT,
    dataset TEXT NOT NULL,
    preset TEXT,
    success INTEGER,
    return_code INTEGER,
    failure_reason TEXT,
    pages INTEGER,
    ingest_secs REAL,
    pages_per_sec REAL,
    recall_5 REAL,
    files INTEGER,
    tags TEXT,
    artifact_dir TEXT,
    raw_json TEXT,
    hostname TEXT,
    gpu_type TEXT,
    trigger_source TEXT,
    schedule_id INTEGER
);
"""

CREATE_RUNNERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS runners (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    hostname TEXT,
    url TEXT,
    gpu_type TEXT,
    gpu_count INTEGER,
    cpu_count INTEGER,
    memory_gb REAL,
    status TEXT DEFAULT 'offline',
    registered_at TEXT NOT NULL,
    last_heartbeat TEXT,
    tags TEXT,
    metadata TEXT
);
"""

CREATE_SCHEDULES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    dataset TEXT NOT NULL,
    preset TEXT,
    config TEXT,
    trigger_type TEXT NOT NULL,
    cron_expression TEXT,
    github_repo TEXT,
    github_branch TEXT,
    github_last_sha TEXT,
    min_gpu_count INTEGER,
    gpu_type_pattern TEXT,
    min_cpu_count INTEGER,
    min_memory_gb REAL,
    preferred_runner_id INTEGER,
    enabled INTEGER DEFAULT 1,
    tags TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    last_triggered_at TEXT
);
"""

CREATE_PRESETS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS presets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    config TEXT NOT NULL,
    tags TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
"""

CREATE_DATASETS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    query_csv TEXT,
    input_type TEXT DEFAULT 'pdf',
    recall_required INTEGER DEFAULT 0,
    recall_match_mode TEXT DEFAULT 'pdf_page',
    recall_adapter TEXT DEFAULT 'none',
    description TEXT,
    tags TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
"""

CREATE_DATASET_RUNNERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS dataset_runners (
    dataset_id INTEGER NOT NULL,
    runner_id INTEGER NOT NULL,
    PRIMARY KEY (dataset_id, runner_id),
    FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
    FOREIGN KEY (runner_id) REFERENCES runners(id) ON DELETE CASCADE
);
"""

CREATE_JOBS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    schedule_id INTEGER,
    trigger_source TEXT NOT NULL,
    dataset TEXT NOT NULL,
    preset TEXT,
    config TEXT,
    assigned_runner_id INTEGER,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    result TEXT,
    error TEXT,
    tags TEXT
);
"""

CREATE_ALERT_RULES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alert_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    metric TEXT NOT NULL,
    operator TEXT NOT NULL,
    threshold REAL NOT NULL,
    dataset_filter TEXT,
    preset_filter TEXT,
    enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT
);
"""

CREATE_ALERT_EVENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alert_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    metric TEXT NOT NULL,
    metric_value REAL,
    threshold REAL NOT NULL,
    operator TEXT NOT NULL,
    message TEXT,
    git_commit TEXT,
    dataset TEXT,
    preset TEXT,
    hostname TEXT,
    acknowledged INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (rule_id) REFERENCES alert_rules(id),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
"""

CREATE_PORTAL_SETTINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS portal_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN hostname TEXT",
    "ALTER TABLE runs ADD COLUMN gpu_type TEXT",
    "ALTER TABLE runs ADD COLUMN trigger_source TEXT",
    "ALTER TABLE runs ADD COLUMN schedule_id INTEGER",
    "ALTER TABLE runners ADD COLUMN cpu_count INTEGER",
    "ALTER TABLE runners ADD COLUMN memory_gb REAL",
    "ALTER TABLE jobs ADD COLUMN dataset_path TEXT",
    "ALTER TABLE jobs ADD COLUMN dataset_overrides TEXT",
    "ALTER TABLE jobs ADD COLUMN git_commit TEXT",
    "ALTER TABLE jobs ADD COLUMN git_ref TEXT",
    "ALTER TABLE jobs ADD COLUMN log_tail TEXT",
    "ALTER TABLE jobs ADD COLUMN rejected_runners TEXT",
    "ALTER TABLE runs ADD COLUMN ray_cluster_mode TEXT",
    "ALTER TABLE runs ADD COLUMN ray_dashboard_url TEXT",
    "ALTER TABLE runs ADD COLUMN recall_1 REAL",
    "ALTER TABLE runs ADD COLUMN recall_10 REAL",
    "ALTER TABLE runners ADD COLUMN heartbeat_interval INTEGER DEFAULT 30",
    "ALTER TABLE runners ADD COLUMN git_commit TEXT",
    "ALTER TABLE runners ADD COLUMN pending_update_commit TEXT",
    "ALTER TABLE runners ADD COLUMN ray_address TEXT",
    "ALTER TABLE runs ADD COLUMN execution_commit TEXT",
]

RUNNER_MISSED_HEARTBEATS_THRESHOLD = 4

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_runs_dataset_ts ON runs(dataset, timestamp);
"""


def _db_path() -> str:
    return os.environ.get("RETRIEVER_HARNESS_HISTORY_DB") or str(DEFAULT_DB_PATH)


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(CREATE_RUNNERS_TABLE_SQL)
    conn.execute(CREATE_PRESETS_TABLE_SQL)
    conn.execute(CREATE_DATASETS_TABLE_SQL)
    conn.execute(CREATE_DATASET_RUNNERS_TABLE_SQL)
    conn.execute(CREATE_SCHEDULES_TABLE_SQL)
    conn.execute(CREATE_JOBS_TABLE_SQL)
    conn.execute(CREATE_ALERT_RULES_TABLE_SQL)
    conn.execute(CREATE_ALERT_EVENTS_TABLE_SQL)
    conn.execute(CREATE_PORTAL_SETTINGS_TABLE_SQL)
    conn.execute(CREATE_INDEX_SQL)
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_summary_metric(result: dict[str, Any], key: str) -> Any:
    """Pull a metric from summary_metrics first, then fall back to metrics."""
    value = (result.get("summary_metrics") or {}).get(key)
    if value is None:
        value = (result.get("metrics") or {}).get(key)
    return value


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def record_run(
    result: dict[str, Any],
    artifact_dir: Path | str,
    db_path: str | None = None,
    *,
    trigger_source: str | None = None,
    schedule_id: int | None = None,
    execution_commit: str | None = None,
) -> int:
    """Insert a single run result into the history database. Returns the row id."""
    conn = _connect(db_path)
    try:
        tags_raw = result.get("tags")
        tags_json = json.dumps(tags_raw) if tags_raw else None
        run_meta = result.get("run_metadata") or {}

        row = {
            "timestamp": result.get("timestamp", ""),
            "git_commit": result.get("latest_commit"),
            "dataset": (result.get("test_config") or {}).get("dataset_label", "unknown"),
            "preset": (result.get("test_config") or {}).get("preset"),
            "success": 1 if result.get("success") else 0,
            "return_code": result.get("return_code"),
            "failure_reason": result.get("failure_reason"),
            "pages": _extract_summary_metric(result, "pages"),
            "ingest_secs": _extract_summary_metric(result, "ingest_secs"),
            "pages_per_sec": _extract_summary_metric(result, "pages_per_sec_ingest"),
            "recall_1": _extract_summary_metric(result, "recall_1"),
            "recall_5": _extract_summary_metric(result, "recall_5"),
            "recall_10": _extract_summary_metric(result, "recall_10"),
            "files": _extract_summary_metric(result, "files"),
            "tags": tags_json,
            "artifact_dir": str(artifact_dir),
            "raw_json": json.dumps(result),
            "hostname": run_meta.get("host"),
            "gpu_type": run_meta.get("gpu_type"),
            "trigger_source": trigger_source,
            "schedule_id": schedule_id,
            "ray_cluster_mode": run_meta.get("ray_cluster_mode"),
            "ray_dashboard_url": run_meta.get("ray_dashboard_url"),
            "execution_commit": execution_commit,
        }

        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        sql = f"INSERT INTO runs ({columns}) VALUES ({placeholders})"
        cursor = conn.execute(sql, list(row.values()))
        conn.commit()
        return cursor.lastrowid or 0
    finally:
        conn.close()


def get_runs(
    *,
    dataset: str | None = None,
    commit: str | None = None,
    limit: int = 200,
    offset: int = 0,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    """Return runs from the history DB, newest first."""
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = (
            "SELECT id, timestamp, git_commit, dataset, preset, success, return_code,"
            " failure_reason, pages, ingest_secs, pages_per_sec, recall_1, recall_5,"
            " recall_10, files, tags,"
            " artifact_dir, hostname, gpu_type, trigger_source, schedule_id,"
            " ray_cluster_mode, ray_dashboard_url, execution_commit"
            " FROM runs WHERE 1=1"
        )
        params: list[Any] = []

        if dataset:
            query += " AND dataset = ?"
            params.append(dataset)
        if commit:
            query += " AND git_commit LIKE ?"
            params.append(f"%{commit}%")

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("tags"):
                try:
                    d["tags"] = json.loads(d["tags"])
                except (json.JSONDecodeError, TypeError):
                    d["tags"] = []
            else:
                d["tags"] = []
            results.append(d)
        return results
    finally:
        conn.close()


def get_run_by_id(run_id: int, db_path: str | None = None) -> dict[str, Any] | None:
    """Return full run detail including raw_json."""
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        if d.get("tags"):
            try:
                d["tags"] = json.loads(d["tags"])
            except (json.JSONDecodeError, TypeError):
                d["tags"] = []
        if d.get("raw_json"):
            try:
                d["raw_json"] = json.loads(d["raw_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
    finally:
        conn.close()


def delete_run(run_id: int, db_path: str | None = None) -> bool:
    """Delete a run from the history database. Returns True if the row was deleted."""
    conn = _connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_runs_bulk(run_ids: list[int], db_path: str | None = None) -> int:
    """Delete multiple runs at once. Returns the number of rows deleted."""
    if not run_ids:
        return 0
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(run_ids))
        cursor = conn.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", run_ids)
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def get_datasets(db_path: str | None = None) -> list[str]:
    """Return distinct dataset names from the history DB (legacy: from runs table)."""
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT dataset FROM runs ORDER BY dataset").fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Preset CRUD
# ---------------------------------------------------------------------------


def _deserialize_preset_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("config"):
        try:
            d["config"] = json.loads(d["config"])
        except (json.JSONDecodeError, TypeError):
            pass
    if d.get("tags"):
        try:
            d["tags"] = json.loads(d["tags"])
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
    else:
        d["tags"] = []
    return d


def create_preset(data: dict[str, Any], db_path: str | None = None) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        now = _now_iso()
        config = data.get("config", {})
        config_json = json.dumps(config) if isinstance(config, dict) else config
        tags = json.dumps(data.get("tags") or [])
        conn.execute(
            "INSERT INTO presets (name, description, config, tags, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (data["name"], data.get("description"), config_json, tags, now, now),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return get_preset_by_id(row_id, db_path)  # type: ignore[return-value]
    finally:
        conn.close()


def get_all_presets(db_path: str | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM presets ORDER BY name").fetchall()
        return [_deserialize_preset_row(r) for r in rows]
    finally:
        conn.close()


def get_preset_by_id(preset_id: int, db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM presets WHERE id = ?", (preset_id,)).fetchone()
        return _deserialize_preset_row(row) if row else None
    finally:
        conn.close()


def update_preset(preset_id: int, data: dict[str, Any], db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        sets: list[str] = []
        vals: list[Any] = []
        if "name" in data:
            sets.append("name = ?")
            vals.append(data["name"])
        if "description" in data:
            sets.append("description = ?")
            vals.append(data["description"])
        if "config" in data:
            sets.append("config = ?")
            cfg = data["config"]
            vals.append(json.dumps(cfg) if isinstance(cfg, dict) else cfg)
        if "tags" in data:
            sets.append("tags = ?")
            vals.append(json.dumps(data["tags"] if isinstance(data["tags"], list) else []))
        if not sets:
            return get_preset_by_id(preset_id, db_path)
        sets.append("updated_at = ?")
        vals.append(_now_iso())
        vals.append(preset_id)
        conn.execute(f"UPDATE presets SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()
        return get_preset_by_id(preset_id, db_path)
    finally:
        conn.close()


def delete_preset(preset_id: int, db_path: str | None = None) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM presets WHERE id = ?", (preset_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_preset_names(db_path: str | None = None) -> list[str]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT name FROM presets ORDER BY name").fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------

_DATASET_FIELDS = (
    "name",
    "path",
    "query_csv",
    "input_type",
    "recall_required",
    "recall_match_mode",
    "recall_adapter",
    "description",
)


def _deserialize_dataset_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["recall_required"] = bool(d.get("recall_required"))
    if d.get("tags"):
        try:
            d["tags"] = json.loads(d["tags"])
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
    else:
        d["tags"] = []
    return d


def create_dataset(data: dict[str, Any], db_path: str | None = None) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        now = _now_iso()
        tags = json.dumps(data.get("tags") or [])
        conn.execute(
            "INSERT INTO datasets (name, path, query_csv, input_type, recall_required,"
            " recall_match_mode, recall_adapter, description, tags, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                data["name"],
                data["path"],
                data.get("query_csv") or None,
                data.get("input_type", "pdf"),
                1 if data.get("recall_required") else 0,
                data.get("recall_match_mode", "pdf_page"),
                data.get("recall_adapter", "none"),
                data.get("description") or None,
                tags,
                now,
                now,
            ),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return get_dataset_by_id(row_id, db_path)  # type: ignore[return-value]
    finally:
        conn.close()


def get_all_datasets(db_path: str | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM datasets ORDER BY name").fetchall()
        datasets = [_deserialize_dataset_row(r) for r in rows]
        for ds in datasets:
            rids = conn.execute("SELECT runner_id FROM dataset_runners WHERE dataset_id = ?", (ds["id"],)).fetchall()
            ds["runner_ids"] = [r[0] for r in rids]
        return datasets
    finally:
        conn.close()


def get_dataset_by_id(dataset_id: int, db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            return None
        ds = _deserialize_dataset_row(row)
        rids = conn.execute("SELECT runner_id FROM dataset_runners WHERE dataset_id = ?", (dataset_id,)).fetchall()
        ds["runner_ids"] = [r[0] for r in rids]
        return ds
    finally:
        conn.close()


def update_dataset(dataset_id: int, data: dict[str, Any], db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        sets: list[str] = []
        vals: list[Any] = []
        for field in _DATASET_FIELDS:
            if field in data:
                val = data[field]
                if field == "recall_required":
                    val = 1 if val else 0
                sets.append(f"{field} = ?")
                vals.append(val)
        if "tags" in data:
            sets.append("tags = ?")
            vals.append(json.dumps(data["tags"] if isinstance(data["tags"], list) else []))
        if not sets:
            return get_dataset_by_id(dataset_id, db_path)
        sets.append("updated_at = ?")
        vals.append(_now_iso())
        vals.append(dataset_id)
        conn.execute(f"UPDATE datasets SET {', '.join(sets)} WHERE id = ?", vals)
        conn.commit()
        return get_dataset_by_id(dataset_id, db_path)
    finally:
        conn.close()


def delete_dataset(dataset_id: int, db_path: str | None = None) -> bool:
    conn = _connect(db_path)
    try:
        cur = conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_dataset_names(db_path: str | None = None) -> list[str]:
    """Return all dataset names from the datasets table (managed datasets)."""
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT name FROM datasets ORDER BY name").fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def get_dataset_by_name(name: str, db_path: str | None = None) -> dict[str, Any] | None:
    """Look up a managed dataset by its unique name."""
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM datasets WHERE name = ?", (name,)).fetchone()
        if row is None:
            return None
        ds = _deserialize_dataset_row(row)
        rids = conn.execute("SELECT runner_id FROM dataset_runners WHERE dataset_id = ?", (ds["id"],)).fetchall()
        ds["runner_ids"] = [r[0] for r in rids]
        return ds
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dataset ↔ Runner associations
# ---------------------------------------------------------------------------


def set_dataset_runners(dataset_id: int, runner_ids: list[int], db_path: str | None = None) -> None:
    """Replace the set of runners associated with a dataset."""
    conn = _connect(db_path)
    try:
        conn.execute("DELETE FROM dataset_runners WHERE dataset_id = ?", (dataset_id,))
        for rid in runner_ids:
            conn.execute(
                "INSERT OR IGNORE INTO dataset_runners (dataset_id, runner_id) VALUES (?, ?)",
                (dataset_id, rid),
            )
        conn.commit()
    finally:
        conn.close()


def get_dataset_runner_ids(dataset_id: int, db_path: str | None = None) -> list[int]:
    """Return runner IDs associated with a dataset."""
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT runner_id FROM dataset_runners WHERE dataset_id = ?", (dataset_id,)).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def get_runner_ids_for_dataset_name(dataset_name: str, db_path: str | None = None) -> list[int] | None:
    """Return runner IDs that have *dataset_name*, or ``None`` if no restriction.

    If the dataset is not in the managed table, or if it exists but has no
    runners explicitly associated, returns ``None`` (no restriction — any
    runner may run it).  Only when runners are explicitly assigned does the
    returned list restrict which runners are eligible.
    """
    ds = get_dataset_by_name(dataset_name, db_path)
    if ds is None:
        return None
    ids = get_dataset_runner_ids(ds["id"], db_path)
    return ids if ids else None


def import_yaml_datasets(yaml_datasets: dict[str, dict[str, Any]], db_path: str | None = None) -> int:
    """Import datasets from the YAML config into the managed datasets table.

    Only datasets whose name does not already exist are inserted.
    Returns the number of newly imported datasets.
    """
    imported = 0
    for name, cfg in yaml_datasets.items():
        if not isinstance(cfg, dict):
            continue
        existing = get_dataset_by_name(name, db_path)
        if existing:
            continue
        data = {
            "name": name,
            "path": cfg.get("path", ""),
            "query_csv": cfg.get("query_csv"),
            "input_type": cfg.get("input_type", "pdf"),
            "recall_required": cfg.get("recall_required", False),
            "recall_match_mode": cfg.get("recall_match_mode", "pdf_page"),
            "recall_adapter": cfg.get("recall_adapter", "none"),
            "description": f"Imported from test_configs.yaml",
        }
        try:
            create_dataset(data, db_path)
            imported += 1
        except Exception:
            pass
    return imported


# ---------------------------------------------------------------------------
# Runner CRUD
# ---------------------------------------------------------------------------

_RUNNER_SCALAR_FIELDS = (
    "name",
    "hostname",
    "url",
    "gpu_type",
    "gpu_count",
    "cpu_count",
    "memory_gb",
    "status",
    "heartbeat_interval",
    "git_commit",
    "pending_update_commit",
    "ray_address",
)


def _deserialize_runner_row(d: dict[str, Any]) -> dict[str, Any]:
    for field in ("tags", "metadata"):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = [] if field == "tags" else None
        elif field == "tags":
            d[field] = []
    return d


def register_runner(data: dict[str, Any], db_path: str | None = None) -> dict[str, Any]:
    """Register a runner, reusing an existing record if one matches by hostname.

    If a runner with the same ``hostname`` already exists, its metadata is
    updated and the existing record (with its original ID) is returned.  This
    makes registration idempotent so runners survive portal restarts without
    creating duplicate entries.
    """
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        now = _now_iso()
        hostname = data.get("hostname")

        existing = None
        if hostname:
            existing = conn.execute("SELECT * FROM runners WHERE hostname = ?", (hostname,)).fetchone()

        if existing:
            runner_id = existing["id"]
            ex = dict(existing)
            new_git = data.get("git_commit")
            conn.execute(
                "UPDATE runners SET name=?, url=?, gpu_type=?, gpu_count=?, cpu_count=?,"
                " memory_gb=?, status=?, last_heartbeat=?, tags=?, metadata=?,"
                " heartbeat_interval=?, git_commit=?, ray_address=? WHERE id=?",
                (
                    data.get("name", ex.get("name")),
                    data.get("url", ex.get("url")),
                    data.get("gpu_type", ex.get("gpu_type")),
                    data.get("gpu_count", ex.get("gpu_count")),
                    data.get("cpu_count", ex.get("cpu_count")),
                    data.get("memory_gb", ex.get("memory_gb")),
                    data.get("status", "online"),
                    now,
                    json.dumps(data["tags"]) if data.get("tags") else ex.get("tags"),
                    json.dumps(data["metadata"]) if data.get("metadata") else ex.get("metadata"),
                    data.get("heartbeat_interval", 30),
                    new_git if new_git else ex.get("git_commit"),
                    data.get("ray_address") if "ray_address" in data else ex.get("ray_address"),
                    runner_id,
                ),
            )
            pending = ex.get("pending_update_commit")
            if new_git and pending and new_git.startswith(pending[:7]):
                conn.execute("UPDATE runners SET pending_update_commit = NULL WHERE id = ?", (runner_id,))
            conn.commit()
            row = conn.execute("SELECT * FROM runners WHERE id = ?", (runner_id,)).fetchone()
            return _deserialize_runner_row(dict(row))

        row_data = {
            "name": data.get("name", "unnamed"),
            "hostname": hostname,
            "url": data.get("url"),
            "gpu_type": data.get("gpu_type"),
            "gpu_count": data.get("gpu_count"),
            "cpu_count": data.get("cpu_count"),
            "memory_gb": data.get("memory_gb"),
            "status": data.get("status", "offline"),
            "registered_at": now,
            "last_heartbeat": now,
            "tags": json.dumps(data["tags"]) if data.get("tags") else None,
            "metadata": json.dumps(data["metadata"]) if data.get("metadata") else None,
            "heartbeat_interval": data.get("heartbeat_interval", 30),
            "git_commit": data.get("git_commit"),
            "ray_address": data.get("ray_address"),
        }
        columns = ", ".join(row_data.keys())
        placeholders = ", ".join("?" * len(row_data))
        cursor = conn.execute(f"INSERT INTO runners ({columns}) VALUES ({placeholders})", list(row_data.values()))
        conn.commit()
        result = dict(row_data)
        result["id"] = cursor.lastrowid or 0
        return _deserialize_runner_row(result)
    finally:
        conn.close()


def get_runners(db_path: str | None = None) -> list[dict[str, Any]]:
    """Return all runners ordered by name."""
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM runners ORDER BY name").fetchall()
        return [_deserialize_runner_row(dict(r)) for r in rows]
    finally:
        conn.close()


def get_runner_by_id(runner_id: int, db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM runners WHERE id = ?", (runner_id,)).fetchone()
        if row is None:
            return None
        return _deserialize_runner_row(dict(row))
    finally:
        conn.close()


def update_runner(runner_id: int, data: dict[str, Any], db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        updates: dict[str, Any] = {}
        for key in _RUNNER_SCALAR_FIELDS:
            if key in data:
                updates[key] = data[key]
        if "tags" in data:
            updates["tags"] = json.dumps(data["tags"]) if data["tags"] else None
        if "metadata" in data:
            updates["metadata"] = json.dumps(data["metadata"]) if data["metadata"] else None
        if not updates:
            return get_runner_by_id(runner_id, db_path)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [runner_id]
        conn.execute(f"UPDATE runners SET {set_clause} WHERE id = ?", values)
        conn.commit()
    finally:
        conn.close()
    return get_runner_by_id(runner_id, db_path)


def delete_runner(runner_id: int, db_path: str | None = None) -> bool:
    conn = _connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM runners WHERE id = ?", (runner_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def heartbeat_runner(runner_id: int, db_path: str | None = None) -> str | None:
    """Update heartbeat timestamp. Returns current status, or None if runner not found.

    Preserves the ``paused`` status — heartbeats from a paused runner keep it
    paused rather than flipping it back to online.
    """
    now = _now_iso()
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT status FROM runners WHERE id = ?", (runner_id,)).fetchone()
        if row is None:
            return None
        current_status = row[0]
        new_status = current_status if current_status == "paused" else "online"
        conn.execute(
            "UPDATE runners SET last_heartbeat = ?, status = ? WHERE id = ?",
            (now, new_status, runner_id),
        )
        conn.commit()
        return new_status
    finally:
        conn.close()


def set_pending_update_all_runners(commit: str, db_path: str | None = None) -> int:
    """Set pending_update_commit on all online/paused runners. Returns count updated."""
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "UPDATE runners SET pending_update_commit = ? WHERE status IN ('online', 'paused')",
            (commit,),
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def clear_pending_update(runner_id: int, db_path: str | None = None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("UPDATE runners SET pending_update_commit = NULL WHERE id = ?", (runner_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Portal settings (key/value store)
# ---------------------------------------------------------------------------

_PORTAL_SETTINGS_DEFAULTS: dict[str, str] = {
    "run_code_ref": "nvidia/main",
}


def get_portal_setting(key: str, db_path: str | None = None) -> str | None:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT value FROM portal_settings WHERE key = ?", (key,)).fetchone()
        if row:
            return row[0]
        return _PORTAL_SETTINGS_DEFAULTS.get(key)
    finally:
        conn.close()


def set_portal_setting(key: str, value: str, db_path: str | None = None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO portal_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_portal_settings(db_path: str | None = None) -> dict[str, str]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT key, value FROM portal_settings").fetchall()
        result = dict(_PORTAL_SETTINGS_DEFAULTS)
        for k, v in rows:
            result[k] = v
        return result
    finally:
        conn.close()


def pause_runner(runner_id: int, db_path: str | None = None) -> bool:
    conn = _connect(db_path)
    try:
        cursor = conn.execute("UPDATE runners SET status = 'paused' WHERE id = ?", (runner_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def resume_runner(runner_id: int, db_path: str | None = None) -> bool:
    conn = _connect(db_path)
    try:
        cursor = conn.execute("UPDATE runners SET status = 'online' WHERE id = ?", (runner_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def mark_stale_runners_offline(db_path: str | None = None) -> list[dict[str, Any]]:
    """Mark runners as offline if they missed N heartbeats. Returns newly-offline runners."""
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM runners WHERE status IN ('online', 'paused') AND last_heartbeat IS NOT NULL"
        ).fetchall()

        now = datetime.now(timezone.utc)
        newly_offline: list[dict[str, Any]] = []

        for row in rows:
            r = dict(row)
            interval = r.get("heartbeat_interval") or 30
            timeout_secs = interval * RUNNER_MISSED_HEARTBEATS_THRESHOLD

            try:
                last_hb = datetime.fromisoformat(r["last_heartbeat"])
                if last_hb.tzinfo is None:
                    last_hb = last_hb.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            if (now - last_hb).total_seconds() > timeout_secs:
                conn.execute("UPDATE runners SET status = 'offline' WHERE id = ?", (r["id"],))
                newly_offline.append(_deserialize_runner_row(r))

        if newly_offline:
            conn.commit()
        return newly_offline
    finally:
        conn.close()


def create_system_alert_event(data: dict[str, Any], db_path: str | None = None) -> dict[str, Any]:
    """Create an alert event not tied to a specific run (system-level alert)."""
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO alert_events (rule_id,run_id,metric,metric_value,threshold,"
            "operator,message,git_commit,dataset,preset,hostname,acknowledged,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)",
            (
                data.get("rule_id", 0),
                data.get("run_id", 0),
                data.get("metric", "runner_status"),
                data.get("metric_value"),
                data.get("threshold", 0),
                data.get("operator", "system"),
                data.get("message"),
                data.get("git_commit"),
                data.get("dataset"),
                data.get("preset"),
                data.get("hostname"),
                _now_iso(),
            ),
        )
        conn.commit()
        event_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM alert_events WHERE id = ?", (event_id,)).fetchone()
        return dict(row) if row else {"id": event_id}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schedule CRUD
# ---------------------------------------------------------------------------

_SCHEDULE_SCALAR_FIELDS = (
    "name",
    "description",
    "dataset",
    "preset",
    "config",
    "trigger_type",
    "cron_expression",
    "github_repo",
    "github_branch",
    "github_last_sha",
    "min_gpu_count",
    "gpu_type_pattern",
    "min_cpu_count",
    "min_memory_gb",
    "preferred_runner_id",
    "enabled",
)


def _deserialize_schedule_row(d: dict[str, Any]) -> dict[str, Any]:
    if d.get("tags"):
        try:
            d["tags"] = json.loads(d["tags"])
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []
    else:
        d["tags"] = []
    d["enabled"] = bool(d.get("enabled", 0))
    return d


def create_schedule(data: dict[str, Any], db_path: str | None = None) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        now = _now_iso()
        row: dict[str, Any] = {k: data.get(k) for k in _SCHEDULE_SCALAR_FIELDS}
        row["tags"] = json.dumps(data["tags"]) if data.get("tags") else None
        row["created_at"] = now
        row["updated_at"] = now
        row.setdefault("enabled", 1)
        if isinstance(row["enabled"], bool):
            row["enabled"] = int(row["enabled"])

        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        cursor = conn.execute(f"INSERT INTO schedules ({columns}) VALUES ({placeholders})", list(row.values()))
        conn.commit()
        result = dict(row)
        result["id"] = cursor.lastrowid or 0
        return _deserialize_schedule_row(result)
    finally:
        conn.close()


def get_schedules(db_path: str | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM schedules ORDER BY name").fetchall()
        return [_deserialize_schedule_row(dict(r)) for r in rows]
    finally:
        conn.close()


def get_schedule_by_id(schedule_id: int, db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
        if row is None:
            return None
        return _deserialize_schedule_row(dict(row))
    finally:
        conn.close()


def update_schedule(schedule_id: int, data: dict[str, Any], db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        updates: dict[str, Any] = {}
        for key in _SCHEDULE_SCALAR_FIELDS:
            if key in data:
                val = data[key]
                if key == "enabled" and isinstance(val, bool):
                    val = int(val)
                updates[key] = val
        if "tags" in data:
            updates["tags"] = json.dumps(data["tags"]) if data["tags"] else None
        updates["updated_at"] = _now_iso()
        if not updates:
            return get_schedule_by_id(schedule_id, db_path)
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [schedule_id]
        conn.execute(f"UPDATE schedules SET {set_clause} WHERE id = ?", values)
        conn.commit()
    finally:
        conn.close()
    return get_schedule_by_id(schedule_id, db_path)


def delete_schedule(schedule_id: int, db_path: str | None = None) -> bool:
    conn = _connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def mark_schedule_triggered(schedule_id: int, db_path: str | None = None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE schedules SET last_triggered_at = ? WHERE id = ?",
            (_now_iso(), schedule_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_enabled_schedules(trigger_type: str | None = None, db_path: str | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM schedules WHERE enabled = 1"
        params: list[Any] = []
        if trigger_type:
            query += " AND trigger_type = ?"
            params.append(trigger_type)
        rows = conn.execute(query, params).fetchall()
        return [_deserialize_schedule_row(dict(r)) for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Persistent Jobs
# ---------------------------------------------------------------------------


def _deserialize_job_row(d: dict[str, Any]) -> dict[str, Any]:
    for field in ("tags",):
        if d.get(field):
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = []
        else:
            d[field] = []
    if d.get("result"):
        try:
            d["result"] = json.loads(d["result"])
        except (json.JSONDecodeError, TypeError):
            pass
    if d.get("dataset_overrides"):
        try:
            d["dataset_overrides"] = json.loads(d["dataset_overrides"])
        except (json.JSONDecodeError, TypeError):
            d["dataset_overrides"] = None
    if d.get("log_tail"):
        try:
            d["log_tail"] = json.loads(d["log_tail"])
        except (json.JSONDecodeError, TypeError):
            d["log_tail"] = []
    else:
        d["log_tail"] = []
    if d.get("rejected_runners"):
        try:
            d["rejected_runners"] = json.loads(d["rejected_runners"])
        except (json.JSONDecodeError, TypeError):
            d["rejected_runners"] = []
    else:
        d["rejected_runners"] = []
    return d


def create_job(data: dict[str, Any], db_path: str | None = None) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        job_id = data.get("id") or uuid.uuid4().hex[:12]
        overrides = data.get("dataset_overrides")
        overrides_json = json.dumps(overrides) if overrides else None
        row = {
            "id": job_id,
            "schedule_id": data.get("schedule_id"),
            "trigger_source": data.get("trigger_source", "manual"),
            "dataset": data["dataset"],
            "dataset_path": data.get("dataset_path"),
            "dataset_overrides": overrides_json,
            "preset": data.get("preset"),
            "config": data.get("config"),
            "assigned_runner_id": data.get("assigned_runner_id"),
            "status": data.get("status", "pending"),
            "git_commit": data.get("git_commit"),
            "git_ref": data.get("git_ref"),
            "created_at": _now_iso(),
            "started_at": None,
            "completed_at": None,
            "result": None,
            "error": None,
            "tags": json.dumps(data["tags"]) if data.get("tags") else None,
        }
        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        conn.execute(f"INSERT INTO jobs ({columns}) VALUES ({placeholders})", list(row.values()))
        conn.commit()
        return _deserialize_job_row(dict(row))
    finally:
        conn.close()


def get_jobs(
    *,
    status: str | None = None,
    limit: int = 200,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM jobs"
        params: list[Any] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_deserialize_job_row(dict(r)) for r in rows]
    finally:
        conn.close()


def get_job_by_id(job_id: str, db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return _deserialize_job_row(dict(row))
    finally:
        conn.close()


def runner_has_running_job(runner_id: int, db_path: str | None = None) -> bool:
    """Return True if the runner already has a job in 'running' status."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE assigned_runner_id = ? AND status = 'running' LIMIT 1",
            (runner_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_pending_jobs_for_runner(runner_id: int, db_path: str | None = None) -> list[dict[str, Any]]:
    """Return pending jobs for this runner, but only if it has no running job.

    Jobs whose ``rejected_runners`` list contains *runner_id* are excluded so
    that runners do not repeatedly pick up jobs they cannot execute (e.g. due
    to a missing dataset).
    """
    if runner_has_running_job(runner_id, db_path):
        return []
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status = 'pending' "
            "AND (assigned_runner_id = ? OR assigned_runner_id IS NULL) "
            "ORDER BY (assigned_runner_id IS NULL) ASC, created_at ASC",
            (runner_id,),
        ).fetchall()
        results = []
        runner_id_str = str(runner_id)
        for r in rows:
            job = _deserialize_job_row(dict(r))
            rejected = job.get("rejected_runners") or []
            if runner_id_str in [str(rid) for rid in rejected]:
                continue
            results.append(job)
        return results
    finally:
        conn.close()


def assign_job_to_runner(job_id: str, runner_id: int, db_path: str | None = None) -> None:
    """Assign an unassigned job to a specific runner."""
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE jobs SET assigned_runner_id = ? WHERE id = ? AND assigned_runner_id IS NULL",
            (runner_id, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def claim_job(job_id: str, db_path: str | None = None) -> bool:
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ? AND status = 'pending'",
            (_now_iso(), job_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def complete_job(
    job_id: str,
    *,
    success: bool,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    db_path: str | None = None,
) -> None:
    conn = _connect(db_path)
    try:
        status = "completed" if success else "failed"
        result_json = json.dumps(result) if result else None
        conn.execute(
            "UPDATE jobs SET status = ?, completed_at = ?, result = ?, error = ? WHERE id = ?",
            (status, _now_iso(), result_json, error, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_pending_jobs_for_schedule(schedule_id: int, db_path: str | None = None) -> list[dict[str, Any]]:
    """Return all pending jobs for a specific schedule, oldest first."""
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE schedule_id = ? AND status = 'pending' ORDER BY created_at ASC",
            (schedule_id,),
        ).fetchall()
        return [_deserialize_job_row(dict(r)) for r in rows]
    finally:
        conn.close()


def cancel_job(job_id: str, reason: str = "Cancelled due to backlog limit", db_path: str | None = None) -> bool:
    """Cancel a pending job. Returns True if the job was actually cancelled."""
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "UPDATE jobs SET status = 'cancelled', completed_at = ?, error = ? WHERE id = ? AND status = 'pending'",
            (_now_iso(), reason, job_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def request_job_cancel(job_id: str, db_path: str | None = None) -> bool:
    """Request cancellation of a pending or running job.

    Pending jobs are cancelled immediately.  Running jobs are transitioned to
    ``cancelling`` so the runner can pick up the signal on the next heartbeat.
    """
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return False
        status = row[0]
        if status == "pending":
            conn.execute(
                "UPDATE jobs SET status = 'cancelled', completed_at = ?, error = ? WHERE id = ?",
                (_now_iso(), "Cancelled by user", job_id),
            )
            conn.commit()
            return True
        if status == "running":
            conn.execute("UPDATE jobs SET status = 'cancelling' WHERE id = ?", (job_id,))
            conn.commit()
            return True
        return False
    finally:
        conn.close()


def force_delete_job(job_id: str, db_path: str | None = None) -> bool:
    """Permanently delete a job regardless of its current status."""
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT id FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
        return True
    finally:
        conn.close()


def update_job_log(job_id: str, log_tail: list[str], db_path: str | None = None) -> None:
    """Store the latest log tail for a running job."""
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE jobs SET log_tail = ? WHERE id = ?",
            (json.dumps(log_tail[-500:]), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_job_log(job_id: str, db_path: str | None = None) -> list[str]:
    """Return the stored log tail for a job."""
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT log_tail FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                pass
        return []
    finally:
        conn.close()


def update_job_status(job_id: str, status: str, error: str | None = None, db_path: str | None = None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("UPDATE jobs SET status = ?, error = ? WHERE id = ?", (status, error, job_id))
        conn.commit()
    finally:
        conn.close()


def reject_job_by_runner(
    job_id: str,
    runner_id: int,
    reason: str = "Runner cannot execute this job",
    db_path: str | None = None,
) -> bool:
    """Mark a job as rejected by a specific runner.

    The runner ID is appended to the ``rejected_runners`` JSON list and the
    ``assigned_runner_id`` is cleared so that another runner may pick up the
    job.  Returns ``True`` if the update was applied.
    """
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT rejected_runners FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return False
        existing: list[int] = []
        if row[0]:
            try:
                existing = json.loads(row[0])
            except (json.JSONDecodeError, TypeError):
                existing = []
        if runner_id not in existing:
            existing.append(runner_id)
        conn.execute(
            "UPDATE jobs SET rejected_runners = ?, assigned_runner_id = NULL WHERE id = ?",
            (json.dumps(existing), job_id),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def get_or_create_system_alert_rule(
    name: str,
    description: str | None = None,
    db_path: str | None = None,
) -> dict[str, Any]:
    """Return a system-level alert rule, creating it if it doesn't exist.

    System rules use ``metric='system'`` and are used for operational alerts
    (e.g. missing datasets) rather than run-metric threshold alerts.
    """
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM alert_rules WHERE name = ? AND metric = 'system'", (name,)).fetchone()
        if row:
            return _deserialize_alert_rule(row)
        now = _now_iso()
        conn.execute(
            "INSERT INTO alert_rules (name,description,metric,operator,threshold,enabled,created_at,updated_at)"
            " VALUES (?,?,'system','!=',0,1,?,?)",
            (name, description or name, now, now),
        )
        conn.commit()
        new_row = conn.execute("SELECT * FROM alert_rules WHERE name = ? AND metric = 'system'", (name,)).fetchone()
        return _deserialize_alert_rule(new_row)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Alert Rules
# ---------------------------------------------------------------------------

VALID_ALERT_METRICS = [
    "pages_per_sec",
    "recall_1",
    "recall_5",
    "recall_10",
    "ingest_secs",
    "pages",
    "files",
    "system",
]
VALID_ALERT_OPERATORS = ["<", "<=", ">", ">=", "==", "!="]


def _deserialize_alert_rule(row: sqlite3.Row | dict) -> dict[str, Any]:
    d = dict(row)
    d["enabled"] = bool(d.get("enabled"))
    return d


def create_alert_rule(data: dict[str, Any], db_path: str | None = None) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        now = _now_iso()
        conn.execute(
            "INSERT INTO alert_rules (name,description,metric,operator,threshold,"
            "dataset_filter,preset_filter,enabled,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                data["name"],
                data.get("description"),
                data["metric"],
                data["operator"],
                data["threshold"],
                data.get("dataset_filter"),
                data.get("preset_filter"),
                1 if data.get("enabled", True) else 0,
                now,
                now,
            ),
        )
        conn.commit()
        rule_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM alert_rules WHERE id = ?", (rule_id,)).fetchone()
        return _deserialize_alert_rule(row)
    finally:
        conn.close()


def get_alert_rules(db_path: str | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM alert_rules ORDER BY name").fetchall()
        return [_deserialize_alert_rule(r) for r in rows]
    finally:
        conn.close()


def get_alert_rule_by_id(rule_id: int, db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM alert_rules WHERE id = ?", (rule_id,)).fetchone()
        return _deserialize_alert_rule(row) if row else None
    finally:
        conn.close()


def get_enabled_alert_rules(db_path: str | None = None) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM alert_rules WHERE enabled = 1").fetchall()
        return [_deserialize_alert_rule(r) for r in rows]
    finally:
        conn.close()


def update_alert_rule(rule_id: int, data: dict[str, Any], db_path: str | None = None) -> dict[str, Any] | None:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        existing = conn.execute("SELECT * FROM alert_rules WHERE id = ?", (rule_id,)).fetchone()
        if not existing:
            return None
        fields = []
        values: list[Any] = []
        for key in ("name", "description", "metric", "operator", "threshold", "dataset_filter", "preset_filter"):
            if key in data:
                fields.append(f"{key} = ?")
                values.append(data[key])
        if "enabled" in data:
            fields.append("enabled = ?")
            values.append(1 if data["enabled"] else 0)
        if fields:
            fields.append("updated_at = ?")
            values.append(_now_iso())
            values.append(rule_id)
            conn.execute(f"UPDATE alert_rules SET {', '.join(fields)} WHERE id = ?", values)
            conn.commit()
        row = conn.execute("SELECT * FROM alert_rules WHERE id = ?", (rule_id,)).fetchone()
        return _deserialize_alert_rule(row)
    finally:
        conn.close()


def delete_alert_rule(rule_id: int, db_path: str | None = None) -> bool:
    conn = _connect(db_path)
    try:
        cursor = conn.execute("DELETE FROM alert_rules WHERE id = ?", (rule_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Alert Events
# ---------------------------------------------------------------------------


def _deserialize_alert_event(row: sqlite3.Row | dict) -> dict[str, Any]:
    d = dict(row)
    d["acknowledged"] = bool(d.get("acknowledged"))
    return d


def create_alert_event(data: dict[str, Any], db_path: str | None = None) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO alert_events (rule_id,run_id,metric,metric_value,threshold,"
            "operator,message,git_commit,dataset,preset,hostname,acknowledged,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,0,?)",
            (
                data["rule_id"],
                data["run_id"],
                data["metric"],
                data.get("metric_value"),
                data["threshold"],
                data["operator"],
                data.get("message"),
                data.get("git_commit"),
                data.get("dataset"),
                data.get("preset"),
                data.get("hostname"),
                _now_iso(),
            ),
        )
        conn.commit()
        event_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM alert_events WHERE id = ?", (event_id,)).fetchone()
        return _deserialize_alert_event(row)
    finally:
        conn.close()


def get_alert_events(
    limit: int = 200,
    rule_id: int | None = None,
    acknowledged: bool | None = None,
    db_path: str | None = None,
) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM alert_events"
        conditions: list[str] = []
        params: list[Any] = []
        if rule_id is not None:
            conditions.append("rule_id = ?")
            params.append(rule_id)
        if acknowledged is not None:
            conditions.append("acknowledged = ?")
            params.append(1 if acknowledged else 0)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_deserialize_alert_event(r) for r in rows]
    finally:
        conn.close()


def acknowledge_alert_event(event_id: int, db_path: str | None = None) -> bool:
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            "UPDATE alert_events SET acknowledged = 1 WHERE id = ?",
            (event_id,),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def acknowledge_all_alert_events(db_path: str | None = None) -> int:
    conn = _connect(db_path)
    try:
        cursor = conn.execute("UPDATE alert_events SET acknowledged = 1 WHERE acknowledged = 0")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def evaluate_alerts_for_run(run: dict[str, Any], db_path: str | None = None) -> list[dict[str, Any]]:
    """Check all enabled alert rules against a completed run. Returns created events."""
    rules = get_enabled_alert_rules(db_path)
    events: list[dict[str, Any]] = []
    run_id = run.get("id")
    if not run_id:
        return events

    import operator as op_mod

    ops = {
        "<": op_mod.lt,
        "<=": op_mod.le,
        ">": op_mod.gt,
        ">=": op_mod.ge,
        "==": op_mod.eq,
        "!=": op_mod.ne,
    }

    for rule in rules:
        if rule.get("dataset_filter") and rule["dataset_filter"] != run.get("dataset"):
            continue
        if rule.get("preset_filter") and rule["preset_filter"] != run.get("preset"):
            continue

        metric_key = rule["metric"]
        metric_value = run.get(metric_key)
        if metric_value is None:
            continue

        op_fn = ops.get(rule["operator"])
        if op_fn is None:
            continue

        threshold = rule["threshold"]
        if op_fn(metric_value, threshold):
            human_op = rule["operator"]
            metric_label = metric_key.replace("_", " ").title()
            msg = (
                f"{metric_label} is {metric_value:.4g} which violates rule "
                f'"{rule["name"]}" ({metric_key} {human_op} {threshold})'
            )
            event = create_alert_event(
                {
                    "rule_id": rule["id"],
                    "run_id": run_id,
                    "metric": metric_key,
                    "metric_value": metric_value,
                    "threshold": threshold,
                    "operator": rule["operator"],
                    "message": msg,
                    "git_commit": run.get("git_commit"),
                    "dataset": run.get("dataset"),
                    "preset": run.get("preset"),
                    "hostname": run.get("hostname"),
                },
                db_path,
            )
            events.append(event)

    return events


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def backfill_from_artifacts(artifacts_root: Path | None = None, db_path: str | None = None) -> int:
    """Scan artifact directories for results.json and import into the history DB.

    Returns the number of runs imported.
    """
    from nemo_retriever.harness.artifacts import DEFAULT_ARTIFACTS_ROOT

    root = artifacts_root or DEFAULT_ARTIFACTS_ROOT
    if not root.exists():
        return 0

    conn = _connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        existing_dirs = {
            row[0] for row in conn.execute("SELECT artifact_dir FROM runs WHERE artifact_dir IS NOT NULL").fetchall()
        }
    finally:
        conn.close()

    imported = 0
    for results_file in sorted(root.rglob("results.json")):
        artifact_dir = str(results_file.parent.resolve())
        if artifact_dir in existing_dirs:
            continue

        try:
            payload = json.loads(results_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if not isinstance(payload, dict) or "timestamp" not in payload:
            continue

        record_run(payload, artifact_dir, db_path=db_path)
        imported += 1

    return imported
