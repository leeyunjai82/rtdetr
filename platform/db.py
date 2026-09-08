# Apache-2.0
"""SQLite for the platform: datasets, jobs, and one row per training epoch.

One file, no server to run. A single-box tool does not need Postgres, and
anything that outgrows this file wants a real queue anyway.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    images_dir TEXT NOT NULL,
    labels_dir TEXT NOT NULL,
    images INTEGER NOT NULL,
    labelled INTEGER NOT NULL,
    classes TEXT NOT NULL,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'train',
    dataset_id INTEGER NOT NULL REFERENCES datasets(id),
    model TEXT NOT NULL,
    epochs INTEGER,
    imgsz INTEGER,
    batch INTEGER,
    freeze TEXT,
    device TEXT,
    conf REAL,
    resume_of INTEGER,
    source TEXT,
    status TEXT NOT NULL,
    detail TEXT,
    progress REAL DEFAULT 0,
    run_dir TEXT,
    best_map REAL,
    created REAL NOT NULL,
    started REAL,
    finished REAL
);
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    job_id INTEGER,
    classes TEXT NOT NULL,
    note TEXT,
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS epochs (
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    epoch INTEGER NOT NULL,
    loss REAL,
    map50_95 REAL,
    seconds REAL,
    PRIMARY KEY (job_id, epoch)
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def execute(self, sql: str, params=()) -> int:
        with self._lock, self.connect() as conn:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor.lastrowid

    def query(self, sql: str, params=()) -> list[dict]:
        with self._lock, self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params)]

    def one(self, sql: str, params=()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # -- the writes the app makes ------------------------------------------

    def add_dataset(self, **fields) -> int:
        fields["classes"] = json.dumps(fields.get("classes", []), ensure_ascii=False)
        fields["created"] = time.time()
        columns = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        return self.execute(
            f"INSERT INTO datasets ({columns}) VALUES ({marks})", tuple(fields.values())
        )

    def update_dataset(self, dataset_id: int, **fields) -> None:
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.execute(
            f"UPDATE datasets SET {assignments} WHERE id = ?", (*fields.values(), dataset_id)
        )

    def add_model(self, **fields) -> int:
        fields["classes"] = json.dumps(fields.get("classes", []), ensure_ascii=False)
        fields["created"] = time.time()
        columns = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        return self.execute(
            f"INSERT INTO models ({columns}) VALUES ({marks})", tuple(fields.values())
        )

    def add_job(self, **fields) -> int:
        fields.setdefault("status", "queued")
        fields["created"] = time.time()
        columns = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        return self.execute(
            f"INSERT INTO jobs ({columns}) VALUES ({marks})", tuple(fields.values())
        )

    def update_job(self, job_id: int, **fields) -> None:
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*fields.values(), job_id))

    def add_epoch(self, job_id: int, row: dict) -> None:
        self.execute(
            "INSERT OR REPLACE INTO epochs (job_id, epoch, loss, map50_95, seconds)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                job_id,
                row.get("epoch"),
                row.get("loss"),
                row.get("map50_95") if row.get("map50_95") != "" else None,
                row.get("seconds"),
            ),
        )
