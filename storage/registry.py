"""
storage/registry.py — SQLite-backed document registry.

Replaces data/doc_registry.json. Drop-in replacement:
  same 4 functions used by pipeline/ingest.py:
    get_registry()           → dict of all docs
    update_registry()        → upsert one doc's fields
    find_existing()          → dedup check by filename
    delete_registry_entry()  → remove a doc_id

WHY SQLITE OVER JSON:
  - Concurrent writes safe (WAL mode) — JSON has race conditions with BackgroundTasks
  - Survives partial writes / crashes without corruption
  - Queryable — Phase 4 eval queries become trivial
  - Same zero-infra story as JSON (single file, no server)

TABLE SCHEMA:
  doc_id           TEXT PRIMARY KEY
  filename         TEXT NOT NULL
  status           TEXT NOT NULL   (processing | ready | failed)
  ingested_at      TEXT
  pages            INTEGER
  small_chunks     INTEGER
  large_chunks     INTEGER
  total_chars      INTEGER
  ingestion_time_s REAL
  error            TEXT
"""

import sqlite3
import logging
import shutil
from pathlib import Path
from contextlib import contextmanager
from config.settings import settings

logger = logging.getLogger(__name__)

# SQLite file lives next to where the old JSON file was
DB_PATH: Path = settings.doc_registry_path.parent / "doc_registry.db"


# ── CONNECTION ─────────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    """
    Yields a connection with:
      - WAL mode   → concurrent reads + writes don't block each other
      - Row factory → rows behave like dicts (row["doc_id"] not row[0])
      - Timeout    → wait up to 5s if another writer holds the lock
    """
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── SCHEMA INIT ────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the docs table if it doesn't exist.
    Safe to call on every startup — CREATE TABLE IF NOT EXISTS is idempotent.
    Called once at import time (bottom of this file).
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS docs (
                doc_id           TEXT PRIMARY KEY,
                filename         TEXT NOT NULL,
                status           TEXT NOT NULL DEFAULT 'processing',
                ingested_at      TEXT,
                pages            INTEGER,
                small_chunks     INTEGER,
                large_chunks     INTEGER,
                total_chars      INTEGER,
                ingestion_time_s REAL,
                error            TEXT
            )
        """)
    logger.info(f"[REGISTRY] SQLite DB ready at {DB_PATH}")


# ── PUBLIC API (same interface as the old JSON functions) ──────────────────────

def get_registry() -> dict:
    """
    Return all docs as a dict keyed by doc_id.
    Matches the old _load_registry() return shape exactly.
    """
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM docs").fetchall()
    return {row["doc_id"]: dict(row) for row in rows}


def update_registry(doc_id: str, update: dict) -> None:
    """
    Upsert a doc entry. On conflict (same doc_id), only the provided
    fields are updated — other fields stay unchanged.

    Uses INSERT OR IGNORE + UPDATE pattern so:
      - First call (status=processing): inserts the row
      - Later calls (status=ready): updates only the changed fields
    """
    with _get_conn() as conn:
        # Ensure the row exists first
        conn.execute(
            "INSERT OR IGNORE INTO docs (doc_id, filename, status) VALUES (?, ?, ?)",
            (
                doc_id,
                update.get("filename", ""),
                update.get("status", "processing"),
            ),
        )
        # Now update only the fields we were given
        allowed = {
            "filename", "status", "ingested_at", "pages",
            "small_chunks", "large_chunks", "total_chars",
            "ingestion_time_s", "error",
        }
        fields = {k: v for k, v in update.items() if k in allowed}
        if fields:
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            values = list(fields.values()) + [doc_id]
            conn.execute(f"UPDATE docs SET {set_clause} WHERE doc_id = ?", values)


def find_existing(filename: str) -> str | None:
    """
    Return doc_id if this filename is already ingested with status='ready'.
    Also deletes stale processing/failed entries for the same filename
    and removes their index directories.

    Same logic as the old _find_existing() in ingest.py.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT doc_id, status FROM docs WHERE filename = ?", (filename,)
        ).fetchall()

    found_id = None
    stale_ids = []

    for row in rows:
        if row["status"] == "ready":
            found_id = row["doc_id"]
        elif row["status"] in ("processing", "failed"):
            stale_ids.append(row["doc_id"])

    if stale_ids:
        with _get_conn() as conn:
            conn.executemany(
                "DELETE FROM docs WHERE doc_id = ?",
                [(sid,) for sid in stale_ids],
            )
        for stale_id in stale_ids:
            logger.info(f"[REGISTRY] Removed stale '{filename}' doc_id={stale_id}")
            stale_index_dir = settings.indexes_dir / stale_id
            if stale_index_dir.exists():
                shutil.rmtree(stale_index_dir)

    return found_id


def delete_registry_entry(doc_id: str) -> None:
    """
    Hard-delete a doc from the registry.
    Used by Phase 3 session cleanup.
    Does NOT delete the index directory — caller is responsible.
    """
    with _get_conn() as conn:
        conn.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))
    logger.info(f"[REGISTRY] Deleted doc_id={doc_id}")


# ── MIGRATION HELPER ───────────────────────────────────────────────────────────

def migrate_from_json(json_path: Path) -> int:
    """
    One-time migration: reads old doc_registry.json and inserts all entries
    into SQLite. Safe to call even if already migrated (INSERT OR IGNORE).

    Usage (run once from terminal):
        python3 -c "from storage.registry import migrate_from_json; from pathlib import Path; migrate_from_json(Path('data/doc_registry.json'))"

    Returns number of rows inserted.
    """
    import json
    if not json_path.exists():
        logger.info("[REGISTRY] No JSON registry found, skipping migration.")
        return 0

    with open(json_path) as f:
        data = json.load(f)

    count = 0
    for doc_id, meta in data.items():
        update_registry(doc_id, {**meta, "doc_id": doc_id})
        count += 1

    logger.info(f"[REGISTRY] Migrated {count} entries from {json_path}")
    return count


# ── AUTO-INIT ──────────────────────────────────────────────────────────────────
# Runs once when this module is first imported.
# Creates the table if it doesn't exist — zero manual setup required.
init_db()