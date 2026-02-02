import os
import json
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime

# Change this if you want a different DB location
DB_PATH = os.getenv("FACT_VALIDATOR_DB", r"C:\Fact_Validator\data\fact_validator.db")


def _ensure_dir(path: str) -> None:
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def init_db() -> None:
    _ensure_dir(DB_PATH)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              created_utc TEXT NOT NULL,
              input_type TEXT NOT NULL,
              url TEXT,
              text_preview TEXT,
              domain TEXT,
              mode TEXT,
              verifier TEXT,
              response_json TEXT NOT NULL
            )
            """
        )
        con.commit()


def save_run(
    input_type: str,
    url: Optional[str],
    text: Optional[str],
    domain: Optional[str],
    mode: str,
    verifier: str,
    response: Dict[str, Any],
) -> int:
    init_db()
    created_utc = datetime.utcnow().isoformat() + "Z"
    text_preview = (text or "")[:400] if text else None
    response_json = json.dumps(response, ensure_ascii=False)

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            """
            INSERT INTO runs (created_utc, input_type, url, text_preview, domain, mode, verifier, response_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (created_utc, input_type, url, text_preview, domain, mode, verifier, response_json),
        )
        con.commit()
        return int(cur.lastrowid)


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT id, created_utc, input_type, url, domain, mode, verifier
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["response"] = json.loads(d.get("response_json") or "{}")
        except Exception:
            d["response"] = {}
        return d


def export_runs(limit: int = 500) -> List[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT id, created_utc, input_type, url, text_preview, domain, mode, verifier, response_json
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["response"] = json.loads(d.get("response_json") or "{}")
            except Exception:
                d["response"] = {}
            out.append(d)
        return out
