"""SQLite persistence of analysis runs (one row per /analyze call)."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app import config


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(config.DB_PATH))


def init_db() -> None:
    with _connect() as con:
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


def save_run(input_type: str, url: Optional[str], text: Optional[str], domain: Optional[str],
             mode: str, verifier: str, response: Dict[str, Any]) -> int:
    init_db()
    with _connect() as con:
        cur = con.execute(
            """
            INSERT INTO runs (created_utc, input_type, url, text_preview, domain, mode, verifier, response_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), input_type, url, (text or "")[:400] if text else None, domain, mode, verifier,
             json.dumps(response, ensure_ascii=False)),
        )
        con.commit()
        return int(cur.lastrowid)


def list_runs(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, created_utc, input_type, url, domain, mode, verifier FROM runs ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with _connect() as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM runs WHERE id = ?", (int(run_id),)).fetchone()
        if not row:
            return None
        d = dict(row)
        try:
            d["response"] = json.loads(d.pop("response_json") or "{}")
        except Exception:
            d["response"] = {}
        return d


def export_runs(limit: int = 500) -> List[Dict[str, Any]]:
    init_db()
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT id, created_utc, input_type, url, text_preview, domain, mode, verifier, response_json "
            "FROM runs ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["response"] = json.loads(d.pop("response_json") or "{}")
        except Exception:
            d["response"] = {}
        out.append(d)
    return out


def export_runs_csv(limit: int = 500) -> str:
    """One CSV row per claim, across the last `limit` runs."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run_id", "created_utc", "input_type", "url", "domain", "verifier", "claim_index", "claim_text",
                "verdict", "confidence", "verdict_source", "adjusted_verdict", "adjusted_confidence",
                "n_evidence", "misinformation_likelihood"])
    for run in export_runs(limit):
        resp = run.get("response") or {}
        for i, c in enumerate(resp.get("claims") or [], 1):
            w.writerow([run["id"], run["created_utc"], run["input_type"], run.get("url") or "", run.get("domain") or "",
                        run.get("verifier") or "", i, c.get("claim_text", ""), c.get("verdict", ""), c.get("confidence", ""),
                        c.get("verdict_source", ""), c.get("adjusted_verdict") or "", c.get("adjusted_confidence") or "",
                        len(c.get("evidence") or []), resp.get("final_misinformation_likelihood", "")])
    return buf.getvalue()
