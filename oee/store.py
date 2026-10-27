"""SQLite persistence: document trees, node vectors, case state, turn traces."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  doc_id TEXT PRIMARY KEY, ord INTEGER, title TEXT, framing TEXT, stance TEXT,
  limitations TEXT, n_leaves INTEGER, n_sections INTEGER, chars INTEGER,
  created_at REAL, ingest_log TEXT
);
CREATE TABLE IF NOT EXISTS nodes(
  node_id TEXT PRIMARY KEY, doc_id TEXT, kind TEXT, text TEXT, page_hint TEXT,
  tokens INTEGER, parents TEXT, meta TEXT, ord INTEGER
);
CREATE TABLE IF NOT EXISTS vectors(
  node_id TEXT PRIMARY KEY, dim INTEGER, mode TEXT, vec BLOB
);
CREATE TABLE IF NOT EXISTS turns(
  turn_id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT, created_at REAL,
  raw TEXT, clean TEXT, sanitized TEXT, bias TEXT, chat_reply TEXT,
  sidecar TEXT, trace TEXT, context_tokens INTEGER, model TEXT
);
CREATE TABLE IF NOT EXISTS case_state(
  session TEXT PRIMARY KEY, state TEXT, updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_nodes_doc ON nodes(doc_id, kind, ord);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session, turn_id);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        with self._lock:
            self.db.executescript(SCHEMA)
            self.db.execute(
                "DELETE FROM vectors WHERE node_id NOT IN (SELECT node_id FROM nodes)")
            self.db.commit()

    def close(self) -> None:
        with self._lock:
            self.db.close()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self.db.execute(sql, params)
            self.db.commit()

    # ------------------------------------------------------------- documents
    def next_doc_id(self) -> str:
        with self._lock:
            row = self.db.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return f"R{int(row['n']) + 1}"

    def save_document(self, doc: dict) -> None:
        self._exec(
            """INSERT OR REPLACE INTO documents
               (doc_id, ord, title, framing, stance, limitations, n_leaves,
                n_sections, chars, created_at, ingest_log)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (doc["doc_id"], doc.get("ord", 0), doc.get("title", ""), doc.get("framing", ""),
             doc.get("stance", ""), json.dumps(doc.get("limitations", [])),
             doc.get("n_leaves", 0), doc.get("n_sections", 0), doc.get("chars", 0),
             time.time(), json.dumps(doc.get("ingest_log", []))),
        )

    def save_nodes(self, nodes: list[dict], vectors: dict[str, list[float]],
                   modes: dict[str, str]) -> None:
        with self._lock:
            for node in nodes:
                self.db.execute(
                    """INSERT OR REPLACE INTO nodes
                       (node_id, doc_id, kind, text, page_hint, tokens, parents,
                        meta, ord) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (node["node_id"], node["doc_id"], node["kind"], node["text"],
                     node.get("page_hint", ""), node.get("tokens", 0),
                     json.dumps(node.get("parents", [])), json.dumps(node.get("meta", {})),
                     node.get("ord", 0)),
                )
            for node_id, vec in vectors.items():
                self.db.execute(
                    "INSERT OR REPLACE INTO vectors (node_id, dim, mode, vec) VALUES (?,?,?,?)",
                    (node_id, len(vec), modes.get(node_id, "hash-local"),
                     sqlite3.Binary(_pack(vec))),
                )
            self.db.commit()

    def documents(self) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM documents ORDER BY ord"
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["limitations"] = json.loads(d["limitations"] or "[]")
            d["ingest_log"] = json.loads(d["ingest_log"] or "[]")
            out.append(d)
        return out

    def doc(self, doc_id: str) -> dict | None:
        rows = self.documents()
        return next((d for d in rows if d["doc_id"] == doc_id), None)

    def delete_document(self, doc_id: str) -> None:
        self._exec("DELETE FROM vectors WHERE node_id IN "
                   "(SELECT node_id FROM nodes WHERE doc_id=?)", (doc_id,))
        self._exec("DELETE FROM nodes WHERE doc_id=?", (doc_id,))
        self._exec("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        self.prune_orphan_vectors()

    def prune_orphan_vectors(self) -> int:
        with self._lock:
            cur = self.db.execute(
                "DELETE FROM vectors WHERE node_id NOT IN (SELECT node_id FROM nodes)")
            self.db.commit()
        return cur.rowcount

    def nodes(self, kinds: tuple[str, ...] = ("leaf", "section", "card"),
              doc_id: str | None = None) -> list[dict]:
        q = "SELECT * FROM nodes WHERE kind IN (%s)" % ",".join("?" * len(kinds))
        params: list = list(kinds)
        if doc_id:
            q += " AND doc_id=?"
            params.append(doc_id)
        q += " ORDER BY doc_id, ord"
        with self._lock:
            rows = self.db.execute(q, tuple(params)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["parents"] = json.loads(d["parents"] or "[]")
            d["meta"] = json.loads(d["meta"] or "{}")
            out.append(d)
        return out

    def node(self, node_id: str) -> dict | None:
        with self._lock:
            row = self.db.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["parents"] = json.loads(d["parents"] or "[]")
        d["meta"] = json.loads(d["meta"] or "{}")
        return d

    def vectors(self) -> dict[str, tuple[list[float], str]]:
        with self._lock:
            rows = self.db.execute("SELECT node_id, dim, mode, vec FROM vectors").fetchall()
        return {r["node_id"]: (_unpack(r["vec"], r["dim"]), r["mode"]) for r in rows}

    def vector_mode_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self.db.execute(
                "SELECT mode, COUNT(*) AS n FROM vectors GROUP BY mode"
            ).fetchall()
        return {r["mode"]: int(r["n"]) for r in rows}

    def vector_dims(self) -> list[int]:
        with self._lock:
            rows = self.db.execute(
                "SELECT dim, COUNT(*) AS n FROM vectors GROUP BY dim"
            ).fetchall()
        out: list[int] = []
        for r in rows:
            out += [int(r["dim"])] * int(r["n"])
        return out

    def save_vectors(self, vectors: dict[str, list[float]], modes: dict[str, str]) -> None:
        with self._lock:
            for node_id, vec in vectors.items():
                self.db.execute(
                    "INSERT OR REPLACE INTO vectors (node_id, dim, mode, vec) VALUES (?,?,?,?)",
                    (node_id, len(vec), modes.get(node_id, "hash-local"),
                     sqlite3.Binary(_pack(vec))),
                )
            self.db.commit()

    # ----------------------------------------------------------------- turns
    def add_turn(self, turn: dict) -> int:
        with self._lock:
            cur = self.db.execute(
                """INSERT INTO turns
                   (session, created_at, raw, clean, sanitized, bias, chat_reply,
                    sidecar, trace, context_tokens, model) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (turn["session"], time.time(), turn.get("raw", ""), turn.get("clean", ""),
                 turn.get("sanitized", ""), json.dumps(turn.get("bias", [])),
                 turn.get("chat_reply", ""), json.dumps(turn.get("sidecar", {})),
                 json.dumps(turn.get("trace", [])), turn.get("context_tokens", 0),
                 turn.get("model", "")),
            )
            self.db.commit()
            return int(cur.lastrowid)

    def turns(self, session: str = "default", limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM turns WHERE session=? ORDER BY turn_id DESC LIMIT ?",
                (session, limit),
            ).fetchall()
        out = []
        for r in reversed(rows):
            d = dict(r)
            d["bias"] = json.loads(d["bias"] or "[]")
            d["sidecar"] = json.loads(d["sidecar"] or "{}")
            d["trace"] = json.loads(d["trace"] or "[]")
            out.append(d)
        return out

    # ------------------------------------------------------------ case state
    def get_case_state(self, session: str) -> dict:
        with self._lock:
            row = self.db.execute(
                "SELECT state FROM case_state WHERE session=?", (session,)
            ).fetchone()
        if not row:
            return {
                "rolling_summary": "",
                "established_claims": [],
                "open_questions": [],
                "user_constraints": [],
                "user_pressure": [],
                "turn_count": 0,
            }
        return json.loads(row["state"])

    def set_case_state(self, session: str, state: dict) -> None:
        self._exec(
            "INSERT OR REPLACE INTO case_state (session, state, updated_at) VALUES (?,?,?)",
            (session, json.dumps(state), time.time()),
        )

    def stats(self) -> dict:
        with self._lock:
            counts = {}
            for table in ("documents", "nodes", "vectors", "turns"):
                counts[table] = int(
                    self.db.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                )
        return counts


def _pack(vec) -> bytes:
    import struct

    return b"".join(struct.pack("<f", float(x)) for x in vec)


def _unpack(blob: bytes, dim: int) -> list[float]:
    import struct

    return list(struct.unpack(f"<{dim}f", bytes(blob)))
