import json
import re
import sqlite3
import threading
import zlib
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def mock_embed(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    for token in re.findall(r"\w+|[^\s\w]", text.lower()):
        idx = zlib.adler32(token.encode("utf-8")) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


class SocialMemoryBank:
    def __init__(
        self,
        db_path: str,
        player_id: int,
        embedding_dim: int = 128,
        alpha: float = 0.1,
    ) -> None:
        self.db_path = db_path
        self.player_id = int(player_id)
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.table_name = f"player_{self.player_id}_bank"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_emb TEXT NOT NULL,
                    experience TEXT NOT NULL,
                    q_value REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_q ON {self.table_name} (q_value)"
            )
            self._conn.commit()

    def _ensure_vector(self, emb: Sequence[float]) -> np.ndarray:
        vec = np.array(emb, dtype=np.float32).reshape(-1)
        if vec.size == self.embedding_dim:
            return vec
        if vec.size > self.embedding_dim:
            return vec[: self.embedding_dim]
        padded = np.zeros(self.embedding_dim, dtype=np.float32)
        padded[: vec.size] = vec
        return padded

    def add_experience(self, intent_emb: Sequence[float], experience_text: str, initial_q: float = 0.0) -> int:
        now = datetime.utcnow().isoformat()
        emb = self._ensure_vector(intent_emb)
        emb_json = json.dumps(emb.tolist())
        with self._lock:
            cur = self._conn.execute(
                f"""
                INSERT INTO {self.table_name} (intent_emb, experience, q_value, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (emb_json, experience_text, float(initial_q), now, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def retrieve_context(self, current_intent_emb: Sequence[float], top_k1: int, top_k2: int) -> List[Dict]:
        top_k1 = max(int(top_k1), int(top_k2))
        emb = self._ensure_vector(current_intent_emb)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, intent_emb, experience, q_value FROM {self.table_name}"
            ).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            stored_emb = np.array(json.loads(row["intent_emb"]), dtype=np.float32)
            sim = cosine_similarity(emb, stored_emb)
            scored.append((row, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        phase_a = scored[:top_k1]
        phase_a.sort(key=lambda x: float(x[0]["q_value"]), reverse=True)

        results = []
        for row, sim in phase_a[:top_k2]:
            results.append(
                {
                    "id": int(row["id"]),
                    "experience": row["experience"],
                    "q_value": float(row["q_value"]),
                    "similarity": float(sim),
                }
            )
        return results

    def update_q_values(self, retrieved_ids: Iterable[int], reward: float, alpha: Optional[float] = None) -> int:
        ids = [int(mem_id) for mem_id in retrieved_ids]
        if not ids:
            return 0
        alpha = float(self.alpha if alpha is None else alpha)
        now = datetime.utcnow().isoformat()
        placeholders = ",".join(["?"] * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, q_value FROM {self.table_name} WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            updates = []
            for row in rows:
                q_old = float(row["q_value"])
                q_new = q_old + alpha * (float(reward) - q_old)
                updates.append((q_new, now, int(row["id"])))
            if updates:
                self._conn.executemany(
                    f"UPDATE {self.table_name} SET q_value = ?, updated_at = ? WHERE id = ?",
                    updates,
                )
                self._conn.commit()
        return len(updates)
