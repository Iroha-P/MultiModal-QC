import sqlite3
from datetime import datetime
from pathlib import Path


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_type VARCHAR(20) NOT NULL,
                input_path TEXT NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER REFERENCES inspections(id),
                agent_step VARCHAR(20) NOT NULL,
                agent_output TEXT NOT NULL,
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER REFERENCES inspections(id),
                defect_type VARCHAR(50),
                location TEXT,
                severity VARCHAR(10),
                suggestion TEXT
            );
        """)
        conn.commit()
        conn.close()

    def create_inspection(self, scene_type: str, input_path: str) -> int:
        conn = self._conn()
        cur = conn.execute(
            "INSERT INTO inspections (scene_type, input_path) VALUES (?, ?)",
            (scene_type, input_path),
        )
        conn.commit()
        insp_id = cur.lastrowid
        conn.close()
        return insp_id

    def get_inspection(self, insp_id: int) -> dict:
        conn = self._conn()
        row = conn.execute("SELECT * FROM inspections WHERE id=?", (insp_id,)).fetchone()
        conn.close()
        return dict(row)

    def update_status(self, insp_id: int, status: str):
        conn = self._conn()
        completed = datetime.now().isoformat() if status == "completed" else None
        conn.execute(
            "UPDATE inspections SET status=?, completed_at=? WHERE id=?",
            (status, completed, insp_id),
        )
        conn.commit()
        conn.close()

    def add_result(self, inspection_id: int, agent_step: str, agent_output: str, confidence: float):
        conn = self._conn()
        conn.execute(
            "INSERT INTO results (inspection_id, agent_step, agent_output, confidence) VALUES (?,?,?,?)",
            (inspection_id, agent_step, agent_output, confidence),
        )
        conn.commit()
        conn.close()

    def get_results(self, inspection_id: int) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM results WHERE inspection_id=? ORDER BY id", (inspection_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def add_defect(self, inspection_id: int, defect_type: str, location: str, severity: str, suggestion: str):
        conn = self._conn()
        conn.execute(
            "INSERT INTO defects (inspection_id, defect_type, location, severity, suggestion) VALUES (?,?,?,?,?)",
            (inspection_id, defect_type, location, severity, suggestion),
        )
        conn.commit()
        conn.close()

    def get_defects(self, inspection_id: int) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM defects WHERE inspection_id=? ORDER BY id", (inspection_id,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
