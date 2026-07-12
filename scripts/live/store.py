from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3

import pandas as pd

from race_state import RaceStateSnapshot


SNAPSHOT_TABLES = {
    "drivers": "drivers",
    "laps": "laps",
    "positions": "positions_intervals",
    "stints": "stints",
    "pits": "pits",
    "weather": "weather",
    "track_status": "track_status",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiveRaceStore:
    """SQLite-backed normalized race-state store for the live MVP."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_key TEXT PRIMARY KEY,
                    race_name TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    session_status TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            for table in SNAPSHOT_TABLES.values():
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        session_key TEXT NOT NULL,
                        row_json TEXT NOT NULL,
                        updated_at_utc TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table}_session ON {table} (session_key)"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshot_metadata (
                    session_key TEXT PRIMARY KEY,
                    as_of_lap INTEGER,
                    timestamp_utc TEXT,
                    quality_flags_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )

    def write_snapshot(self, snapshot: RaceStateSnapshot):
        session_key = str(snapshot.session_key or snapshot.race_name)
        updated_at = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_key, race_name, year, mode, provider, session_status, updated_at_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    race_name=excluded.race_name,
                    year=excluded.year,
                    mode=excluded.mode,
                    provider=excluded.provider,
                    session_status=excluded.session_status,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (
                    session_key,
                    snapshot.race_name,
                    int(snapshot.year),
                    snapshot.mode,
                    snapshot.provider,
                    snapshot.session_status,
                    updated_at,
                ),
            )
            for attr, table in SNAPSHOT_TABLES.items():
                conn.execute(f"DELETE FROM {table} WHERE session_key = ?", (session_key,))
                frame = getattr(snapshot, attr)
                if frame is None or frame.empty:
                    continue
                rows = [
                    (
                        session_key,
                        json.dumps(row, default=str, allow_nan=False),
                        updated_at,
                    )
                    for row in frame.where(pd.notna(frame), None).to_dict(orient="records")
                ]
                conn.executemany(
                    f"INSERT INTO {table} (session_key, row_json, updated_at_utc) VALUES (?, ?, ?)",
                    rows,
                )
            conn.execute(
                """
                INSERT INTO snapshot_metadata (
                    session_key, as_of_lap, timestamp_utc, quality_flags_json, updated_at_utc
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_key) DO UPDATE SET
                    as_of_lap=excluded.as_of_lap,
                    timestamp_utc=excluded.timestamp_utc,
                    quality_flags_json=excluded.quality_flags_json,
                    updated_at_utc=excluded.updated_at_utc
                """,
                (
                    session_key,
                    snapshot.as_of_lap,
                    snapshot.timestamp_utc,
                    json.dumps(list(snapshot.quality_flags)),
                    updated_at,
                ),
            )

    def available_sessions(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT session_key, race_name, year, mode, provider, session_status, updated_at_utc
                FROM sessions
                ORDER BY updated_at_utc DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def _load_frame(self, conn, table: str, session_key: str) -> pd.DataFrame:
        rows = conn.execute(
            f"SELECT row_json FROM {table} WHERE session_key = ?",
            (session_key,),
        ).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([json.loads(row["row_json"]) for row in rows])

    def load_snapshot(self, session_key: str | int | None = None) -> RaceStateSnapshot | None:
        with self.connect() as conn:
            if session_key is None:
                session = conn.execute(
                    """
                    SELECT * FROM sessions
                    ORDER BY updated_at_utc DESC
                    LIMIT 1
                    """
                ).fetchone()
            else:
                session = conn.execute(
                    "SELECT * FROM sessions WHERE session_key = ?",
                    (str(session_key),),
                ).fetchone()
            if session is None:
                return None
            session_key = session["session_key"]
            metadata = conn.execute(
                "SELECT * FROM snapshot_metadata WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            frames = {
                attr: self._load_frame(conn, table, session_key)
                for attr, table in SNAPSHOT_TABLES.items()
            }

        laps = frames["laps"]
        for column in ["LapTime", "LapStartTime", "PitInTime", "PitOutTime"]:
            if column in laps:
                laps[column] = pd.to_timedelta(laps[column], errors="coerce")

        flags = ()
        as_of_lap = None
        timestamp = None
        if metadata is not None:
            flags = tuple(json.loads(metadata["quality_flags_json"]))
            as_of_lap = metadata["as_of_lap"]
            timestamp = metadata["timestamp_utc"]

        return RaceStateSnapshot(
            race_name=session["race_name"],
            year=int(session["year"]),
            mode=session["mode"],
            provider=session["provider"],
            session_key=session_key,
            session_status=session["session_status"],
            as_of_lap=as_of_lap,
            timestamp_utc=timestamp,
            laps=laps,
            drivers=frames["drivers"],
            positions=frames["positions"],
            stints=frames["stints"],
            pits=frames["pits"],
            weather=frames["weather"],
            track_status=frames["track_status"],
            quality_flags=flags,
        )

