"""
Race State module.

This module owns the seam between presentation code and race-state data. Replay
mode is the first adapter; future live providers should return the same
RaceStateSnapshot shape instead of teaching the Streamlit app provider details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


TIMEDELTA_COLUMNS = ["LapTime", "LapStartTime", "PitInTime", "PitOutTime"]


@dataclass(frozen=True)
class RaceStateSnapshot:
    race_name: str
    year: int
    mode: str
    provider: str
    laps: pd.DataFrame
    as_of_lap: int | None = None
    timestamp_utc: str | None = None
    session_key: str | int | None = None
    session_status: str = "unknown"
    drivers: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    stints: pd.DataFrame = field(default_factory=pd.DataFrame)
    pits: pd.DataFrame = field(default_factory=pd.DataFrame)
    weather: pd.DataFrame = field(default_factory=pd.DataFrame)
    track_status: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality_flags: tuple[str, ...] = ()

    @property
    def current_lap(self) -> int:
        if self.as_of_lap is not None:
            return int(self.as_of_lap)
        if self.laps.empty:
            return 0
        return int(self.laps["LapNumber"].max())

    @property
    def total_laps(self) -> int:
        if self.laps.empty:
            return 0
        return int(self.laps["LapNumber"].max())

    def to_session(self):
        """Compatibility shape for strategy helpers that expect session.laps."""
        return SimpleNamespace(laps=self.laps.copy())


class RaceStateAdapter(ABC):
    """Provider-neutral race-state boundary used by Streamlit and live workers."""

    @abstractmethod
    def available_sessions(self) -> list[dict]:
        """Return session descriptors that can be passed to load_snapshot."""

    @abstractmethod
    def load_snapshot(self, session, as_of_lap: int | None = None) -> RaceStateSnapshot:
        """Return the normalized snapshot for a replay or live session."""


class ReplayRaceStateAdapter(RaceStateAdapter):
    """Adapter backed by the bundled static 2024 lap cache."""

    def __init__(self, static_laps_path: Path, year: int = 2024):
        self.static_laps_path = Path(static_laps_path)
        self.year = year

    def load_laps(self) -> pd.DataFrame:
        laps = pd.read_csv(self.static_laps_path)
        for col in TIMEDELTA_COLUMNS:
            if col in laps:
                laps[col] = pd.to_timedelta(laps[col], unit="s")
        if "TrackStatus" in laps:
            laps["TrackStatus"] = laps["TrackStatus"].astype(str)
        return laps

    def available_races(self) -> list[str]:
        laps = self.load_laps()
        if "Race" not in laps:
            return []
        return sorted(laps["Race"].dropna().unique().tolist())

    def available_sessions(self) -> list[dict]:
        return [
            {
                "session_key": race_name,
                "race_name": race_name,
                "year": self.year,
                "mode": "replay",
                "provider": "bundled_static_laps",
            }
            for race_name in self.available_races()
        ]

    def load_snapshot(self, race_name: str, as_of_lap: int | None = None) -> RaceStateSnapshot:
        laps = self.load_laps()
        race_laps = laps[laps["Race"] == race_name].drop(columns=["Race"]).copy()
        if race_laps.empty:
            raise ValueError(f"No bundled {self.year} lap data found for {race_name}.")
        if as_of_lap is not None:
            race_laps = race_laps[race_laps["LapNumber"] <= as_of_lap].copy()

        drivers = (
            race_laps[["Driver"]]
            .dropna()
            .drop_duplicates()
            .sort_values("Driver")
            .reset_index(drop=True)
        )
        drivers["active"] = True

        return RaceStateSnapshot(
            race_name=race_name,
            year=self.year,
            mode="replay",
            provider="bundled_static_laps",
            session_key=race_name,
            session_status="replay",
            as_of_lap=as_of_lap,
            laps=race_laps,
            drivers=drivers,
            quality_flags=("replay_data",),
        )
