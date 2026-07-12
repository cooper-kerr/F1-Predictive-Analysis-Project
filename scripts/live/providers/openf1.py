from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import time

import pandas as pd

from race_state import RaceStateSnapshot


OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPENF1_RATE_LIMIT_SECONDS = 0.25


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=columns)


def _seconds(value):
    if value is None or pd.isna(value):
        return pd.NaT
    return pd.to_timedelta(float(value), unit="s")


def normalize_openf1_snapshot(
    payloads: dict[str, list[dict]],
    session: dict,
    as_of_lap: int | None = None,
    timestamp_utc: str | None = None,
) -> RaceStateSnapshot:
    """Map OpenF1-shaped endpoint payloads into the app snapshot contract."""

    drivers_raw = payloads.get("drivers", [])
    laps_raw = payloads.get("laps", [])
    positions_raw = payloads.get("position", [])
    intervals_raw = payloads.get("intervals", [])
    stints_raw = payloads.get("stints", [])
    pits_raw = payloads.get("pit", [])
    weather_raw = payloads.get("weather", [])
    race_control_raw = payloads.get("race_control", [])

    driver_lookup = {
        row.get("driver_number"): row.get("name_acronym") or row.get("broadcast_name")
        for row in drivers_raw
    }

    drivers = _frame(
        [
            {
                "DriverNumber": row.get("driver_number"),
                "Driver": row.get("name_acronym") or row.get("broadcast_name"),
                "FullName": row.get("full_name") or row.get("broadcast_name"),
                "TeamName": row.get("team_name"),
                "TeamColour": row.get("team_colour"),
                "active": True,
            }
            for row in drivers_raw
        ],
        ["DriverNumber", "Driver", "FullName", "TeamName", "TeamColour", "active"],
    )

    lap_rows = []
    for row in laps_raw:
        lap_number = row.get("lap_number")
        if as_of_lap is not None and lap_number is not None and int(lap_number) > as_of_lap:
            continue
        lap_rows.append(
            {
                "DriverNumber": row.get("driver_number"),
                "Driver": driver_lookup.get(row.get("driver_number"), str(row.get("driver_number"))),
                "LapNumber": lap_number,
                "LapTime": _seconds(row.get("lap_duration")),
                "LapStartTime": pd.to_timedelta(lap_number or 0, unit="m"),
                "IsAccurate": not bool(row.get("deleted")),
                "TrackStatus": "1",
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
                "TyreLife": row.get("tyre_age") or row.get("tire_age"),
                "Compound": row.get("compound") or "UNKNOWN",
            }
        )
    laps = _frame(
        lap_rows,
        [
            "DriverNumber",
            "Driver",
            "LapNumber",
            "LapTime",
            "LapStartTime",
            "IsAccurate",
            "TrackStatus",
            "PitInTime",
            "PitOutTime",
            "TyreLife",
            "Compound",
        ],
    )
    for timing_col in ["LapTime", "LapStartTime", "PitInTime", "PitOutTime"]:
        if timing_col in laps:
            values = laps[timing_col].map(
                lambda value: pd.NaT if pd.isna(value) else pd.to_timedelta(value)
            ).tolist()
            laps[timing_col] = pd.Series(values, dtype="timedelta64[ns]", index=laps.index)

    stints = _frame(
        [
            {
                "DriverNumber": row.get("driver_number"),
                "Driver": driver_lookup.get(row.get("driver_number"), str(row.get("driver_number"))),
                "Stint": row.get("stint_number"),
                "Compound": row.get("compound"),
                "LapStart": row.get("lap_start"),
                "LapEnd": row.get("lap_end"),
                "TyreAgeStart": row.get("tyre_age_at_start") or row.get("tire_age_at_start"),
            }
            for row in stints_raw
        ],
        ["DriverNumber", "Driver", "Stint", "Compound", "LapStart", "LapEnd", "TyreAgeStart"],
    )
    if not stints.empty and not laps.empty:
        for lap_idx, lap in laps.iterrows():
            matches = stints[
                (stints["DriverNumber"] == lap["DriverNumber"])
                & (stints["LapStart"].fillna(-1) <= lap["LapNumber"])
                & (stints["LapEnd"].fillna(lap["LapNumber"]) >= lap["LapNumber"])
            ]
            if not matches.empty:
                stint = matches.iloc[-1]
                laps.at[lap_idx, "Compound"] = stint.get("Compound") or "UNKNOWN"
                age_start = stint.get("TyreAgeStart")
                if pd.notna(age_start) and pd.notna(stint.get("LapStart")):
                    laps.at[lap_idx, "TyreLife"] = int(lap["LapNumber"] - stint["LapStart"] + age_start)

    pits = _frame(
        [
            {
                "DriverNumber": row.get("driver_number"),
                "Driver": driver_lookup.get(row.get("driver_number"), str(row.get("driver_number"))),
                "LapNumber": row.get("lap_number"),
                "PitDuration": row.get("pit_duration"),
                "Date": row.get("date"),
            }
            for row in pits_raw
        ],
        ["DriverNumber", "Driver", "LapNumber", "PitDuration", "Date"],
    )

    positions = _frame(
        [
            {
                "Date": row.get("date"),
                "DriverNumber": row.get("driver_number"),
                "Driver": driver_lookup.get(row.get("driver_number"), str(row.get("driver_number"))),
                "Position": row.get("position"),
            }
            for row in positions_raw
        ],
        ["Date", "DriverNumber", "Driver", "Position"],
    )
    intervals = _frame(
        [
            {
                "Date": row.get("date"),
                "DriverNumber": row.get("driver_number"),
                "Driver": driver_lookup.get(row.get("driver_number"), str(row.get("driver_number"))),
                "GapToLeader": row.get("gap_to_leader"),
                "Interval": row.get("interval"),
            }
            for row in intervals_raw
        ],
        ["Date", "DriverNumber", "Driver", "GapToLeader", "Interval"],
    )
    if not intervals.empty:
        positions = positions.merge(
            intervals,
            on=["Date", "DriverNumber", "Driver"],
            how="outer",
        )

    weather = _frame(weather_raw, [])
    track_status = _frame(
        [
            {
                "Date": row.get("date"),
                "Category": row.get("category"),
                "Flag": row.get("flag"),
                "Scope": row.get("scope"),
                "Message": row.get("message"),
            }
            for row in race_control_raw
        ],
        ["Date", "Category", "Flag", "Scope", "Message"],
    )

    if not track_status.empty and not laps.empty:
        messages = track_status["Message"].fillna("").astype(str).str.upper()
        status = "4" if messages.str.contains("SAFETY CAR").any() else "1"
        laps["TrackStatus"] = status

    if not pits.empty and not laps.empty:
        for pit in pits.itertuples(index=False):
            mask = (laps["DriverNumber"] == pit.DriverNumber) & (laps["LapNumber"] == pit.LapNumber)
            laps.loc[mask, "PitInTime"] = pd.to_timedelta(laps.loc[mask, "LapNumber"], unit="m")
            out_mask = (laps["DriverNumber"] == pit.DriverNumber) & (laps["LapNumber"] == pit.LapNumber + 1)
            laps.loc[out_mask, "PitOutTime"] = pd.to_timedelta(
                laps.loc[out_mask, "LapNumber"], unit="m"
            ) + pd.to_timedelta(pit.PitDuration or 0, unit="s")

    session_key = session.get("session_key")
    race_name = session.get("race_name") or session.get("session_name") or str(session_key)
    year = int(session.get("year") or 0)
    return RaceStateSnapshot(
        race_name=race_name,
        year=year,
        mode="live",
        provider="openf1",
        session_key=session_key,
        session_status=session.get("session_status", "unknown"),
        as_of_lap=as_of_lap,
        timestamp_utc=timestamp_utc or _utc_now_iso(),
        laps=laps,
        drivers=drivers,
        positions=positions,
        stints=stints,
        pits=pits,
        weather=weather,
        track_status=track_status,
        quality_flags=("openf1_snapshot",),
    )


@dataclass
class OpenF1Client:
    base_url: str = OPENF1_BASE_URL
    api_key: str | None = None
    min_request_interval_seconds: float = OPENF1_RATE_LIMIT_SECONDS

    def __post_init__(self):
        self._last_request_at = 0.0

    def get(self, endpoint: str, **params) -> list[dict]:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)
        query = urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, headers=headers)
        with urlopen(request, timeout=15) as response:
            self._last_request_at = time.monotonic()
            return json.loads(response.read().decode("utf-8"))

    def available_sessions(self, year: int | None = None) -> list[dict]:
        params = {"year": year} if year else {}
        return self.get("sessions", **params)

    def load_payloads(self, session_key) -> dict[str, list[dict]]:
        endpoints = ["drivers", "laps", "position", "intervals", "stints", "pit", "weather", "race_control"]
        return {endpoint: self.get(endpoint, session_key=session_key) for endpoint in endpoints}

    def load_snapshot(self, session: dict, as_of_lap: int | None = None) -> RaceStateSnapshot:
        payloads = self.load_payloads(session["session_key"])
        return normalize_openf1_snapshot(payloads, session, as_of_lap=as_of_lap)
