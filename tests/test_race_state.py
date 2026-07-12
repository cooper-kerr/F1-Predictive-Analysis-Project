from pathlib import Path

import pandas as pd
import pytest

from race_state import ReplayRaceStateAdapter


def write_static_laps(path: Path):
    rows = [
        {
            "Race": "Bahrain Grand Prix",
            "Driver": "ALB",
            "LapNumber": 1,
            "LapTime": 92.1,
            "LapStartTime": 0.0,
            "PitInTime": None,
            "PitOutTime": None,
            "TrackStatus": 1,
        },
        {
            "Race": "Bahrain Grand Prix",
            "Driver": "TSU",
            "LapNumber": 1,
            "LapTime": 91.9,
            "LapStartTime": 1.2,
            "PitInTime": None,
            "PitOutTime": None,
            "TrackStatus": 1,
        },
        {
            "Race": "Saudi Arabian Grand Prix",
            "Driver": "ALB",
            "LapNumber": 1,
            "LapTime": 90.5,
            "LapStartTime": 0.0,
            "PitInTime": None,
            "PitOutTime": None,
            "TrackStatus": 1,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_replay_adapter_returns_provider_neutral_snapshot(tmp_path):
    path = tmp_path / "laps.csv"
    write_static_laps(path)

    snapshot = ReplayRaceStateAdapter(path).load_snapshot(
        "Bahrain Grand Prix", as_of_lap=1
    )

    assert snapshot.mode == "replay"
    assert snapshot.provider == "bundled_static_laps"
    assert snapshot.race_name == "Bahrain Grand Prix"
    assert snapshot.session_key == "Bahrain Grand Prix"
    assert snapshot.current_lap == 1
    assert snapshot.total_laps == 1
    assert snapshot.quality_flags == ("replay_data",)
    assert snapshot.laps["Driver"].tolist() == ["ALB", "TSU"]
    assert snapshot.drivers["Driver"].tolist() == ["ALB", "TSU"]
    assert snapshot.laps["LapTime"].dt.total_seconds().tolist() == [92.1, 91.9]
    assert snapshot.laps["TrackStatus"].tolist() == ["1", "1"]


def test_replay_snapshot_keeps_strategy_session_compatibility(tmp_path):
    path = tmp_path / "laps.csv"
    write_static_laps(path)

    session = ReplayRaceStateAdapter(path).load_snapshot("Bahrain Grand Prix").to_session()

    assert hasattr(session, "laps")
    assert session.laps["Driver"].tolist() == ["ALB", "TSU"]


def test_replay_adapter_exposes_available_sessions(tmp_path):
    path = tmp_path / "laps.csv"
    write_static_laps(path)

    sessions = ReplayRaceStateAdapter(path).available_sessions()

    assert sessions == [
        {
            "session_key": "Bahrain Grand Prix",
            "race_name": "Bahrain Grand Prix",
            "year": 2024,
            "mode": "replay",
            "provider": "bundled_static_laps",
        },
        {
            "session_key": "Saudi Arabian Grand Prix",
            "race_name": "Saudi Arabian Grand Prix",
            "year": 2024,
            "mode": "replay",
            "provider": "bundled_static_laps",
        },
    ]


def test_replay_adapter_rejects_unknown_race(tmp_path):
    path = tmp_path / "laps.csv"
    write_static_laps(path)

    with pytest.raises(ValueError, match="No bundled 2024 lap data found"):
        ReplayRaceStateAdapter(path).load_snapshot("Monaco Grand Prix")
