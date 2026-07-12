import pandas as pd

from live.store import LiveRaceStore
from race_state import RaceStateSnapshot


def test_live_store_round_trips_latest_snapshot(tmp_path):
    store = LiveRaceStore(tmp_path / "live.sqlite")
    snapshot = RaceStateSnapshot(
        race_name="Test Grand Prix",
        year=2026,
        mode="live",
        provider="openf1",
        session_key="abc",
        session_status="live",
        timestamp_utc="2026-07-12T20:00:00+00:00",
        laps=pd.DataFrame(
            [
                {
                    "Driver": "AAA",
                    "LapNumber": 1,
                    "LapTime": pd.to_timedelta(90, unit="s"),
                    "LapStartTime": pd.to_timedelta(1, unit="m"),
                    "IsAccurate": True,
                    "TrackStatus": "1",
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                    "TyreLife": 1,
                    "Compound": "MEDIUM",
                }
            ]
        ),
        drivers=pd.DataFrame([{"Driver": "AAA", "active": True}]),
        quality_flags=("openf1_snapshot",),
    )

    store.write_snapshot(snapshot)
    loaded = store.load_snapshot()

    assert loaded is not None
    assert loaded.race_name == "Test Grand Prix"
    assert loaded.provider == "openf1"
    assert loaded.quality_flags == ("openf1_snapshot",)
    assert loaded.laps["LapTime"].dt.total_seconds().tolist() == [90.0]
    assert store.available_sessions()[0]["session_key"] == "abc"

