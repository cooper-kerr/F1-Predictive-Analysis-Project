import pandas as pd

from live.providers.openf1 import normalize_openf1_snapshot


def test_openf1_payload_mapping_returns_race_state_snapshot():
    payloads = {
        "drivers": [
            {"driver_number": 1, "name_acronym": "VER", "full_name": "Max Verstappen", "team_name": "Red Bull"},
            {"driver_number": 16, "name_acronym": "LEC", "full_name": "Charles Leclerc", "team_name": "Ferrari"},
        ],
        "laps": [
            {"driver_number": 1, "lap_number": 1, "lap_duration": 91.1},
            {"driver_number": 16, "lap_number": 1, "lap_duration": 91.8},
            {"driver_number": 1, "lap_number": 2, "lap_duration": 90.9},
        ],
        "position": [
            {"date": "2026-07-12T20:00:00Z", "driver_number": 1, "position": 1},
            {"date": "2026-07-12T20:00:00Z", "driver_number": 16, "position": 2},
        ],
        "intervals": [
            {"date": "2026-07-12T20:00:00Z", "driver_number": 16, "interval": 1.2, "gap_to_leader": 1.2},
        ],
        "stints": [
            {"driver_number": 1, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1, "lap_end": 20, "tyre_age_at_start": 1},
            {"driver_number": 16, "stint_number": 1, "compound": "SOFT", "lap_start": 1, "lap_end": 18, "tyre_age_at_start": 1},
        ],
        "pit": [{"driver_number": 16, "lap_number": 1, "pit_duration": 23.4, "date": "2026-07-12T20:02:00Z"}],
        "weather": [{"date": "2026-07-12T20:00:00Z", "air_temperature": 28.0}],
        "race_control": [{"date": "2026-07-12T20:00:00Z", "message": "GREEN LIGHT"}],
    }
    session = {"session_key": 123, "race_name": "British Grand Prix", "year": 2026, "session_status": "live"}

    snapshot = normalize_openf1_snapshot(payloads, session, as_of_lap=1, timestamp_utc="2026-07-12T20:00:00+00:00")

    assert snapshot.mode == "live"
    assert snapshot.provider == "openf1"
    assert snapshot.session_key == 123
    assert snapshot.current_lap == 1
    assert snapshot.laps["Driver"].tolist() == ["VER", "LEC"]
    assert snapshot.laps["LapTime"].dt.total_seconds().tolist() == [91.1, 91.8]
    assert snapshot.laps["Compound"].tolist() == ["MEDIUM", "SOFT"]
    assert pd.notna(snapshot.laps.loc[snapshot.laps["Driver"] == "LEC", "PitInTime"]).all()
    assert "Interval" in snapshot.positions.columns

