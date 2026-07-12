from datetime import datetime, timedelta, timezone

import pandas as pd

from live.predict import make_action_signal
from live.quality import quality_flags_for_prediction
from race_state import RaceStateSnapshot


def _snapshot(track_status="1", compound="MEDIUM", timestamp=None):
    rows = []
    for lap in range(1, 5):
        rows.append(
            {
                "Driver": "A",
                "LapNumber": lap,
                "LapTime": pd.to_timedelta(90 + lap, unit="s"),
                "LapStartTime": pd.to_timedelta(lap * 100 + 2, unit="s"),
                "IsAccurate": True,
                "TrackStatus": track_status,
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
                "TyreLife": lap,
                "Compound": compound,
            }
        )
        rows.append(
            {
                "Driver": "B",
                "LapNumber": lap,
                "LapTime": pd.to_timedelta(90 + lap, unit="s"),
                "LapStartTime": pd.to_timedelta(lap * 100, unit="s"),
                "IsAccurate": True,
                "TrackStatus": track_status,
                "PitInTime": pd.NaT,
                "PitOutTime": pd.NaT,
                "TyreLife": lap,
                "Compound": compound,
            }
        )
    return RaceStateSnapshot(
        race_name="Race",
        year=2026,
        mode="live",
        provider="openf1",
        session_key="race",
        session_status="live",
        timestamp_utc=timestamp or datetime.now(timezone.utc).isoformat(),
        laps=pd.DataFrame(rows),
    )


def test_quality_flags_suppress_stale_safety_car_and_wet_modes():
    stale_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
    snapshot = _snapshot(track_status="4", compound="INTERMEDIATE", timestamp=stale_time)
    context = {
        "driver": "A",
        "rival": "B",
        "lap": 4,
        "compound": "INTERMEDIATE",
        "gap_ahead": 2.0,
        "own_pace": 90.0,
        "threat_pace": 91.0,
        "pace_delta": -1.0,
        "deg_delta": 0.1,
        "ca_deg_delta": 0.2,
        "pit_loss": 22.0,
        "tire_age": 4,
        "rival_tire_age": 4,
    }

    flags = quality_flags_for_prediction(snapshot, context, max_staleness_seconds=30)
    signal = make_action_signal(
        2.0,
        0.6,
        0.4,
        context,
        {
            "undercut": {"overall_rate": 0.4, "median_gap": 2.5},
            "overcut": {"overall_rate": 0.3, "median_gap": 3.5},
        },
        flags,
    )

    assert {"stale_feed", "safety_car_or_vsc", "wet_mode"}.issubset(flags)
    assert signal.suppressed is True
    assert signal.call == "Suppressed"
