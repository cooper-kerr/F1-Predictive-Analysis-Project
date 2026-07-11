from types import SimpleNamespace

import numpy as np
import pandas as pd

from race_strategy import (
    build_strategy_context,
    current_lap_row,
    nearest_pit_label_row,
)


def _time(seconds):
    return pd.to_timedelta(seconds, unit="s")


def _session():
    rows = []
    for lap in range(1, 7):
        for driver, offset in [("B", 0), ("A", 2)]:
            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": lap,
                    "LapTime": _time(90 + lap * 0.1 + (0.2 if driver == "A" else 0)),
                    "LapStartTime": _time(lap * 100 + offset),
                    "IsAccurate": True,
                    "TrackStatus": "1",
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                    "TyreLife": lap,
                    "Compound": "MEDIUM",
                }
            )
    return SimpleNamespace(laps=pd.DataFrame(rows))


def test_nearest_pit_label_row_uses_exact_then_nearest():
    pit_df = pd.DataFrame(
        [
            {"circuit": "Race", "driver": "A", "lap": 3, "value": "exact"},
            {"circuit": "Race", "driver": "A", "lap": 5, "value": "nearest"},
        ]
    )

    row, note = nearest_pit_label_row(pit_df, "Race", "A", 3)
    assert row["value"] == "exact"
    assert note is None

    row, note = nearest_pit_label_row(pit_df, "Race", "A", 4)
    assert row["value"] == "exact"
    assert note == "No exact pit-window row for lap 4; using nearest saved row at lap 3."


def test_current_lap_row_returns_none_for_missing_lap():
    laps = _session().laps

    assert current_lap_row(laps, "A", 2)["Driver"] == "A"
    assert current_lap_row(laps, "A", 99) is None


def test_build_strategy_context_returns_direct_rival_context():
    context, reason = build_strategy_context(_session(), "A", 5)

    assert reason is None
    assert context["driver"] == "A"
    assert context["rival"] == "B"
    assert context["compound"] == "MEDIUM"
    assert context["gap_ahead"] == 2.0
    assert context["total_laps"] == 6
    assert 0.0 < context["race_progress"] < 1.0
    assert np.isfinite(context["own_pace"])
    assert np.isfinite(context["threat_pace"])
