from types import SimpleNamespace

import numpy as np
import pandas as pd

from strategy_attempts import build_overcut_records, build_undercut_records


def _time(seconds):
    return pd.to_timedelta(seconds, unit="s")


def _pit_in_for(driver, lap, pit_laps):
    return _time(1000 + lap * 90) if pit_laps.get(driver) == lap else pd.NaT


def _pit_out_for(driver, lap, pit_laps):
    out_lap = pit_laps.get(driver, 0) + 1
    return _time(1022 + (lap - 1) * 90) if out_lap == lap else pd.NaT


def _session(pit_laps, eval_order):
    rows = []
    total_laps = 11
    for lap in range(1, total_laps + 1):
        for driver in ["B", "A"]:
            if lap == total_laps:
                order = eval_order
                offset = order.index(driver) * 2
            else:
                offset = 0 if driver == "B" else 2

            rows.append(
                {
                    "Driver": driver,
                    "LapNumber": lap,
                    "LapTime": _time(90 + lap * 0.1 + (0.2 if driver == "A" else 0)),
                    "LapStartTime": _time(lap * 100 + offset),
                    "IsAccurate": True,
                    "TrackStatus": "1",
                    "PitInTime": _pit_in_for(driver, lap, pit_laps),
                    "PitOutTime": _pit_out_for(driver, lap, pit_laps),
                    "TyreLife": lap + (1 if driver == "B" else 0),
                    "Compound": "MEDIUM",
                }
            )
    return SimpleNamespace(laps=pd.DataFrame(rows))


def test_build_undercut_records_preserves_legacy_output_shape():
    session = _session(pit_laps={"A": 5, "B": 7}, eval_order=["A", "B"])

    records = build_undercut_records(session, 2024, "Synthetic GP")

    assert len(records) == 1
    record = records[0]
    assert list(record) == [
        "year",
        "circuit",
        "driver",
        "car_ahead",
        "pit_lap",
        "gap_ahead",
        "tire_age",
        "car_ahead_tire_age",
        "tire_age_advantage",
        "compound",
        "own_pace",
        "threat_pace",
        "pace_delta",
        "deg_delta",
        "ca_deg_delta",
        "closing_rate",
        "pit_loss",
        "pit_loss_fraction",
        "race_progress",
        "undercut_success",
    ]
    assert record["driver"] == "A"
    assert record["car_ahead"] == "B"
    assert record["pit_lap"] == 5
    assert record["gap_ahead"] == 2.0
    assert record["tire_age_advantage"] == 1.0
    assert record["compound"] == "MEDIUM"
    assert record["undercut_success"] == 1
    assert np.isfinite(record["own_pace"])
    assert np.isfinite(record["pit_loss_fraction"])


def test_build_overcut_records_preserves_legacy_output_shape():
    session = _session(pit_laps={"B": 5, "A": 7}, eval_order=["A", "B"])

    records = build_overcut_records(session, 2024, "Synthetic GP")

    assert len(records) == 1
    record = records[0]
    assert list(record) == [
        "year",
        "circuit",
        "stay_out_driver",
        "pitting_car",
        "pit_lap",
        "stay_out_laps",
        "gap_ahead",
        "tire_age",
        "ca_tire_age",
        "tire_age_delta",
        "compound",
        "own_pace",
        "threat_pace",
        "pace_delta",
        "deg_delta",
        "ca_deg_delta",
        "closing_rate",
        "pit_loss",
        "pit_loss_fraction",
        "race_progress",
        "overcut_success",
    ]
    assert record["stay_out_driver"] == "A"
    assert record["pitting_car"] == "B"
    assert record["pit_lap"] == 5
    assert record["stay_out_laps"] == 2
    assert record["gap_ahead"] == 2.0
    assert record["tire_age_delta"] == -1.0
    assert record["compound"] == "MEDIUM"
    assert record["overcut_success"] == 1
    assert np.isfinite(record["own_pace"])
    assert np.isfinite(record["pit_loss_fraction"])
