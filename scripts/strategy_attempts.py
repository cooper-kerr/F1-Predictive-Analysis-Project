"""
Strategy attempt extraction for undercut and overcut datasets.

This module owns the shared lap mechanics for rival-focused strategy attempts:
pit-stop event selection, decision-lap lookup, safety-car filtering,
evaluation-order lookup, and common pace/degradation features.
"""

import numpy as np
import pandas as pd

from f1_strategy_common import (
    build_gap_timeseries,
    compute_pit_loss,
    get_deg_delta,
    get_pace,
    has_sc_between,
)

STABILIZATION = 4
GAP_WINDOW = 3
MAX_GAP = 30.0
MIN_HIST_LAPS = 3


def _first_row(value):
    if isinstance(value, pd.DataFrame):
        return value.iloc[0]
    return value


def _race_order(laps, eval_lap):
    eval_laps_session = (
        laps[laps["LapNumber"] == eval_lap][["Driver", "LapStartTime"]]
        .dropna()
        .sort_values("LapStartTime")
        .reset_index(drop=True)
    )
    if eval_laps_session.empty:
        return None
    return {row["Driver"]: idx for idx, row in eval_laps_session.iterrows()}


def _closing_rate(gaps_df, driver, gap_col, dec_lap):
    recent_gaps = (
        gaps_df[
            (gaps_df["Driver"] == driver)
            & (gaps_df["LapNumber"] >= dec_lap - GAP_WINDOW)
            & (gaps_df["LapNumber"] <= dec_lap)
        ]
        .sort_values("LapNumber")
        .dropna(subset=[gap_col])
    )
    if len(recent_gaps) < 2:
        return np.nan
    return float(
        np.polyfit(
            recent_gaps["LapNumber"].values,
            recent_gaps[gap_col].values,
            1,
        )[0]
    )


def _pace_features(laps, driver, rival, dec_lap):
    driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
    rival_laps = laps[laps["Driver"] == rival].sort_values("LapNumber")

    own_pace = get_pace(driver_laps, dec_lap + 1)
    threat_pace = get_pace(rival_laps, dec_lap + 1)
    deg_delta = get_deg_delta(driver_laps, dec_lap + 1)
    ca_deg_delta = get_deg_delta(rival_laps, dec_lap + 1)
    pace_delta = (
        own_pace - threat_pace
        if not pd.isna(own_pace) and not pd.isna(threat_pace)
        else np.nan
    )
    return own_pace, threat_pace, pace_delta, deg_delta, ca_deg_delta


def _pit_loss_fraction(pit_loss, own_pace):
    if pd.isna(own_pace) or own_pace <= 0:
        return np.nan
    return pit_loss / own_pace


def _session_context(session):
    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()].copy()
    total_laps = int(laps["LapNumber"].max())
    gaps_df = build_gap_timeseries(session)
    return laps, total_laps, gaps_df, compute_pit_loss(session)


def build_undercut_records(session, year, circuit_name):
    laps, total_laps, gaps_df, pit_loss = _session_context(session)
    gaps_idx = gaps_df.set_index(["Driver", "LapNumber"])
    laps_idx = laps.set_index(["Driver", "LapNumber"])
    records = []

    pit_rows = laps[laps["PitInTime"].notna()][["Driver", "LapNumber"]].copy()
    for _, pit_row in pit_rows.iterrows():
        driver = pit_row["Driver"]
        pit_lap = int(pit_row["LapNumber"])
        dec_lap = pit_lap - 1

        if dec_lap < MIN_HIST_LAPS:
            continue

        try:
            gap_row = _first_row(gaps_idx.loc[(driver, dec_lap)])
        except KeyError:
            continue

        car_ahead = gap_row["car_ahead_driver"]
        gap_ahead = gap_row["gap_ahead"]
        if car_ahead is None or pd.isna(car_ahead) or pd.isna(gap_ahead):
            continue
        if gap_ahead > MAX_GAP or gap_ahead <= 0:
            continue

        car_ahead_pits = laps[
            (laps["Driver"] == car_ahead)
            & (laps["PitInTime"].notna())
            & (laps["LapNumber"] > pit_lap)
        ]["LapNumber"]
        if car_ahead_pits.empty:
            continue

        eval_lap = min(int(car_ahead_pits.min()) + STABILIZATION, total_laps)
        if has_sc_between(laps, driver, pit_lap, eval_lap):
            continue

        order = _race_order(laps, eval_lap)
        if order is None or driver not in order or car_ahead not in order:
            continue

        try:
            dec_row = _first_row(laps_idx.loc[(driver, dec_lap)])
            ca_dec_row = _first_row(laps_idx.loc[(car_ahead, dec_lap)])
        except KeyError:
            continue

        own_pace, threat_pace, pace_delta, deg_delta, ca_deg_delta = _pace_features(
            laps, driver, car_ahead, dec_lap
        )
        tire_age = dec_row.get("TyreLife", np.nan)
        compound = dec_row.get("Compound", "UNKNOWN")
        ca_tire_age = ca_dec_row.get("TyreLife", np.nan)
        tire_age_advantage = (
            float(ca_tire_age) - float(tire_age)
            if not pd.isna(ca_tire_age) and not pd.isna(tire_age)
            else np.nan
        )

        records.append(
            {
                "year": year,
                "circuit": circuit_name,
                "driver": driver,
                "car_ahead": car_ahead,
                "pit_lap": pit_lap,
                "gap_ahead": gap_ahead,
                "tire_age": float(tire_age) if not pd.isna(tire_age) else np.nan,
                "car_ahead_tire_age": (
                    float(ca_tire_age) if not pd.isna(ca_tire_age) else np.nan
                ),
                "tire_age_advantage": tire_age_advantage,
                "compound": str(compound),
                "own_pace": own_pace,
                "threat_pace": threat_pace,
                "pace_delta": pace_delta,
                "deg_delta": deg_delta,
                "ca_deg_delta": ca_deg_delta,
                "closing_rate": _closing_rate(gaps_df, driver, "gap_ahead", dec_lap),
                "pit_loss": pit_loss,
                "pit_loss_fraction": _pit_loss_fraction(pit_loss, own_pace),
                "race_progress": dec_lap / total_laps if total_laps > 0 else np.nan,
                "undercut_success": 1 if order[driver] < order[car_ahead] else 0,
            }
        )

    return records


def build_overcut_records(session, year, circuit_name):
    laps, total_laps, gaps_df, pit_loss = _session_context(session)
    gaps_idx = gaps_df.set_index(["Driver", "LapNumber"])
    laps_idx = laps.set_index(["Driver", "LapNumber"])
    records = []

    pit_rows = laps[laps["PitInTime"].notna()][["Driver", "LapNumber"]].copy()
    for _, pit_row in pit_rows.iterrows():
        pitting_car = pit_row["Driver"]
        pit_lap = int(pit_row["LapNumber"])
        dec_lap = pit_lap - 1

        if dec_lap < MIN_HIST_LAPS:
            continue

        try:
            gap_row = _first_row(gaps_idx.loc[(pitting_car, dec_lap)])
        except KeyError:
            continue

        stay_out = gap_row["car_behind_driver"]
        gap_behind_pitter = gap_row["gap_behind"]
        if stay_out is None or pd.isna(stay_out):
            continue
        if (
            pd.isna(gap_behind_pitter)
            or gap_behind_pitter > MAX_GAP
            or gap_behind_pitter <= 0
        ):
            continue

        stay_out_pits = laps[
            (laps["Driver"] == stay_out)
            & (laps["PitInTime"].notna())
            & (laps["LapNumber"] > pit_lap)
        ]["LapNumber"]
        if stay_out_pits.empty:
            continue

        stay_out_pit_lap = int(stay_out_pits.min())
        stay_out_laps_num = stay_out_pit_lap - pit_lap
        eval_lap = min(stay_out_pit_lap + STABILIZATION, total_laps)
        if stay_out_laps_num < 1:
            continue
        if has_sc_between(laps, stay_out, pit_lap, eval_lap):
            continue

        order = _race_order(laps, eval_lap)
        if order is None or stay_out not in order or pitting_car not in order:
            continue

        try:
            so_dec = _first_row(laps_idx.loc[(stay_out, dec_lap)])
            pit_dec = _first_row(laps_idx.loc[(pitting_car, dec_lap)])
        except KeyError:
            continue

        own_pace, threat_pace, pace_delta, deg_delta, ca_deg_delta = _pace_features(
            laps, stay_out, pitting_car, dec_lap
        )
        tire_age = so_dec.get("TyreLife", np.nan)
        compound = so_dec.get("Compound", "UNKNOWN")
        ca_tire_age = pit_dec.get("TyreLife", np.nan)
        tire_age_delta = (
            float(tire_age) - float(ca_tire_age)
            if not pd.isna(tire_age) and not pd.isna(ca_tire_age)
            else np.nan
        )

        records.append(
            {
                "year": year,
                "circuit": circuit_name,
                "stay_out_driver": stay_out,
                "pitting_car": pitting_car,
                "pit_lap": pit_lap,
                "stay_out_laps": stay_out_laps_num,
                "gap_ahead": gap_behind_pitter,
                "tire_age": float(tire_age) if not pd.isna(tire_age) else np.nan,
                "ca_tire_age": (
                    float(ca_tire_age) if not pd.isna(ca_tire_age) else np.nan
                ),
                "tire_age_delta": tire_age_delta,
                "compound": str(compound),
                "own_pace": own_pace,
                "threat_pace": threat_pace,
                "pace_delta": pace_delta,
                "deg_delta": deg_delta,
                "ca_deg_delta": ca_deg_delta,
                "closing_rate": _closing_rate(
                    gaps_df, pitting_car, "gap_behind", dec_lap
                ),
                "pit_loss": pit_loss,
                "pit_loss_fraction": _pit_loss_fraction(pit_loss, own_pace),
                "race_progress": dec_lap / total_laps if total_laps > 0 else np.nan,
                "overcut_success": 1 if order[stay_out] < order[pitting_car] else 0,
            }
        )

    return records
