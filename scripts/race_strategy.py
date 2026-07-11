"""
Race-state prediction helpers used by the Streamlit app.

The module keeps pit-window encoding and undercut/overcut prediction logic
behind a small interface. Streamlit remains an adapter that supplies cached
data, model artifacts, and medians.
"""

import numpy as np
import pandas as pd

from f1_strategy_common import (
    build_gap_timeseries,
    compute_pit_loss,
    get_deg_delta,
    get_pace,
)
from strategy_features import (
    overcut_values_from_context,
    strategy_vector,
    undercut_values_from_context,
)


def encode_pit_row(row, pit_artifact, medians):
    model = pit_artifact["model"]
    encoders = pit_artifact["encoders"]
    features = list(model.feature_names_in_)
    values = {}
    issues = []
    filled = []

    for feature in features:
        if feature not in row.index:
            issues.append(f"`{feature}` missing from pit label row")
            values[feature] = medians.get(feature, np.nan)
            if pd.notna(values[feature]):
                filled.append(feature)
            continue

        value = row[feature]
        if feature in encoders:
            encoder = encoders[feature]
            value = str(value)
            if value not in encoder.classes_:
                if "nan" in encoder.classes_:
                    value = "nan"
                else:
                    issues.append(f"`{feature}` value `{value}` not in saved encoder")
                    value = encoder.classes_[0]
            values[feature] = int(encoder.transform([value])[0])
        else:
            if pd.notna(value):
                values[feature] = float(value)
            else:
                values[feature] = medians.get(feature, np.nan)
                if pd.notna(values[feature]):
                    filled.append(feature)

    return pd.DataFrame([values], columns=features), issues, filled


def nearest_pit_label_row(pit_df, race, driver, lap):
    rows = pit_df[
        (pit_df["circuit"] == race)
        & (pit_df["driver"] == driver)
        & (pit_df["lap"] == lap)
    ]
    if not rows.empty:
        return rows.iloc[0], None

    driver_rows = pit_df[(pit_df["circuit"] == race) & (pit_df["driver"] == driver)]
    if driver_rows.empty:
        return None, "No pit-window feature row exists for this driver in the saved labels CSV."

    idx = (driver_rows["lap"] - lap).abs().idxmin()
    row = driver_rows.loc[idx]
    return row, f"No exact pit-window row for lap {lap}; using nearest saved row at lap {int(row['lap'])}."


def current_lap_row(laps, driver, lap):
    rows = laps[(laps["Driver"] == driver) & (laps["LapNumber"] == lap)]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_strategy_context(session, driver, lap):
    laps = session.laps.copy()
    laps = laps[laps["LapTime"].notna()].copy()
    total_laps = int(laps["LapNumber"].max()) if not laps.empty else 0

    if lap < 3:
        return None, "Strategy models need at least three historical laps for pace/gap trends."

    gaps_df = build_gap_timeseries(session)
    gap_rows = gaps_df[(gaps_df["Driver"] == driver) & (gaps_df["LapNumber"] == lap)]
    if gap_rows.empty:
        return None, "No clean same-lap gap row exists for this driver/lap."

    gap_row = gap_rows.iloc[0]
    rival = gap_row.get("car_ahead_driver")
    gap_ahead = gap_row.get("gap_ahead")
    if pd.isna(rival) or rival is None:
        return None, "Not applicable: selected driver has no car directly ahead at this lap."
    if pd.isna(gap_ahead) or gap_ahead <= 0 or gap_ahead > 30:
        return None, "Not applicable: direct rival gap is missing or outside the strategy model range."

    driver_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")
    rival_laps = laps[laps["Driver"] == rival].sort_values("LapNumber")
    selected = current_lap_row(laps, driver, lap)
    rival_selected = current_lap_row(laps, rival, lap)
    if selected is None or rival_selected is None:
        return None, "Not applicable: selected driver or rival has no lap record at this lap."

    pit_loss = compute_pit_loss(session)
    own_pace = get_pace(driver_laps, lap + 1)
    threat_pace = get_pace(rival_laps, lap + 1)
    deg_delta = get_deg_delta(driver_laps, lap + 1)
    ca_deg_delta = get_deg_delta(rival_laps, lap + 1)

    recent_driver_gaps = (
        gaps_df[
            (gaps_df["Driver"] == driver)
            & (gaps_df["LapNumber"] >= lap - 3)
            & (gaps_df["LapNumber"] <= lap)
        ]
        .sort_values("LapNumber")
        .dropna(subset=["gap_ahead"])
    )
    closing_undercut = (
        float(np.polyfit(recent_driver_gaps["LapNumber"], recent_driver_gaps["gap_ahead"], 1)[0])
        if len(recent_driver_gaps) >= 2
        else np.nan
    )

    recent_rival_gaps = (
        gaps_df[
            (gaps_df["Driver"] == rival)
            & (gaps_df["LapNumber"] >= lap - 3)
            & (gaps_df["LapNumber"] <= lap)
        ]
        .sort_values("LapNumber")
        .dropna(subset=["gap_behind"])
    )
    closing_overcut = (
        float(np.polyfit(recent_rival_gaps["LapNumber"], recent_rival_gaps["gap_behind"], 1)[0])
        if len(recent_rival_gaps) >= 2
        else np.nan
    )

    tire_age = selected.get("TyreLife", np.nan)
    rival_tire_age = rival_selected.get("TyreLife", np.nan)
    compound = str(selected.get("Compound", "UNKNOWN"))
    pace_delta = own_pace - threat_pace if pd.notna(own_pace) and pd.notna(threat_pace) else np.nan
    pit_loss_fraction = pit_loss / own_pace if pd.notna(own_pace) and own_pace > 0 else np.nan

    return {
        "driver": driver,
        "rival": rival,
        "lap": lap,
        "total_laps": total_laps,
        "compound": compound,
        "tire_age": float(tire_age) if pd.notna(tire_age) else np.nan,
        "rival_tire_age": float(rival_tire_age) if pd.notna(rival_tire_age) else np.nan,
        "gap_ahead": float(gap_ahead),
        "own_pace": own_pace,
        "threat_pace": threat_pace,
        "pace_delta": pace_delta,
        "deg_delta": deg_delta,
        "ca_deg_delta": ca_deg_delta,
        "pit_loss": pit_loss,
        "pit_loss_fraction": pit_loss_fraction,
        "race_progress": lap / total_laps if total_laps else np.nan,
        "closing_undercut": closing_undercut,
        "closing_overcut": closing_overcut,
    }, None


def predict_strategy(context, artifacts, medians):
    undercut_features = artifacts["undercut"]["features"]
    undercut_x, undercut_row = strategy_vector(
        undercut_features, undercut_values_from_context(context), medians["undercut"]
    )

    overcut_features = artifacts["overcut"]["features"]
    overcut_x, overcut_row = strategy_vector(
        overcut_features, overcut_values_from_context(context), medians["overcut"]
    )

    undercut_prob = artifacts["undercut"]["model"].predict_proba(
        undercut_x.values.astype(float)
    )[0, 1]
    overcut_prob = artifacts["overcut"]["model"].predict_proba(
        overcut_x.values.astype(float)
    )[0, 1]
    return float(undercut_prob), float(overcut_prob), undercut_row, overcut_row
