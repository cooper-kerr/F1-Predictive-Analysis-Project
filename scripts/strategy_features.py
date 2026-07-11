"""
Feature preparation for undercut and overcut strategy models.

The saved model artifacts still expose the legacy contract:
{"model": fitted_model, "features": feature_names}. This module centralizes
the DataFrame and scenario-vector preparation around that contract.
"""

import numpy as np
import pandas as pd

UNDERCUT_BASE_FEATURES = [
    "gap_ahead",
    "tire_age",
    "car_ahead_tire_age",
    "tire_age_advantage",
    "own_pace",
    "threat_pace",
    "pace_delta",
    "deg_delta",
    "ca_deg_delta",
    "closing_rate",
    "pit_loss",
    "pit_loss_fraction",
    "race_progress",
]
UNDERCUT_REQUIRED = [
    "gap_ahead",
    "tire_age",
    "car_ahead_tire_age",
    "own_pace",
    "threat_pace",
]
UNDERCUT_TARGET = "undercut_success"

OVERCUT_BASE_FEATURES = [
    "gap_ahead",
    "tire_age",
    "ca_tire_age",
    "tire_age_delta",
    "own_pace",
    "threat_pace",
    "pace_delta",
    "deg_delta",
    "ca_deg_delta",
    "closing_rate",
    "pit_loss",
    "pit_loss_fraction",
    "race_progress",
]
OVERCUT_REQUIRED = [
    "gap_ahead",
    "tire_age",
    "ca_tire_age",
    "own_pace",
    "threat_pace",
]
OVERCUT_TARGET = "overcut_success"


def _schema(strategy):
    if strategy == "undercut":
        return UNDERCUT_BASE_FEATURES, UNDERCUT_REQUIRED, UNDERCUT_TARGET
    if strategy == "overcut":
        return OVERCUT_BASE_FEATURES, OVERCUT_REQUIRED, OVERCUT_TARGET
    raise ValueError(f"Unknown strategy: {strategy}")


def prepare_strategy_frame(df, strategy, features=None):
    base_features, required, target = _schema(strategy)
    compound_dummies = pd.get_dummies(df["compound"], prefix="compound")
    df_feat = pd.concat([df, compound_dummies], axis=1)
    feature_names = list(features) if features is not None else (
        base_features + [c for c in compound_dummies.columns]
    )

    df_model = df_feat.dropna(subset=required + [target]).copy()
    for col in feature_names:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())
        else:
            df_model[col] = 0

    return df_model, feature_names


def strategy_feature_medians(df, strategy):
    _, _, target = _schema(strategy)
    compound_dummies = pd.get_dummies(df["compound"], prefix="compound")
    df_feat = pd.concat([df, compound_dummies], axis=1)
    numeric = (
        df_feat.drop(columns=[target], errors="ignore")
        .select_dtypes(include=[np.number])
    )
    return numeric.median(numeric_only=True).to_dict()


def strategy_vector(features, base_values, medians):
    row = {feature: 0.0 for feature in features}
    for feature, value in base_values.items():
        if feature in row:
            row[feature] = value
    for feature in features:
        if pd.isna(row[feature]):
            row[feature] = medians.get(feature, 0.0)
    return pd.DataFrame([[row[feature] for feature in features]], columns=features), row


def undercut_values(
    gap_ahead,
    tire_age,
    car_ahead_tire_age,
    own_pace,
    threat_pace,
    deg_delta,
    ca_deg_delta,
    closing_rate,
    pit_loss,
    race_progress,
    compound,
):
    tire_age_advantage = car_ahead_tire_age - tire_age
    pace_delta = own_pace - threat_pace
    pit_loss_fraction = pit_loss / own_pace if own_pace > 0 else np.nan
    return {
        "gap_ahead": gap_ahead,
        "tire_age": float(tire_age),
        "car_ahead_tire_age": float(car_ahead_tire_age),
        "tire_age_advantage": float(tire_age_advantage),
        "own_pace": own_pace,
        "threat_pace": threat_pace,
        "pace_delta": pace_delta,
        "deg_delta": deg_delta,
        "ca_deg_delta": ca_deg_delta,
        "closing_rate": closing_rate,
        "pit_loss": pit_loss,
        "pit_loss_fraction": pit_loss_fraction,
        "race_progress": race_progress,
        f"compound_{compound}": 1.0,
    }


def overcut_values(
    gap_ahead,
    tire_age,
    ca_tire_age,
    own_pace,
    threat_pace,
    deg_delta,
    ca_deg_delta,
    closing_rate,
    pit_loss,
    race_progress,
    compound,
):
    tire_age_delta = tire_age - ca_tire_age
    pace_delta = own_pace - threat_pace
    pit_loss_fraction = pit_loss / own_pace if own_pace > 0 else np.nan
    return {
        "gap_ahead": gap_ahead,
        "tire_age": float(tire_age),
        "ca_tire_age": float(ca_tire_age),
        "tire_age_delta": float(tire_age_delta),
        "own_pace": own_pace,
        "threat_pace": threat_pace,
        "pace_delta": pace_delta,
        "deg_delta": deg_delta,
        "ca_deg_delta": ca_deg_delta,
        "closing_rate": closing_rate,
        "pit_loss": pit_loss,
        "pit_loss_fraction": pit_loss_fraction,
        "race_progress": race_progress,
        f"compound_{compound}": 1.0,
    }


def undercut_values_from_context(context):
    tire_age = context["tire_age"]
    rival_tire_age = context["rival_tire_age"]
    return {
        "gap_ahead": context["gap_ahead"],
        "tire_age": tire_age,
        "car_ahead_tire_age": rival_tire_age,
        "tire_age_advantage": (
            rival_tire_age - tire_age
            if pd.notna(rival_tire_age) and pd.notna(tire_age)
            else np.nan
        ),
        "own_pace": context["own_pace"],
        "threat_pace": context["threat_pace"],
        "pace_delta": context["pace_delta"],
        "deg_delta": context["deg_delta"],
        "ca_deg_delta": context["ca_deg_delta"],
        "closing_rate": context["closing_undercut"],
        "pit_loss": context["pit_loss"],
        "pit_loss_fraction": context["pit_loss_fraction"],
        "race_progress": context["race_progress"],
        f"compound_{context['compound']}": 1.0,
    }


def overcut_values_from_context(context):
    tire_age = context["tire_age"]
    rival_tire_age = context["rival_tire_age"]
    return {
        "gap_ahead": context["gap_ahead"],
        "tire_age": tire_age,
        "ca_tire_age": rival_tire_age,
        "tire_age_delta": (
            tire_age - rival_tire_age
            if pd.notna(rival_tire_age) and pd.notna(tire_age)
            else np.nan
        ),
        "own_pace": context["own_pace"],
        "threat_pace": context["threat_pace"],
        "pace_delta": context["pace_delta"],
        "deg_delta": context["deg_delta"],
        "ca_deg_delta": context["ca_deg_delta"],
        "closing_rate": context["closing_overcut"],
        "pit_loss": context["pit_loss"],
        "pit_loss_fraction": context["pit_loss_fraction"],
        "race_progress": context["race_progress"],
        f"compound_{context['compound']}": 1.0,
    }
