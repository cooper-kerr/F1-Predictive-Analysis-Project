import numpy as np
import pandas as pd

from strategy_features import (
    overcut_values,
    prepare_strategy_frame,
    strategy_vector,
    undercut_values,
)


def test_prepare_undercut_frame_preserves_feature_order_and_fill_policy():
    df = pd.DataFrame(
        [
            {
                "compound": "MEDIUM",
                "gap_ahead": 2.0,
                "tire_age": 10.0,
                "car_ahead_tire_age": 12.0,
                "tire_age_advantage": 2.0,
                "own_pace": 90.0,
                "threat_pace": 91.0,
                "pace_delta": -1.0,
                "deg_delta": np.nan,
                "ca_deg_delta": 0.1,
                "closing_rate": -0.2,
                "pit_loss": 22.0,
                "pit_loss_fraction": 22.0 / 90.0,
                "race_progress": 0.5,
                "undercut_success": 1,
            },
            {
                "compound": "SOFT",
                "gap_ahead": np.nan,
                "tire_age": 9.0,
                "car_ahead_tire_age": 11.0,
                "tire_age_advantage": 2.0,
                "own_pace": 89.0,
                "threat_pace": 90.0,
                "pace_delta": -1.0,
                "deg_delta": 0.2,
                "ca_deg_delta": 0.3,
                "closing_rate": -0.1,
                "pit_loss": 22.0,
                "pit_loss_fraction": 22.0 / 89.0,
                "race_progress": 0.4,
                "undercut_success": 0,
            },
        ]
    )

    model_df, features = prepare_strategy_frame(df, "undercut")

    assert features == [
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
        "compound_MEDIUM",
        "compound_SOFT",
    ]
    assert len(model_df) == 1
    assert pd.isna(model_df.iloc[0]["deg_delta"])


def test_prepare_overcut_frame_uses_saved_feature_contract():
    df = pd.DataFrame(
        [
            {
                "compound": "HARD",
                "gap_ahead": 3.0,
                "tire_age": 15.0,
                "ca_tire_age": 12.0,
                "tire_age_delta": 3.0,
                "own_pace": 90.0,
                "threat_pace": 91.0,
                "pace_delta": -1.0,
                "deg_delta": 0.1,
                "ca_deg_delta": 0.2,
                "closing_rate": -0.1,
                "pit_loss": 22.0,
                "pit_loss_fraction": 22.0 / 90.0,
                "race_progress": 0.5,
                "overcut_success": 1,
            }
        ]
    )
    saved_features = ["gap_ahead", "compound_MEDIUM", "compound_HARD"]

    model_df, features = prepare_strategy_frame(df, "overcut", saved_features)

    assert features == saved_features
    assert model_df[saved_features].iloc[0].to_dict() == {
        "gap_ahead": 3.0,
        "compound_MEDIUM": 0.0,
        "compound_HARD": True,
    }


def test_strategy_vectors_build_legacy_undercut_and_overcut_fields():
    undercut_base = undercut_values(
        2.0, 10, 13, 90.0, 91.0, 0.1, 0.2, -0.3, 22.0, 0.5, "MEDIUM"
    )
    overcut_base = overcut_values(
        2.0, 13, 10, 90.0, 91.0, 0.1, 0.2, -0.3, 22.0, 0.5, "MEDIUM"
    )

    undercut_x, undercut_row = strategy_vector(
        ["tire_age_advantage", "pace_delta", "compound_MEDIUM"],
        undercut_base,
        {},
    )
    overcut_x, overcut_row = strategy_vector(
        ["tire_age_delta", "pace_delta", "compound_MEDIUM"],
        overcut_base,
        {},
    )

    assert undercut_row == {
        "tire_age_advantage": 3.0,
        "pace_delta": -1.0,
        "compound_MEDIUM": 1.0,
    }
    assert overcut_row == {
        "tire_age_delta": 3.0,
        "pace_delta": -1.0,
        "compound_MEDIUM": 1.0,
    }
    assert undercut_x.values.tolist() == [[3.0, -1.0, 1.0]]
    assert overcut_x.values.tolist() == [[3.0, -1.0, 1.0]]
