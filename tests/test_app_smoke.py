from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import app


SAMPLE_RACE = "Bahrain Grand Prix"
SAMPLE_DRIVER = "ALB"
SAMPLE_LAP = 5


def assert_probability(value):
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_app_artifacts_load_with_expected_schema():
    artifacts = app.load_artifacts()

    expected_artifacts = {"pit", "undercut", "overcut", "degradation", "weather"}
    assert expected_artifacts.issubset(artifacts)

    pit_artifact = artifacts["pit"]
    assert {"model", "encoders", "params"}.issubset(pit_artifact)
    assert hasattr(pit_artifact["model"], "predict")
    assert len(pit_artifact["model"].feature_names_in_) > 0
    assert isinstance(pit_artifact["encoders"], dict)

    for key in ["undercut", "overcut"]:
        strategy_artifact = artifacts[key]
        assert {"model", "features"}.issubset(strategy_artifact)
        assert hasattr(strategy_artifact["model"], "predict_proba")
        assert len(strategy_artifact["features"]) > 0

    degradation = artifacts["degradation"]
    assert len(degradation["age_bins"]) > 0
    assert len(degradation["age_labels"]) > 0
    assert not degradation["pace_profile"].empty

    weather = artifacts["weather"]
    assert weather["n_rows"] > 0
    assert np.isfinite(weather["r_squared"])
    assert not weather["coefficients"].empty

    if "position" in artifacts:
        position = artifacts["position"]
        assert not position["curve"].empty
        assert position["details"]
        assert position["baselines"]


def test_sample_pit_window_prediction_from_bundled_labels():
    pit_df = app.load_pit_labels()
    pit_artifact = app.load_artifacts()["pit"]

    row, note = app.nearest_pit_label_row(
        pit_df, SAMPLE_RACE, SAMPLE_DRIVER, SAMPLE_LAP
    )
    assert note is None
    assert row is not None

    features, issues, filled = app.encode_pit_row(
        row, pit_artifact, app.pit_feature_medians()
    )

    assert issues == []
    assert filled == []
    assert features.columns.tolist() == list(pit_artifact["model"].feature_names_in_)
    assert features.notna().all(axis=None)

    prediction = float(pit_artifact["model"].predict(features)[0])
    assert np.isfinite(prediction)
    assert 0.0 <= prediction <= 80.0


def test_sample_strategy_context_and_predictions_from_static_laps():
    session = app.load_session(SAMPLE_RACE)
    context, reason = app.build_strategy_context(session, SAMPLE_DRIVER, SAMPLE_LAP)

    assert reason is None
    assert context is not None
    assert context["driver"] == SAMPLE_DRIVER
    assert context["rival"] == "TSU"
    assert context["lap"] == SAMPLE_LAP
    assert context["total_laps"] == 57
    assert context["compound"] == "SOFT"
    assert context["gap_ahead"] > 0
    assert context["pit_loss"] > 0
    assert 0.0 < context["race_progress"] < 1.0

    undercut_prob, overcut_prob, undercut_row, overcut_row = app.predict_strategy(
        context, app.load_artifacts()
    )

    assert_probability(undercut_prob)
    assert_probability(overcut_prob)
    assert undercut_row
    assert overcut_row
    assert all(np.isfinite(value) for value in undercut_row.values())
    assert all(np.isfinite(value) for value in overcut_row.values())


def test_position_comparison_evidence_artifacts_are_available():
    root = Path(app.ROOT)
    curve_path = root / "data" / "f1_position_comparison_curve.csv"
    model_path = root / "models" / "f1_position_comparison_analysis.pkl"
    figure_path = root / "outputs" / "figures" / "position_comparison_curve.png"

    assert curve_path.exists()
    assert model_path.exists()
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0

    curve = pd.read_csv(curve_path)
    expected_columns = {
        "year",
        "checkpoint_lap",
        "accuracy",
        "std",
        "se",
        "n_folds",
        "n_rows",
    }
    assert expected_columns.issubset(curve.columns)
    assert not curve.empty
    assert curve["accuracy"].between(0.0, 1.0).all()
    assert (curve["n_rows"] > 0).all()

    artifact = joblib.load(model_path)
    assert artifact["position_included"] is False
    assert artifact["evaluation"] == "leave-one-race-out GroupKFold by GrandPrix"
    assert set(expected_columns).issubset(artifact["curve"].columns)
    assert artifact["details"]
    assert artifact["baselines"]
    assert Path(artifact["figure_path"]).name == figure_path.name
