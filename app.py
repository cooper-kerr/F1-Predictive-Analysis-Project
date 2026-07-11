from pathlib import Path
import sys

import fastf1
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
CACHE_DIR = ROOT / "f1-cache"

sys.path.append(str(ROOT / "scripts"))
from f1_strategy_common import (  # noqa: E402
    build_gap_timeseries,
    compute_pit_loss,
    get_deg_delta,
    get_pace,
)


CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


@st.cache_data
def load_pit_labels():
    df = pd.read_csv(DATA_DIR / "f1_pit_window_labels.csv")
    return df[df["year"] == 2024].copy()


@st.cache_resource
def load_artifacts():
    return {
        "pit": joblib.load(MODEL_DIR / "f1_pit_window_model_tuned.pkl"),
        "undercut": joblib.load(MODEL_DIR / "f1_undercut_model.pkl"),
        "overcut": joblib.load(MODEL_DIR / "f1_overcut_model.pkl"),
        "degradation": joblib.load(MODEL_DIR / "f1_degradation_analysis.pkl"),
        "weather": joblib.load(MODEL_DIR / "f1_weather_analysis.pkl"),
    }


@st.cache_resource(show_spinner="Loading cached FastF1 race data...")
def load_session(race_name):
    session = fastf1.get_session(2024, race_name, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    return session


@st.cache_data
def strategy_medians():
    out = {}
    for key, path, target in [
        ("undercut", DATA_DIR / "f1_undercut_dataset.csv", "undercut_success"),
        ("overcut", DATA_DIR / "f1_overcut_dataset.csv", "overcut_success"),
    ]:
        df = pd.read_csv(path)
        dummies = pd.get_dummies(df["compound"], prefix="compound")
        df = pd.concat([df, dummies], axis=1)
        numeric = df.drop(columns=[target], errors="ignore").select_dtypes(include=[np.number])
        out[key] = numeric.median(numeric_only=True).to_dict()
    return out


@st.cache_data
def load_degradation_support():
    df = pd.read_csv(DATA_DIR / "f1_degradation_dataset.csv")
    degradation_artifact = joblib.load(MODEL_DIR / "f1_degradation_analysis.pkl")
    df = df[df["compound"].isin(["HARD", "MEDIUM", "SOFT"])].copy()
    df["era"] = np.where(df["year"] <= 2023, "2022-2023", "2024")
    df["age_bin"] = pd.cut(
        df["tire_age"],
        bins=degradation_artifact["age_bins"],
        labels=degradation_artifact["age_labels"],
        right=False,
        include_lowest=True,
    )
    df["stint_id"] = (
        df["year"].astype(str)
        + "|"
        + df["circuit"].astype(str)
        + "|"
        + df["driver"].astype(str)
        + "|"
        + df["stint"].astype(str)
        + "|"
        + df["compound"].astype(str)
    )
    return (
        df.dropna(subset=["age_bin"])
        .groupby(["compound", "era", "age_bin"], observed=True)
        .agg(
            n_laps=("delta_pace", "size"),
            n_stints=("stint_id", "nunique"),
            mean_delta_s=("delta_pace", "mean"),
            median_delta_s=("delta_pace", "median"),
        )
        .reset_index()
    )


def age_bin_for_tire_age(tire_age, age_labels):
    if pd.isna(tire_age):
        return None
    age = float(tire_age)
    if age < 5:
        return "0-5"
    if age < 10:
        return "5-10"
    if age < 15:
        return "10-15"
    if age < 20:
        return "15-20"
    if age < 25:
        return "20-25"
    if age < 30:
        return "25-30"
    return "30+" if "30+" in age_labels else None


def encode_pit_row(row, pit_artifact):
    model = pit_artifact["model"]
    encoders = pit_artifact["encoders"]
    features = list(model.feature_names_in_)
    values = {}
    issues = []

    for feature in features:
        if feature not in row.index:
            values[feature] = np.nan
            issues.append(f"`{feature}` missing from pit label row")
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
            values[feature] = float(value) if pd.notna(value) else np.nan

    return pd.DataFrame([values], columns=features), issues


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

    recent_driver_gaps = gaps_df[
        (gaps_df["Driver"] == driver)
        & (gaps_df["LapNumber"] >= lap - 3)
        & (gaps_df["LapNumber"] <= lap)
    ].sort_values("LapNumber").dropna(subset=["gap_ahead"])
    closing_undercut = (
        float(np.polyfit(recent_driver_gaps["LapNumber"], recent_driver_gaps["gap_ahead"], 1)[0])
        if len(recent_driver_gaps) >= 2
        else np.nan
    )

    recent_rival_gaps = gaps_df[
        (gaps_df["Driver"] == rival)
        & (gaps_df["LapNumber"] >= lap - 3)
        & (gaps_df["LapNumber"] <= lap)
    ].sort_values("LapNumber").dropna(subset=["gap_behind"])
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

    context = {
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
    }
    return context, None


def strategy_vector(features, base_values, medians):
    row = {feature: 0.0 for feature in features}
    for feature, value in base_values.items():
        if feature in row:
            row[feature] = value
    for feature in features:
        if pd.isna(row[feature]):
            row[feature] = medians.get(feature, 0.0)
    return pd.DataFrame([[row[feature] for feature in features]], columns=features), row


def predict_strategy(context, artifacts):
    medians = strategy_medians()
    compound = context["compound"]

    undercut_features = artifacts["undercut"]["features"]
    undercut_base = {
        "gap_ahead": context["gap_ahead"],
        "tire_age": context["tire_age"],
        "car_ahead_tire_age": context["rival_tire_age"],
        "tire_age_advantage": context["rival_tire_age"] - context["tire_age"]
        if pd.notna(context["rival_tire_age"]) and pd.notna(context["tire_age"])
        else np.nan,
        "own_pace": context["own_pace"],
        "threat_pace": context["threat_pace"],
        "pace_delta": context["pace_delta"],
        "deg_delta": context["deg_delta"],
        "ca_deg_delta": context["ca_deg_delta"],
        "closing_rate": context["closing_undercut"],
        "pit_loss": context["pit_loss"],
        "pit_loss_fraction": context["pit_loss_fraction"],
        "race_progress": context["race_progress"],
        f"compound_{compound}": 1.0,
    }
    undercut_x, undercut_row = strategy_vector(
        undercut_features, undercut_base, medians["undercut"]
    )

    overcut_features = artifacts["overcut"]["features"]
    overcut_base = {
        "gap_ahead": context["gap_ahead"],
        "tire_age": context["tire_age"],
        "ca_tire_age": context["rival_tire_age"],
        "tire_age_delta": context["tire_age"] - context["rival_tire_age"]
        if pd.notna(context["rival_tire_age"]) and pd.notna(context["tire_age"])
        else np.nan,
        "own_pace": context["own_pace"],
        "threat_pace": context["threat_pace"],
        "pace_delta": context["pace_delta"],
        "deg_delta": context["deg_delta"],
        "ca_deg_delta": context["ca_deg_delta"],
        "closing_rate": context["closing_overcut"],
        "pit_loss": context["pit_loss"],
        "pit_loss_fraction": context["pit_loss_fraction"],
        "race_progress": context["race_progress"],
        f"compound_{compound}": 1.0,
    }
    overcut_x, overcut_row = strategy_vector(
        overcut_features, overcut_base, medians["overcut"]
    )

    undercut_prob = artifacts["undercut"]["model"].predict_proba(undercut_x.values.astype(float))[0, 1]
    overcut_prob = artifacts["overcut"]["model"].predict_proba(overcut_x.values.astype(float))[0, 1]
    return float(undercut_prob), float(overcut_prob), undercut_row, overcut_row


def render_degradation(compound, tire_age, artifact):
    support = load_degradation_support()
    labels = artifact["age_labels"]
    subset = support[support["compound"] == compound].copy()
    if subset.empty:
        st.info(f"No saved degradation pace profile for compound `{compound}`.")
        return

    subset["age_bin"] = pd.Categorical(subset["age_bin"], categories=labels, ordered=True)
    subset = subset.sort_values(["era", "age_bin"])
    current_bin = age_bin_for_tire_age(tire_age, labels)
    current_x = labels.index(current_bin) if current_bin in labels else None

    fig, (ax, ax_count) = plt.subplots(
        2,
        1,
        figsize=(7.5, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    colors = {"2022-2023": "#667085", "2024": "#C43B2F"}
    for era, group in subset.groupby("era"):
        group = group.sort_values("age_bin")
        x = [labels.index(str(age_bin)) for age_bin in group["age_bin"]]
        y = group["median_delta_s"].to_numpy()
        low_support = group["n_stints"].to_numpy() < 15
        color = colors.get(era, None)
        ax.plot(x, y, marker="o", color=color, label=f"{era} median delta")
        if low_support.any():
            ax.scatter(
                np.array(x)[low_support],
                y[low_support],
                s=110,
                facecolors="white",
                edgecolors=color,
                linewidths=2,
                zorder=4,
                label=f"{era} <15 stints",
            )

    era_2024 = subset[subset["era"] == "2024"].sort_values("age_bin")
    if current_x is not None and not era_2024.empty:
        current_row = era_2024[era_2024["age_bin"].astype(str) == current_bin]
        if not current_row.empty:
            y = float(current_row.iloc[0]["median_delta_s"])
            ax.axvline(current_x, color="#1D2939", linestyle=":", linewidth=1.5)
            ax.scatter(
                [current_x],
                [y],
                marker="*",
                s=230,
                color="#1D2939",
                zorder=5,
                label=f"You are here ({current_bin})",
            )

    ax.set_title(f"{compound} saved degradation pace profile")
    ax.set_ylabel("Median delta vs stint best (s/lap)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    width = 0.36
    offsets = {"2022-2023": -width / 2, "2024": width / 2}
    for era, group in subset.groupby("era"):
        group = group.sort_values("age_bin")
        x = np.array([labels.index(str(age_bin)) for age_bin in group["age_bin"]])
        counts = group["n_stints"].to_numpy()
        colors_for_bars = np.where(counts < 15, "#F79009", colors.get(era, "#98A2B3"))
        bars = ax_count.bar(
            x + offsets.get(era, 0),
            counts,
            width=width,
            color=colors_for_bars,
            alpha=0.75,
            label=f"{era} stints",
        )
        for bar, count in zip(bars, counts):
            ax_count.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if count >= 100 else 0,
            )

    if current_x is not None:
        ax_count.axvline(current_x, color="#1D2939", linestyle=":", linewidth=1.5)
    ax_count.axhline(15, color="#F79009", linestyle="--", linewidth=1)
    ax_count.set_ylabel("Stints")
    ax_count.set_xlabel("Tyre age bin")
    ax_count.set_xticks(range(len(labels)))
    ax_count.set_xticklabels(labels, rotation=0)
    ax_count.legend(loc="upper right", fontsize=8)
    ax_count.grid(True, axis="y", alpha=0.25)

    st.pyplot(fig, clear_figure=True)

    support_display = subset[
        ["era", "age_bin", "n_stints", "n_laps", "median_delta_s", "mean_delta_s"]
    ].copy()
    support_display["median_delta_s"] = support_display["median_delta_s"].round(3)
    support_display["mean_delta_s"] = support_display["mean_delta_s"].round(3)
    st.dataframe(support_display, width="stretch", hide_index=True)

    weak = support_display[support_display["n_stints"] < 15]
    if not weak.empty:
        weak_bins = ", ".join(
            f"{row.era} {row.age_bin} (n={int(row.n_stints)})"
            for row in weak.itertuples(index=False)
        )
        st.caption(
            f"Open markers/orange bars flag bins with fewer than 15 distinct stints: {weak_bins}."
        )
    st.caption(
        f"Selected lap tyre age: {tire_age if pd.notna(tire_age) else 'unknown'} laps. "
        f"Current 2024 age bin: {current_bin or 'unknown'}. "
        "This chart is descriptive/retrospective, not a live prediction."
    )


def render_weather(artifact, compound):
    coefs = artifact["coefficients"].copy()
    st.write(
        f"OLS explanatory regression, R² = {artifact['r_squared']:.3f}, "
        f"rows = {artifact['n_rows']:,}."
    )
    temp_rows = coefs[
        coefs["term"].str.contains("TrackTemp|AirTemp|Rainfall|TyreLife|LapNumber", case=False, regex=True)
    ]
    if not temp_rows.empty:
        st.dataframe(temp_rows, width="stretch", hide_index=True)
    else:
        st.dataframe(coefs, width="stretch", hide_index=True)

    track_temp = coefs[coefs["term"].str.contains("TrackTemp", case=False, regex=True)]
    if not track_temp.empty:
        term = track_temp.iloc[0]
        st.caption(
            f"Reference: `{term['term']}` coefficient is {term['coef']:.4f} s/lap "
            f"holding the fitted formula terms constant. Compound shown above: {compound}."
        )
    else:
        st.caption("No explicit TrackTemp coefficient was found in the saved coefficient table.")


def main():
    st.set_page_config(page_title="F1 Predictive Analysis Demo", layout="wide")
    st.title("F1 Predictive Analysis Demo")
    st.warning(
        "Interactive demo -- Pit Window, Undercut, and Overcut are live predictions "
        "from trained models. Degradation and Weather sections show saved "
        "descriptive/explanatory analysis for context."
    )

    pit_df = load_pit_labels()
    artifacts = load_artifacts()

    races = sorted(pit_df["circuit"].dropna().unique())
    with st.sidebar:
        st.header("Race State")
        race = st.selectbox("Race", races, index=0)
        race_rows = pit_df[pit_df["circuit"] == race]
        drivers = sorted(race_rows["driver"].dropna().unique())
        driver = st.selectbox("Driver", drivers, index=0)
        driver_rows = race_rows[race_rows["driver"] == driver]
        min_lap = int(max(1, driver_rows["lap"].min()))
        max_lap = int(driver_rows["lap"].max())
        lap = st.slider("Lap", min_value=min_lap, max_value=max_lap, value=min_lap)

    session = load_session(race)
    laps = session.laps.copy()
    total_laps = int(laps["LapNumber"].max()) if not laps.empty else max_lap

    st.subheader(f"{race} 2024 | {driver} | Lap {lap} of {total_laps}")

    st.header("1. Pit Window")
    pit_row, pit_note = nearest_pit_label_row(pit_df, race, driver, lap)
    if pit_row is None:
        st.info(pit_note)
    else:
        if pit_note:
            st.info(pit_note)
        pit_x, pit_issues = encode_pit_row(pit_row, artifacts["pit"])
        if pit_x.isna().any(axis=None):
            st.warning("Some pit-window features are missing; prediction is not shown.")
            st.dataframe(pit_x, width="stretch")
        else:
            pred = float(artifacts["pit"]["model"].predict(pit_x)[0])
            st.metric("Predicted laps until pit window opens", f"{pred:.1f}")
            st.caption(
                f"Saved label row: lap {int(pit_row['lap'])}, context `{pit_row['context']}`, "
                f"compound `{pit_row['compound']}`."
            )
            with st.expander("Pit-window feature vector"):
                display = pit_row[list(artifacts["pit"]["model"].feature_names_in_)].to_frame("value")
                st.dataframe(display, width="stretch")
        for issue in pit_issues:
            st.warning(issue)

    st.header("2. Undercut / Overcut")
    context, reason = build_strategy_context(session, driver, lap)
    if context is None:
        st.info(reason)
    else:
        undercut_prob, overcut_prob, undercut_row, overcut_row = predict_strategy(context, artifacts)
        c1, c2, c3 = st.columns(3)
        c1.metric("Direct rival ahead", context["rival"])
        c2.metric("Gap to rival", f"{context['gap_ahead']:.2f}s")
        c3.metric("Compound / tyre age", f"{context['compound']} / {context['tire_age']:.0f}")

        p1, p2 = st.columns(2)
        p1.metric("Undercut success probability", f"{undercut_prob:.1%}")
        p2.metric("Overcut success probability", f"{overcut_prob:.1%}")
        st.caption(
            "These are live scenario probabilities using the saved model feature lists. "
            "They are marked not applicable when a clean direct-rival context cannot be built."
        )
        with st.expander("Strategy feature vectors"):
            st.write("Undercut")
            st.dataframe(pd.DataFrame([undercut_row]), width="stretch")
            st.write("Overcut")
            st.dataframe(pd.DataFrame([overcut_row]), width="stretch")

    st.header("3. Degradation Context")
    selected = current_lap_row(laps, driver, lap)
    if selected is None:
        st.info("No lap record available for degradation context.")
        compound = "UNKNOWN"
        tire_age = np.nan
    else:
        compound = str(selected.get("Compound", "UNKNOWN"))
        tire_age = selected.get("TyreLife", np.nan)
        st.write(f"Current compound: `{compound}` | stint age: `{tire_age}` laps")
        render_degradation(compound, tire_age, artifacts["degradation"])

    st.header("4. Weather Context")
    render_weather(artifacts["weather"], compound)


if __name__ == "__main__":
    main()
