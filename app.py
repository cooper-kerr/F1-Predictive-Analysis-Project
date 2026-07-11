from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "outputs" / "figures"
STATIC_LAPS_PATH = DATA_DIR / "f1_2024_static_laps.csv.gz"

sys.path.append(str(ROOT / "scripts"))
from race_strategy import (  # noqa: E402
    build_strategy_context,
    current_lap_row,
    encode_pit_row,
    nearest_pit_label_row,
    predict_strategy,
)
from strategy_features import strategy_feature_medians  # noqa: E402


@st.cache_data
def load_pit_labels():
    df = pd.read_csv(DATA_DIR / "f1_pit_window_labels.csv")
    return df[df["year"] == 2024].copy()


@st.cache_data
def pit_feature_medians():
    df = pd.read_csv(DATA_DIR / "f1_pit_window_labels.csv")
    return df.median(numeric_only=True).to_dict()


def apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --carbon: #0B0F14;
            --asphalt: #151B23;
            --panel: #1B232D;
            --panel-2: #202A35;
            --line: rgba(207, 216, 220, 0.16);
            --line-strong: rgba(242, 201, 76, 0.38);
            --text: #EEF3F7;
            --muted: #AAB7C4;
            --dim: #7C8A96;
            --amber: #F2C94C;
            --red: #FF3B30;
            --green: #28D17C;
            --blue: #64B5F6;
            --soft: #F45B69;
            --medium: #F2C94C;
            --hard: #D9E2EC;
        }

        .stApp {
            background:
                linear-gradient(90deg, rgba(242, 201, 76, 0.03) 1px, transparent 1px) 0 0 / 56px 56px,
                radial-gradient(circle at 82% 8%, rgba(100, 181, 246, 0.14), transparent 26rem),
                linear-gradient(135deg, #070A0D 0%, var(--carbon) 42%, #101820 100%);
            color: var(--text);
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .main .block-container {
            max-width: 1240px;
            padding-top: 1.15rem;
            padding-bottom: 4rem;
        }
        h1, h2, h3, h4 {
            letter-spacing: 0;
            color: var(--text);
            font-family: "Barlow Condensed", "Arial Narrow", "Inter", sans-serif;
            text-transform: uppercase;
        }
        p, li, label, span, div {
            color: inherit;
        }
        div[data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(242, 201, 76, 0.09), transparent 18rem),
                #0A0E13;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--text);
        }
        [data-baseweb="select"] > div,
        [data-testid="stSlider"] [data-baseweb="slider"] {
            color: var(--text);
        }
        [data-testid="stSelectbox"] div[role="button"],
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(255, 255, 255, 0.16);
            border-radius: 7px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
            border-bottom: 1px solid var(--line);
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.7rem;
            border-radius: 7px 7px 0 0;
            color: var(--muted);
            font-weight: 800;
            letter-spacing: 0.02em;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(242, 201, 76, 0.12);
            color: var(--amber) !important;
            border-bottom: 2px solid var(--amber);
        }
        [data-testid="stMetric"] {
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.025)),
                var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        [data-testid="stMetricValue"] {
            color: var(--text);
            font-family: "Barlow Condensed", "Arial Narrow", "Inter", sans-serif;
            font-size: 2.1rem;
            letter-spacing: 0;
        }
        [data-testid="stMetricDelta"] {
            color: var(--amber);
        }
        .race-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line-strong);
            border-radius: 8px;
            padding: 1.25rem 1.35rem 1.15rem;
            margin-bottom: 0.85rem;
            background:
                linear-gradient(90deg, rgba(255, 59, 48, 0.18), transparent 34%),
                repeating-linear-gradient(90deg, rgba(242, 201, 76, 0.12) 0 2px, transparent 2px 18px),
                #111820;
            box-shadow: 0 18px 60px rgba(0,0,0,0.28);
        }
        .race-hero::after {
            content: "";
            position: absolute;
            inset: auto 1.25rem 1rem auto;
            width: min(42vw, 360px);
            height: 4px;
            background: linear-gradient(90deg, var(--green), var(--amber), var(--red));
            border-radius: 99px;
            opacity: 0.9;
        }
        .race-hero h1 {
            margin: 0;
            max-width: 920px;
            font-size: clamp(2.45rem, 6vw, 5.2rem);
            line-height: 0.86;
            color: var(--text);
        }
        .race-hero p {
            max-width: 840px;
            margin: 0.75rem 0 0;
            color: var(--muted);
            font-size: 1.02rem;
        }
        .hero-kicker {
            color: var(--amber);
            font-size: 0.82rem;
            font-weight: 900;
            letter-spacing: 0.18em;
            margin-bottom: 0.45rem;
            text-transform: uppercase;
        }
        .scenario-strip {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 0.65rem 0 1.05rem;
        }
        .scenario-pill {
            border: 1px solid var(--line);
            border-radius: 7px;
            padding: 0.45rem 0.7rem;
            background: rgba(255, 255, 255, 0.055);
            color: var(--muted);
            font-size: 0.88rem;
            font-weight: 650;
        }
        .scenario-pill strong {
            color: var(--text);
        }
        .stAlert {
            border-radius: 8px;
            background: rgba(100, 181, 246, 0.08);
            color: var(--text);
        }
        div[data-testid="stExpander"] {
            border-radius: 8px;
            border-color: var(--line);
            background: rgba(255,255,255,0.035);
        }
        .evidence-card {
            min-height: 100%;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.85rem;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.025)),
                var(--panel);
        }
        .evidence-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1.18rem;
            color: var(--amber);
        }
        .evidence-card p {
            margin: 0.35rem 0;
            color: var(--muted);
        }
        .evidence-card ul {
            margin: 0.55rem 0 0 1.1rem;
            padding: 0;
            color: var(--muted);
        }
        .read-card {
            border-left: 3px solid var(--amber);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            min-height: 92px;
            background: rgba(255, 255, 255, 0.045);
        }
        .read-card strong {
            display: block;
            color: var(--text);
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }
        .read-card span {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.4;
        }
        .section-callout {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 0.25rem 0 1rem;
            background: rgba(255, 255, 255, 0.04);
        }
        .section-callout h3 {
            margin: 0 0 0.35rem;
            color: var(--text);
            font-size: 1.55rem;
        }
        .section-callout p {
            margin: 0;
            color: var(--muted);
        }
        .tyre-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.14rem 0.48rem;
            font-weight: 900;
            color: #06080A;
        }
        .tyre-soft { background: var(--soft); }
        .tyre-medium { background: var(--medium); }
        .tyre-hard { background: var(--hard); }
        .tyre-other { background: var(--blue); }
        .dataframe,
        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }
        hr {
            border-color: var(--line);
        }
        .figure-caption {
            color: var(--dim);
            font-size: 0.9rem;
            margin-top: -0.25rem;
            margin-bottom: 0.9rem;
        }
        button[kind="primary"], .stButton button {
            border-radius: 7px;
        }
        a {
            color: var(--blue);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    return {
        "pit": joblib.load(MODEL_DIR / "f1_pit_window_model_tuned.pkl"),
        "undercut": joblib.load(MODEL_DIR / "f1_undercut_model.pkl"),
        "overcut": joblib.load(MODEL_DIR / "f1_overcut_model.pkl"),
        "degradation": joblib.load(MODEL_DIR / "f1_degradation_analysis.pkl"),
        "weather": joblib.load(MODEL_DIR / "f1_weather_analysis.pkl"),
        "position": joblib.load(MODEL_DIR / "f1_position_comparison_analysis.pkl"),
    }


@st.cache_data
def load_position_curve():
    return pd.read_csv(DATA_DIR / "f1_position_comparison_curve.csv")


@st.cache_data
def load_strategy_dataset_summary():
    summary = {}
    for key, path, target in [
        ("pit_window", DATA_DIR / "f1_pit_window_labels.csv", "opens_in_horizon"),
        ("undercut", DATA_DIR / "f1_undercut_dataset.csv", "undercut_success"),
        ("overcut", DATA_DIR / "f1_overcut_dataset.csv", "overcut_success"),
    ]:
        df = pd.read_csv(path)
        years = sorted(df["year"].dropna().astype(int).unique()) if "year" in df else []
        circuits = int(df["circuit"].nunique()) if "circuit" in df else np.nan
        target_rate = float(df[target].mean()) if target in df else np.nan
        summary[key] = {
            "rows": len(df),
            "years": years,
            "circuits": circuits,
            "target_rate": target_rate,
        }
    return summary


def format_year_range(years):
    if not years:
        return "unknown"
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def format_percent(value):
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def tyre_badge(compound):
    compound_text = str(compound or "UNKNOWN").upper()
    css_class = {
        "SOFT": "tyre-soft",
        "MEDIUM": "tyre-medium",
        "HARD": "tyre-hard",
    }.get(compound_text, "tyre-other")
    return f'<span class="tyre-badge {css_class}">{compound_text}</span>'


def render_read_card(title, body):
    st.markdown(
        f"""
        <div class="read-card">
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_callout(title, body):
    st.markdown(
        f"""
        <div class="section-callout">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_card(title, summary, facts, caveat=None):
    fact_items = "".join(f"<li>{fact}</li>" for fact in facts)
    caveat_html = f"<p><strong>Use carefully:</strong> {caveat}</p>" if caveat else ""
    st.markdown(
        f"""
        <div class="evidence-card">
            <h4>{title}</h4>
            <p>{summary}</p>
            <ul>{fact_items}</ul>
            {caveat_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_static_figure(filename, title, caption=None):
    path = FIGURE_DIR / filename
    st.markdown(f"**{title}**")
    if not path.exists():
        st.warning(f"Missing saved figure: `{path.relative_to(ROOT)}`")
        return
    st.image(str(path), width="stretch")
    if caption:
        st.markdown(f'<div class="figure-caption">{caption}</div>', unsafe_allow_html=True)


def render_static_figure_grid(items, columns=2):
    cols = st.columns(columns)
    for idx, item in enumerate(items):
        with cols[idx % columns]:
            render_static_figure(*item)


@st.cache_data(show_spinner="Loading bundled 2024 lap data...")
def load_static_laps():
    laps = pd.read_csv(STATIC_LAPS_PATH)
    for col in ["LapTime", "LapStartTime", "PitInTime", "PitOutTime"]:
        laps[col] = pd.to_timedelta(laps[col], unit="s")
    laps["TrackStatus"] = laps["TrackStatus"].astype(str)
    return laps


@st.cache_resource(show_spinner="Loading bundled race data...")
def load_session(race_name):
    laps = load_static_laps()
    race_laps = laps[laps["Race"] == race_name].drop(columns=["Race"]).copy()
    if race_laps.empty:
        raise ValueError(f"No bundled 2024 lap data found for {race_name}.")
    return SimpleNamespace(laps=race_laps)


@st.cache_data
def strategy_medians():
    out = {}
    for key, path in [
        ("undercut", DATA_DIR / "f1_undercut_dataset.csv"),
        ("overcut", DATA_DIR / "f1_overcut_dataset.csv"),
    ]:
        df = pd.read_csv(path)
        out[key] = strategy_feature_medians(df, key)
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

    colors = {"2022-2023": "#64B5F6", "2024": "#F2C94C"}
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=(
            f"{compound} degradation pace profile",
            "Stint support by tyre-age bin",
        ),
    )

    for era, group in subset.groupby("era"):
        group = group.sort_values("age_bin")
        x = [str(age_bin) for age_bin in group["age_bin"]]
        y = group["median_delta_s"].round(3).to_numpy()
        low_support = group["n_stints"].to_numpy() < 15
        color = colors.get(era, None)
        custom = np.stack(
            [
                group["mean_delta_s"].round(3).to_numpy(),
                group["n_laps"].to_numpy(),
                group["n_stints"].to_numpy(),
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"{era} median delta",
                line=dict(color=color, width=3),
                marker=dict(
                    size=10,
                    color=np.where(low_support, "#151B23", color),
                    line=dict(color=color, width=np.where(low_support, 2.5, 0)),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Tyre age: %{x}<br>"
                    "Median delta: %{y:.3f}s/lap<br>"
                    "Mean delta: %{customdata[0]:.3f}s/lap<br>"
                    "Laps: %{customdata[1]:,.0f}<br>"
                    "Stints: %{customdata[2]:,.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    era_2024 = subset[subset["era"] == "2024"].sort_values("age_bin")
    if current_x is not None and not era_2024.empty:
        current_row = era_2024[era_2024["age_bin"].astype(str) == current_bin]
        if not current_row.empty:
            y = float(current_row.iloc[0]["median_delta_s"])
            fig.add_vline(
                x=current_bin,
                line_width=2,
                line_dash="dot",
                line_color="#28D17C",
                row="all",
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[current_bin],
                    y=[y],
                    mode="markers",
                    name=f"You are here ({current_bin})",
                    marker=dict(
                        symbol="star",
                        size=18,
                        color="#28D17C",
                        line=dict(color="#EEF3F7", width=1.5),
                    ),
                    hovertemplate=(
                        "<b>Selected lap</b><br>"
                        "Tyre age bin: %{x}<br>"
                        "2024 median delta: %{y:.3f}s/lap<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )

    for era, group in subset.groupby("era"):
        group = group.sort_values("age_bin")
        x = [str(age_bin) for age_bin in group["age_bin"]]
        counts = group["n_stints"].to_numpy()
        low_support = counts < 15
        fig.add_trace(
            go.Bar(
                x=x,
                y=counts,
                name=f"{era} stint count",
                marker=dict(
                    color=np.where(low_support, "#FF3B30", colors.get(era, "#64B5F6")),
                    line=dict(color="rgba(238, 243, 247, 0.22)", width=1),
                ),
                opacity=0.78,
                customdata=np.stack([group["n_laps"].to_numpy()], axis=-1),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Tyre age: %{x}<br>"
                    "Stints: %{y:,.0f}<br>"
                    "Laps: %{customdata[0]:,.0f}<extra></extra>"
                ),
                text=counts,
                textposition="outside",
            ),
            row=2,
            col=1,
        )

    fig.add_hline(
        y=15,
        line_width=1.5,
        line_dash="dash",
        line_color="#FF3B30",
        annotation_text="low-support threshold",
        annotation_position="top left",
        row=2,
        col=1,
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=labels,
        title_text="Tyre age bin",
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Median delta vs stint best (s/lap)", row=1, col=1)
    fig.update_yaxes(title_text="Stints", row=2, col=1, rangemode="tozero")
    fig.update_layout(
        height=680,
        hovermode="x unified",
        barmode="group",
        margin=dict(l=24, r=24, t=74, b=36),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21,27,35,0.96)",
        font=dict(size=13, color="#EEF3F7"),
        xaxis=dict(gridcolor="rgba(207,216,220,0.14)", zerolinecolor="rgba(207,216,220,0.18)"),
        yaxis=dict(gridcolor="rgba(207,216,220,0.14)", zerolinecolor="rgba(207,216,220,0.18)"),
        xaxis2=dict(gridcolor="rgba(207,216,220,0.14)", zerolinecolor="rgba(207,216,220,0.18)"),
        yaxis2=dict(gridcolor="rgba(207,216,220,0.14)", zerolinecolor="rgba(207,216,220,0.18)"),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

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
    st.info(
        "Weather context estimates how track and air conditions relate to lap-time "
        "variation in the saved season data. In race terms, hotter track surfaces can "
        "change tyre behavior, stint durability, and the value of stopping early versus "
        "extending a stint."
    )
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


def render_position_predictability(artifact):
    curve = load_position_curve().sort_values(["year", "checkpoint_lap"])
    baselines = artifact.get("baselines", {})
    latest_lap = int(curve["checkpoint_lap"].max())
    latest = curve[curve["checkpoint_lap"] == latest_lap].copy()

    st.subheader("Position Predictability")
    c1, c2, c3 = st.columns(3)
    c1.metric("Evaluation", artifact.get("evaluation", "GroupKFold"))
    c2.metric("Race seasons", ", ".join(str(year) for year in artifact.get("years", [])))
    c3.metric("Current position included", "No" if not artifact.get("position_included") else "Yes")

    fig = go.Figure()
    for year, group in curve.groupby("year"):
        fig.add_trace(
            go.Scatter(
                x=group["checkpoint_lap"],
                y=group["accuracy"],
                error_y=dict(type="data", array=group["se"], visible=True),
                mode="lines+markers",
                name=str(year),
                line=dict(width=3),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Lap: %{x}<br>"
                    "Accuracy: %{y:.1%}<br>"
                    "SE: %{error_y.array:.1%}<extra></extra>"
                ),
            )
        )
        baseline = baselines.get(int(year), baselines.get(year))
        if baseline is not None:
            fig.add_hline(
                y=float(baseline),
                line_width=1,
                line_dash="dot",
                line_color="rgba(242, 201, 76, 0.45)",
                annotation_text=f"{year} majority baseline {float(baseline):.1%}",
                annotation_position="bottom right",
            )

    fig.update_layout(
        height=420,
        margin=dict(l=24, r=24, t=28, b=36),
        yaxis_tickformat=".0%",
        yaxis_title="Class accuracy",
        xaxis_title="Race checkpoint lap",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21,27,35,0.96)",
        font=dict(color="#EEF3F7"),
        xaxis=dict(gridcolor="rgba(207,216,220,0.14)", zerolinecolor="rgba(207,216,220,0.18)"),
        yaxis=dict(gridcolor="rgba(207,216,220,0.14)", zerolinecolor="rgba(207,216,220,0.18)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    display = curve.copy()
    display["accuracy"] = display["accuracy"].map(lambda value: f"{value:.1%}")
    display["se"] = display["se"].map(lambda value: f"{value:.1%}")
    display["std"] = display["std"].map(lambda value: f"{value:.1%}")
    st.dataframe(display, width="stretch", hide_index=True)

    if not latest.empty:
        facts = [
            f"{int(row.year)} lap {latest_lap}: {row.accuracy:.1%} accuracy over "
            f"{int(row.n_folds)} race folds and {int(row.n_rows)} driver-race rows."
            for row in latest.itertuples(index=False)
        ]
        st.caption(" ".join(facts))


def render_model_evidence(artifacts):
    summary = load_strategy_dataset_summary()
    pit_model = artifacts["pit"]["model"]
    undercut_model = artifacts["undercut"]["model"]
    overcut_model = artifacts["overcut"]["model"]
    position_artifact = artifacts["position"]
    degradation_artifact = artifacts["degradation"]
    weather_artifact = artifacts["weather"]

    st.subheader("Model Evidence & Saved Artifacts")
    st.info(
        "This section exposes the report artifacts behind the interactive demo: model "
        "evaluation figures, feature-importance views, SHAP interpretation plots, and "
        "descriptive analysis outputs. These are static build artifacts, not additional "
        "live predictions."
    )

    pit_stats = summary["pit_window"]
    undercut_stats = summary["undercut"]
    overcut_stats = summary["overcut"]

    cards = st.columns(3)
    with cards[0]:
        render_model_card(
            "Pit-window forecaster",
            "Gradient-boosted lap-level model estimating laps until the strategic pit window opens.",
            [
                f"Training/evaluation rows in labels CSV: {pit_stats['rows']:,}.",
                f"Seasons represented: {format_year_range(pit_stats['years'])}; circuits: {pit_stats['circuits']}.",
                "Saved 2024 holdout: MAE 2.21 laps, RMSE 3.41 laps, 58.5% within +/-2 laps, 85.2% within +/-5 laps.",
                f"Feature count: {len(pit_model.feature_names_in_)}; tuned params saved with the model artifact.",
            ],
            "Use the lap estimate as a strategy signal, not as a deterministic pit-call timer.",
        )
    with cards[1]:
        render_model_card(
            "Undercut classifier",
            "Rival-context classifier for whether stopping before the car ahead is likely to work.",
            [
                f"Rows: {undercut_stats['rows']:,}; seasons: {format_year_range(undercut_stats['years'])}; circuits: {undercut_stats['circuits']}.",
                f"Observed success rate in artifact dataset: {format_percent(undercut_stats['target_rate'])}.",
                f"Feature count: {len(artifacts['undercut']['features'])}; model: {type(undercut_model).__name__}.",
            ],
            "Only applies when the selected car has a clean direct-rival-ahead context.",
        )
    with cards[2]:
        render_model_card(
            "Overcut classifier",
            "Rival-context classifier for whether staying out after the car ahead pits is likely to work.",
            [
                f"Rows: {overcut_stats['rows']:,}; seasons: {format_year_range(overcut_stats['years'])}; circuits: {overcut_stats['circuits']}.",
                f"Observed success rate in artifact dataset: {format_percent(overcut_stats['target_rate'])}.",
                f"Feature count: {len(artifacts['overcut']['features'])}; model: {type(overcut_model).__name__}.",
            ],
            "Traffic, safety-car timing, and pit-lane congestion remain outside this compact classifier.",
        )

    evidence_tabs = st.tabs(
        ["Pit", "Strategy", "Degradation", "Weather", "Position", "Integrated"]
    )

    with evidence_tabs[0]:
        render_static_figure_grid(
            [
                ("pit_window_evaluation.png", "Evaluation", "Holdout-error and model performance summary."),
                ("pit_window_feature_importance.png", "Feature Importance", "Saved importance view for race-state inputs."),
                ("pit_window_per_context.png", "Per-Context Performance", "Error behavior split by traffic/window context."),
                ("pit_window_eda.png", "Training Data Shape", "Exploratory summary of engineered pit-window labels."),
            ]
        )

    with evidence_tabs[1]:
        st.subheader("Undercut Evidence")
        render_static_figure_grid(
            [
                ("undercut_model_evaluation.png", "Undercut Evaluation", "Classifier evaluation for stopping before the rival."),
                ("undercut_shap.png", "Undercut SHAP", "Interpretability view for undercut success probability."),
                ("undercut_gap_vs_success.png", "Gap vs Success", "Observed relationship between direct-rival gap and result."),
                ("undercut_age_vs_success.png", "Tyre Age vs Success", "Observed tyre-age context for undercut attempts."),
            ]
        )
        st.subheader("Overcut Evidence")
        render_static_figure_grid(
            [
                ("overcut_model_evaluation.png", "Overcut Evaluation", "Classifier evaluation for staying out after the rival pits."),
                ("overcut_shap.png", "Overcut SHAP", "Interpretability view for overcut success probability."),
                ("overcut_gap_vs_success.png", "Gap vs Success", "Observed relationship between direct-rival gap and result."),
                ("overcut_stay_out_laps.png", "Stay-Out Laps", "How long successful and unsuccessful overcut attempts extended."),
            ]
        )

    with evidence_tabs[2]:
        d1, d2, d3 = st.columns(3)
        d1.metric("Stint summaries", f"{len(degradation_artifact['stint_summaries']):,}")
        d2.metric("Train years", ", ".join(str(year) for year in degradation_artifact["train_years"]))
        d3.metric("Test years", ", ".join(str(year) for year in degradation_artifact["test_years"]))
        render_static_figure_grid(
            [
                ("degradation_pace_profile.png", "Pace Profile", "Median lap delta by compound, tyre-age bin, and era."),
                ("peak_age_by_compound.png", "Peak Age By Compound", "Where compounds tend to reach best fuel-corrected pace."),
                ("stint_length_vs_degradation.png", "Stint Length vs Degradation", "Relationship between stint duration and degradation slope."),
            ]
        )

    with evidence_tabs[3]:
        w1, w2, w3 = st.columns(3)
        w1.metric("Filtered laps", f"{weather_artifact['n_rows']:,}")
        w2.metric("OLS R-squared", f"{weather_artifact['r_squared']:.3f}")
        w3.metric("Adjusted R-squared", f"{weather_artifact['adj_r_squared']:.3f}")
        render_static_figure(
            "weather_laptime_scatter.png",
            "Weather / Lap-Time Scatter",
            "Saved report figure for weather and temperature effects on lap time.",
        )
        st.dataframe(weather_artifact["coefficients"], width="stretch", hide_index=True)

    with evidence_tabs[4]:
        render_position_predictability(position_artifact)
        render_static_figure(
            "position_comparison_curve.png",
            "Saved Position Comparison Curve",
            "Report artifact generated by scripts/build_position_comparison_analysis.py.",
        )

    with evidence_tabs[5]:
        render_static_figure(
            "integrated_race_dashboard.png",
            "Integrated Hungary Race-Day Strategy Example",
            "Static report artifact combining pit-window, undercut, and overcut outputs.",
        )


def fmt_laps(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}"


def fmt_pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1%}"


def pit_window_read(pred):
    if pred is None or pd.isna(pred):
        return "Pit-window read is unavailable for this selected race state."
    if pred <= 1:
        return (
            "Race read: the pit window is effectively open. A team would start weighing "
            "track position, tyre availability, and traffic on pit exit."
        )
    if pred <= 5:
        return (
            "Race read: the car is approaching its viable stop phase. This is where teams "
            "watch rival gaps closely and prepare to react to undercut pressure."
        )
    return (
        "Race read: the model does not yet see a strong reason to stop. Staying out may "
        "preserve tyre offset or avoid rejoining into traffic."
    )


def strategy_read(undercut_prob, overcut_prob):
    if undercut_prob is None or overcut_prob is None:
        return "Rival strategy read is unavailable until a clean direct-rival context exists."
    if undercut_prob >= overcut_prob + 0.1:
        return (
            "Race read: the model favors attacking early. That usually points to a usable "
            "gap, meaningful fresh-tyre benefit, or a rival ahead who may be vulnerable on older tyres."
        )
    if overcut_prob >= undercut_prob + 0.1:
        return (
            "Race read: the model favors patience. Staying out may be stronger when track "
            "position, traffic, or current tyre performance make an immediate stop less attractive."
        )
    return (
        "Race read: the two options are close. In a real race this is the kind of marginal "
        "call where pit-lane traffic, safety-car risk, and tyre inventory can decide the strategy."
    )


def degradation_read(tire_age):
    if pd.isna(tire_age):
        return "Tyre-age context is unavailable for this selected lap."
    if tire_age < 8:
        return (
            "Race read: this is an early-stint tyre. Pace should still be more about "
            "warm-up, traffic, and fuel load than heavy wear."
        )
    if tire_age < 20:
        return (
            "Race read: this is the middle of the stint, where degradation starts to "
            "matter for defending position and timing the stop."
        )
    return (
        "Race read: this is an older tyre phase. If lap-time delta is rising, the driver "
        "may be exposed to undercuts or forced into tyre management."
    )


def main():
    st.set_page_config(page_title="F1 Race Strategy Intelligence", layout="wide")
    apply_theme()

    pit_df = load_pit_labels()
    pit_medians = pit_feature_medians()
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
        st.caption(
            "Saved 2024 race-state rows and bundled lap data power the live demo."
        )

    session = load_session(race)
    laps = session.laps.copy()
    total_laps = int(laps["LapNumber"].max()) if not laps.empty else max_lap

    pit_row, pit_note = nearest_pit_label_row(pit_df, race, driver, lap)
    pit_x = None
    pit_issues = []
    pit_filled = []
    pit_pred = None
    if pit_row is not None:
        pit_x, pit_issues, pit_filled = encode_pit_row(
            pit_row, artifacts["pit"], pit_medians
        )
        remaining_missing = pit_x.columns[pit_x.isna().iloc[0]].tolist()
        if remaining_missing:
            pit_x = pit_x.fillna(0.0)
            pit_filled.extend(remaining_missing)
        pit_pred = float(artifacts["pit"]["model"].predict(pit_x)[0])

    context, strategy_reason = build_strategy_context(session, driver, lap)
    undercut_prob = None
    overcut_prob = None
    undercut_row = None
    overcut_row = None
    if context is not None:
        undercut_prob, overcut_prob, undercut_row, overcut_row = predict_strategy(
            context, artifacts, strategy_medians()
        )

    selected = current_lap_row(laps, driver, lap)
    if selected is None:
        compound = "UNKNOWN"
        tire_age = np.nan
    else:
        compound = str(selected.get("Compound", "UNKNOWN"))
        tire_age = selected.get("TyreLife", np.nan)

    st.markdown(
        f"""
        <div class="race-hero">
            <div class="hero-kicker">2024 race-control model wall</div>
            <h1>F1 Race Strategy Intelligence</h1>
            <p>
                A race-strategy dashboard for the 2024 {race}: pit-window timing,
                undercut and overcut pressure, tyre life, and weather evidence in one inspection view.
            </p>
        </div>
        <div class="scenario-strip">
            <span class="scenario-pill">Race: <strong>{race} 2024</strong></span>
            <span class="scenario-pill">Driver: <strong>{driver}</strong></span>
            <span class="scenario-pill">Lap: <strong>{lap} / {total_laps}</strong></span>
            <span class="scenario-pill">Compound: {tyre_badge(compound)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Pit Window Opens In",
        f"{fmt_laps(pit_pred)} laps" if pit_pred is not None else "N/A",
    )
    k2.metric("Undercut Success", fmt_pct(undercut_prob))
    k3.metric("Overcut Success", fmt_pct(overcut_prob))
    k4.metric(
        "Direct Rival Gap",
        f"{context['gap_ahead']:.2f}s" if context is not None else "N/A",
        context["rival"] if context is not None else None,
    )

    s1, s2, s3 = st.columns([1.15, 1, 1])
    with s1:
        render_read_card("Pit wall read", pit_window_read(pit_pred))
    with s2:
        render_read_card("Attack choice", strategy_read(undercut_prob, overcut_prob))
    with s3:
        render_read_card("Tyre phase", degradation_read(tire_age))

    overview, pit_tab, strategy_tab, degradation_tab, weather_tab, evidence_tab = st.tabs(
        [
            "Overview",
            "Pit Window",
            "Undercut / Overcut",
            "Tyre Degradation",
            "Weather",
            "Evidence",
        ]
    )

    with overview:
        render_section_callout(
            "Race-State Snapshot",
            "The overview pairs the live strategy outputs with the report artifacts behind the model: pit-window forecasting, rival attack probabilities, tyre degradation, weather effects, and race-position predictability.",
        )
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Analysed 2024 Races", f"{len(races)}")
        o2.metric("Bundled Lap Rows", f"{len(laps):,}")
        o3.metric(
            "Weather Model Fit",
            f"R² {artifacts['weather']['r_squared']:.3f}",
            f"{artifacts['weather']['n_rows']:,} rows",
        )
        o4.metric(
            "Degradation Stints",
            f"{len(artifacts['degradation']['stint_summaries']):,}",
        )
        if pit_note:
            st.caption(pit_note)
        if context is None and strategy_reason:
            st.caption(strategy_reason)

    with pit_tab:
        render_section_callout(
            "Pit Window",
            "Estimates how many laps remain before a stop becomes strategically available. Lower values mean the car is near the crossover where tyre age, pace, traffic gaps, and pit loss make a stop viable.",
        )
        if pit_row is None:
            st.caption(pit_note)
        else:
            st.metric("Predicted laps until pit window opens", f"{pit_pred:.1f}")
            st.caption(pit_window_read(pit_pred))
            st.caption(
                f"Saved label row: lap {int(pit_row['lap'])}, context `{pit_row['context']}`, "
                f"compound `{pit_row['compound']}`."
            )
            with st.expander("Pit-window feature vector"):
                display = pit_row[
                    list(artifacts["pit"]["model"].feature_names_in_)
                ].astype(str).to_frame("value")
                st.dataframe(display, width="stretch")
            if pit_filled:
                st.caption(
                    "Missing pit-window feature values filled before prediction: "
                    + ", ".join(f"`{feature}`" for feature in dict.fromkeys(pit_filled))
                    + "."
                )
            for issue in pit_issues:
                st.warning(issue)

    with strategy_tab:
        render_section_callout(
            "Undercut / Overcut",
            "Compares two rival-focused strategy attacks: stop before the car ahead to exploit fresh tyres, or stay out longer to use clean air, tyre offset, or track position.",
        )
        if context is None:
            st.caption(strategy_reason)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Direct rival ahead", context["rival"])
            c2.metric("Gap to rival", f"{context['gap_ahead']:.2f}s")
            c3.metric(
                "Compound / tyre age",
                f"{context['compound']} / {context['tire_age']:.0f}",
            )

            p1, p2 = st.columns(2)
            p1.metric("Undercut success probability", f"{undercut_prob:.1%}")
            p2.metric("Overcut success probability", f"{overcut_prob:.1%}")
            st.caption(strategy_read(undercut_prob, overcut_prob))
            st.caption(
                "These are live scenario probabilities using the saved model feature lists. "
                "They are marked not applicable when a clean direct-rival context cannot be built."
            )
            with st.expander("Strategy feature vectors"):
                st.write("Undercut")
                st.dataframe(pd.DataFrame([undercut_row]), width="stretch")
                st.write("Overcut")
                st.dataframe(pd.DataFrame([overcut_row]), width="stretch")

    with degradation_tab:
        render_section_callout(
            "Tyre Degradation",
            "Shows how the selected compound has behaved as tyre age increases, helping explain whether the stint still has enough life to defend, attack, or extend.",
        )
        if selected is None:
            st.caption("No lap record available for degradation context.")
        else:
            st.markdown(
                f"Current compound: {tyre_badge(compound)} &nbsp; Stint age: `{tire_age}` laps",
                unsafe_allow_html=True,
            )
            st.caption(degradation_read(tire_age))
            render_degradation(compound, tire_age, artifacts["degradation"])

    with weather_tab:
        render_section_callout(
            "Weather Context",
            "Connects track and air conditions to lap-time variation in the saved season sample, giving context for tyre behavior and stint durability.",
        )
        render_weather(artifacts["weather"], compound)

    with evidence_tab:
        render_model_evidence(artifacts)


if __name__ == "__main__":
    main()
