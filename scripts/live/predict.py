from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from live.features import strategy_context_from_snapshot
from live.quality import quality_flags_for_prediction, suppress_prediction
from race_strategy import predict_strategy


@dataclass(frozen=True)
class StrategySignal:
    call: str
    confidence: str
    reasons: list[str]
    quality_flags: list[str]
    suppressed: bool
    selected_strategy: str | None = None
    next_step: str = ""
    risk: str = ""
    undercut_prob: float | None = None
    overcut_prob: float | None = None
    edge: float | None = None
    pace_advantage: float | None = None
    deg_advantage: float | None = None
    context: dict | None = None
    undercut_row: dict | None = None
    overcut_row: dict | None = None


def _fmt_laps(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.1f}"


def _fmt_seconds(value):
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):+.2f}s"


def _probability_signal(probability, benchmark):
    if probability is None or pd.isna(probability) or pd.isna(benchmark):
        return "n/a", np.nan
    lift = float(probability) - float(benchmark)
    return f"{lift:+.1%} vs historical base rate", lift


def make_action_signal(
    pit_pred,
    undercut_prob,
    overcut_prob,
    context,
    benchmarks,
    quality_flags: list[str] | None = None,
) -> StrategySignal:
    quality_flags = quality_flags or []
    suppressed = suppress_prediction(quality_flags)
    if context is None or undercut_prob is None or overcut_prob is None:
        return StrategySignal(
            call="No clean strategy call",
            next_step="Select a lap where the driver has a measurable direct rival ahead.",
            confidence="Unavailable",
            reasons=["The direct-rival feature vector could not be built for this race state."],
            risk="No live undercut/overcut comparison is available for this lap.",
            selected_strategy=None,
            quality_flags=quality_flags,
            suppressed=True,
        )

    undercut_lift = undercut_prob - benchmarks["undercut"]["overall_rate"]
    overcut_lift = overcut_prob - benchmarks["overcut"]["overall_rate"]
    edge = float(undercut_prob - overcut_prob)
    gap = context["gap_ahead"]
    pace_advantage = -context["pace_delta"] if pd.notna(context["pace_delta"]) else np.nan
    deg_advantage = (
        context["ca_deg_delta"] - context["deg_delta"]
        if pd.notna(context["ca_deg_delta"]) and pd.notna(context["deg_delta"])
        else np.nan
    )

    if pit_pred is not None and pd.notna(pit_pred) and pit_pred <= 1.0:
        urgency = "Box window is open now"
    elif pit_pred is not None and pd.notna(pit_pred) and pit_pred <= 4.0:
        urgency = "Prepare for the stop window"
    else:
        urgency = "Keep monitoring"

    if edge >= 0.1 and undercut_lift > 0:
        call = "Attack with undercut"
        next_step = "Prioritize the next viable stop if pit-exit traffic is acceptable."
        selected_strategy = "undercut"
    elif edge <= -0.1 and overcut_lift > 0:
        call = "Extend for overcut"
        next_step = "Stay out while lap-time loss and rival undercut threat remain controlled."
        selected_strategy = "overcut"
    elif max(undercut_lift, overcut_lift) < 0:
        call = "Hold position"
        next_step = "Avoid forcing a low-edge strategy move; wait for a clearer gap or tyre delta."
        selected_strategy = None
    else:
        call = "Marginal call"
        next_step = "Treat this as a race-engineering judgement call and resolve with traffic, tyre inventory, and safety-car risk."
        selected_strategy = "undercut" if edge >= 0 else "overcut"

    if abs(edge) >= 0.18:
        confidence = "Strong"
    elif abs(edge) >= 0.08:
        confidence = "Moderate"
    else:
        confidence = "Weak"

    undercut_signal, _ = _probability_signal(
        undercut_prob, benchmarks["undercut"]["overall_rate"]
    )
    overcut_signal, _ = _probability_signal(
        overcut_prob, benchmarks["overcut"]["overall_rate"]
    )
    gap_signal = (
        f"direct rival gap {gap:.2f}s vs undercut median "
        f"{benchmarks['undercut']['median_gap']:.2f}s"
    )
    if selected_strategy == "overcut":
        gap_signal = (
            f"direct rival gap {gap:.2f}s vs overcut median "
            f"{benchmarks['overcut']['median_gap']:.2f}s"
        )

    reasons = [
        f"{urgency}: pit-window model says {_fmt_laps(pit_pred)} laps.",
        f"Undercut signal is {undercut_signal}; overcut signal is {overcut_signal}.",
        gap_signal,
        f"Recent pace advantage vs rival: {_fmt_seconds(pace_advantage)} per lap; degradation advantage: {_fmt_seconds(deg_advantage)} per tyre-age lap.",
    ]

    if gap < 1.5:
        risk = "Very small gaps are traffic-sensitive; pit-lane timing can dominate the model signal."
    elif gap > 8:
        risk = "Large gaps require a big tyre or pace offset, so probability should be treated as directional."
    elif confidence == "Weak":
        risk = "The model probabilities are close; external race context can flip the call."
    else:
        risk = "Primary residual risks are safety-car timing, pit-lane congestion, and unmodelled tyre inventory."

    if suppressed:
        call = "Suppressed"
        confidence = "Unavailable"
        next_step = "Wait for a supported, fresh green-flag race state before making an actionable call."
        risk = "Prediction suppressed by quality gates: " + ", ".join(quality_flags)

    return StrategySignal(
        call=call,
        next_step=next_step,
        confidence=confidence,
        reasons=reasons,
        risk=risk,
        selected_strategy=selected_strategy,
        quality_flags=quality_flags,
        suppressed=suppressed,
        undercut_prob=float(undercut_prob),
        overcut_prob=float(overcut_prob),
        edge=edge,
        pace_advantage=pace_advantage,
        deg_advantage=deg_advantage,
        context=context,
    )


def predict_snapshot_signal(
    snapshot,
    driver,
    artifacts,
    medians,
    benchmarks,
    lap: int | None = None,
    pit_pred=None,
    max_staleness_seconds: float = 30.0,
) -> StrategySignal:
    context, reason = strategy_context_from_snapshot(snapshot, driver, lap=lap)
    flags = quality_flags_for_prediction(
        snapshot,
        context,
        context_reason=reason,
        max_staleness_seconds=max_staleness_seconds,
    )
    if context is None:
        return make_action_signal(pit_pred, None, None, None, benchmarks, flags)
    undercut_prob, overcut_prob, undercut_row, overcut_row = predict_strategy(
        context, artifacts, medians
    )
    signal = make_action_signal(
        pit_pred,
        undercut_prob,
        overcut_prob,
        context,
        benchmarks,
        flags,
    )
    return StrategySignal(
        **{
            **signal.__dict__,
            "undercut_row": undercut_row,
            "overcut_row": overcut_row,
        }
    )
