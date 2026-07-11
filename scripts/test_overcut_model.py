"""
Interactive test harness for the F1 overcut prediction model.

Four modes:
  1. inspect   — browse real 2024 holdout predictions sorted by confidence
  2. errors    — show the worst mispredictions to sanity-check the model
  3. driver    — filter holdout predictions for a specific driver
  4. scenario  — predict success probability for a custom hand-crafted scenario

Run:
    python scripts/test_overcut_model.py inspect
    python scripts/test_overcut_model.py errors
    python scripts/test_overcut_model.py driver SAI
    python scripts/test_overcut_model.py scenario
    python scripts/test_overcut_model.py  (defaults to inspect)

Overcut framing:
  stay_out_driver = driver who stays out (executing the overcut)
  pitting_car     = car directly ahead that pits first
  Success         = stay_out_driver is ahead of pitting_car after both have pitted
"""

import sys
import warnings
import joblib
import pandas as pd
from pathlib import Path

from strategy_features import (
    overcut_values,
    prepare_strategy_frame,
    strategy_vector,
)

warnings.filterwarnings('ignore')

ROOT         = Path(__file__).parent.parent
MODEL_PATH   = ROOT / 'models' / 'f1_overcut_model.pkl'
DATASET_PATH = ROOT / 'data'   / 'f1_overcut_dataset.csv'

TRAIN_YEARS = [2022, 2023]
TEST_YEARS  = [2024]

# ── Load model & data ────────────────────────────────────────────────────────

artefacts = joblib.load(MODEL_PATH)
model     = artefacts['model']
FEATURES  = artefacts['features']

df_raw = pd.read_csv(DATASET_PATH)
df, FEATURES = prepare_strategy_frame(df_raw, 'overcut', FEATURES)

# Predict probabilities on full dataset
X     = df[FEATURES].values.astype(float)
probs = model.predict_proba(X)[:, 1]
preds = (probs >= 0.5).astype(int)

df['prob_success'] = probs
df['predicted']    = preds
df['correct']      = (preds == df['overcut_success']).astype(int)

# Test set only
test_df = df[df['year'].isin(TEST_YEARS)].copy()

# ── Display helpers ──────────────────────────────────────────────────────────

def _row_str(row):
    outcome  = 'SUCCESS' if row['overcut_success'] == 1 else 'FAILURE'
    pred_lbl = 'SUCCESS' if row['predicted']        == 1 else 'FAILURE'
    hit      = '✓' if row['correct'] == 1 else '✗'
    stay_out_laps = int(row['stay_out_laps']) if pd.notna(row.get('stay_out_laps')) else '?'
    return (
        f"{hit}  {row['circuit'][:28]:28s}  "
        f"stay={row['stay_out_driver']:3s}  pit={row['pitting_car']:3s}  "
        f"gap={row['gap_ahead']:5.1f}s  "
        f"age={int(row['tire_age']):2d}vs{int(row['ca_tire_age']):2d}  "
        f"out={stay_out_laps:>2}L  "
        f"p={row['prob_success']:.2f}  "
        f"pred={pred_lbl:7s}  actual={outcome}"
    )


# ── Mode: inspect ────────────────────────────────────────────────────────────

def mode_inspect():
    print(f"\n{'─'*110}")
    print(f"2024 holdout predictions  ({len(test_df)} attempts)  "
          f"| Model accuracy: {test_df['correct'].mean():.1%}  "
          f"| Base rate (success): {test_df['overcut_success'].mean():.1%}")
    print(f"{'─'*110}")
    print(f"  {'circuit':28s}  {'stay':3s}  {'pit':3s}  "
          f"{'gap':6s}  {'ages':9s}  {'out':4s}  {'p':4s}  "
          f"{'predicted':9s}  actual")
    print(f"{'─'*110}")

    sorted_df = test_df.sort_values('prob_success', ascending=False)
    for _, row in sorted_df.iterrows():
        print(_row_str(row))

    acc = test_df['correct'].mean()
    tp  = ((test_df['predicted'] == 1) & (test_df['overcut_success'] == 1)).sum()
    fp  = ((test_df['predicted'] == 1) & (test_df['overcut_success'] == 0)).sum()
    tn  = ((test_df['predicted'] == 0) & (test_df['overcut_success'] == 0)).sum()
    fn  = ((test_df['predicted'] == 0) & (test_df['overcut_success'] == 1)).sum()

    print(f"\n{'─'*110}")
    print(f"Accuracy: {acc:.1%}  |  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    if (tp + fp) > 0:
        print(f"Precision (of predicted successes): {tp/(tp+fp):.1%}  |  "
              f"Recall (caught actual successes): {tp/(tp+fn):.1%}")


# ── Mode: errors ─────────────────────────────────────────────────────────────

def mode_errors():
    wrong = test_df[test_df['correct'] == 0].copy()
    wrong = wrong.sort_values('prob_success', ascending=False)

    false_pos = wrong[wrong['predicted'] == 1]   # model said success, was failure
    false_neg = wrong[wrong['predicted'] == 0]   # model said failure, was success

    print(f"\n{'─'*110}")
    print(f"FALSE POSITIVES ({len(false_pos)})  —  model predicted SUCCESS but actual FAILURE")
    print(f"These are overcuts the model thought would work but didn't")
    print(f"{'─'*110}")
    for _, row in false_pos.iterrows():
        print(_row_str(row))

    print(f"\n{'─'*110}")
    print(f"FALSE NEGATIVES ({len(false_neg)})  —  model predicted FAILURE but actual SUCCESS")
    print(f"These are overcuts the model missed / didn't believe in")
    print(f"{'─'*110}")
    for _, row in false_neg.sort_values('prob_success').iterrows():
        print(_row_str(row))


# ── Mode: driver ─────────────────────────────────────────────────────────────

def mode_driver(driver_code):
    driver_df = test_df[test_df['stay_out_driver'] == driver_code.upper()].copy()
    if driver_df.empty:
        available = sorted(test_df['stay_out_driver'].unique())
        print(f"Driver '{driver_code}' not found in 2024 holdout (as stay-out driver).")
        print(f"Available: {', '.join(available)}")
        return

    print(f"\n{'─'*110}")
    print(f"2024 overcut predictions for stay-out driver {driver_code.upper()}  "
          f"({len(driver_df)} overcut attempts  |  "
          f"accuracy: {driver_df['correct'].mean():.1%})")
    print(f"{'─'*110}")
    for _, row in driver_df.sort_values('prob_success', ascending=False).iterrows():
        print(_row_str(row))


# ── Mode: scenario ───────────────────────────────────────────────────────────

def mode_scenario():
    """
    Build a custom overcut scenario and predict success probability.
    The stay-out driver is attempting to overcut the car that just pitted.
    """
    print("\n── Custom Overcut Scenario Predictor ──")
    print("Framing: the car ahead has just pitted. You (stay-out driver) choose")
    print("to stay out and pit later, hoping to emerge ahead.")
    print("Press Enter to accept default values.\n")

    def ask(prompt, default, cast=float):
        raw = input(f"  {prompt} [{default}]: ").strip()
        return cast(raw) if raw else cast(default)

    gap_ahead      = ask("Gap to car ahead before they pitted (seconds)",    5.0)
    tire_age       = ask("Your (stay-out) tire age (laps)",                  18, int)
    ca_tire_age    = ask("Car-ahead tire age when they pitted (laps)",       22, int)
    own_pace_raw   = ask("Your recent avg lap time (seconds)",               90.0)
    threat_pace_raw = ask("Car-ahead recent avg lap time (s)",               90.5)
    deg_delta      = ask("Your deg rate (s/lap, ~0.05–0.15)",                0.10)
    ca_deg_delta   = ask("Car-ahead deg rate (s/lap)",                       0.08)
    closing_rate   = ask("Closing rate (negative = you closing on them)",   -0.2)
    pit_loss       = ask("Pit lane time loss (seconds)",                     22.0)
    race_progress  = ask("Race progress (0=start, 1=end)",                   0.5)

    print("\n  Compound options: SOFT / MEDIUM / HARD / INTERMEDIATE / WET")
    compound_raw = input("  Your (stay-out) compound [MEDIUM]: ").strip().upper() or 'MEDIUM'

    row = overcut_values(
        gap_ahead, tire_age, ca_tire_age, own_pace_raw, threat_pace_raw,
        deg_delta, ca_deg_delta, closing_rate, pit_loss, race_progress, compound_raw,
    )
    X_scenario, row = strategy_vector(FEATURES, row, {})
    prob = model.predict_proba(X_scenario)[0, 1]

    # Sensitivity: what changes if gap halves / doubles?
    row_half   = dict(row); row_half['gap_ahead']   = gap_ahead / 2
    row_double = dict(row); row_double['gap_ahead']  = gap_ahead * 2
    X_half, _ = strategy_vector(FEATURES, row_half, {})
    X_double, _ = strategy_vector(FEATURES, row_double, {})
    prob_half = model.predict_proba(X_half)[0, 1]
    prob_double = model.predict_proba(X_double)[0, 1]

    verdict = 'LIKELY SUCCESS' if prob >= 0.5 else 'LIKELY FAILURE'
    bar_len  = int(prob * 30)
    bar      = '█' * bar_len + '░' * (30 - bar_len)

    print(f"\n{'─'*60}")
    print(f"  P(overcut success) = {prob:.1%}  [{bar}]")
    print(f"  Verdict: {verdict}")
    print(f"{'─'*60}")
    print(f"  Scenario inputs:")
    print(f"    Gap ahead:        {gap_ahead:.1f}s (before car pitted)")
    print(f"    Tire ages:        yours={tire_age}L  car_ahead={ca_tire_age}L  "
          f"(delta = {row['tire_age_delta']:+.0f} laps, positive = you more worn)")
    print(f"    Pace delta:       {row['pace_delta']:+.3f}s (negative = you're faster)")
    print(f"    Closing rate:     {closing_rate:+.3f}s/lap")
    print(f"    Compound:         {compound_raw}")
    print(f"{'─'*60}")
    print(f"  Sensitivity:")
    print(f"    If gap were {gap_ahead/2:.1f}s:  P = {prob_half:.1%}  ({prob_half-prob:+.1%})")
    print(f"    If gap were {gap_ahead*2:.1f}s:  P = {prob_double:.1%}  ({prob_double-prob:+.1%})")
    print(f"{'─'*60}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'inspect'

    if mode == 'inspect':
        mode_inspect()
    elif mode == 'errors':
        mode_errors()
    elif mode == 'scenario':
        mode_scenario()
    elif mode == 'driver':
        driver = sys.argv[2] if len(sys.argv) > 2 else 'SAI'
        mode_driver(driver)
    else:
        print(f"Unknown mode '{mode}'. Use: inspect | errors | scenario | driver <CODE>")
