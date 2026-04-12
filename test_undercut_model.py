"""
Interactive test harness for the F1 undercut prediction model.

Three modes:
  1. inspect   — browse real 2024 holdout predictions sorted by confidence
  2. errors    — show the worst mispredictions to sanity-check the model
  3. scenario  — predict success probability for a custom hand-crafted scenario
  4. driver    — filter holdout predictions for a specific driver

Run:
    python test_undercut_model.py inspect
    python test_undercut_model.py errors
    python test_undercut_model.py scenario
    python test_undercut_model.py driver VER
    python test_undercut_model.py  (defaults to inspect)
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

MODEL_PATH   = './f1_undercut_model.pkl'
DATASET_PATH = './f1_undercut_dataset.csv'

TRAIN_YEARS = [2022, 2023]
TEST_YEARS  = [2024]

# ── Load model & data ────────────────────────────────────────────────────────

artefacts = joblib.load(MODEL_PATH)
model     = artefacts['model']
FEATURES  = artefacts['features']

df_raw = pd.read_csv(DATASET_PATH)

# Rebuild compound dummies (must match training)
compound_dummies = pd.get_dummies(df_raw['compound'], prefix='compound')
df = pd.concat([df_raw, compound_dummies], axis=1)

for col in FEATURES:
    if col not in df.columns:
        df[col] = 0

REQUIRED = ['gap_ahead', 'tire_age', 'car_ahead_tire_age', 'own_pace', 'threat_pace']
df = df.dropna(subset=REQUIRED + ['undercut_success']).copy()
for col in FEATURES:
    df[col] = df[col].fillna(df[col].median())

# Predict probabilities on full dataset
X     = df[FEATURES].values.astype(float)
probs = model.predict_proba(X)[:, 1]
preds = (probs >= 0.5).astype(int)

df['prob_success'] = probs
df['predicted']    = preds
df['correct']      = (preds == df['undercut_success']).astype(int)

# Test set only
test_df = df[df['year'].isin(TEST_YEARS)].copy()

# ── Display helpers ──────────────────────────────────────────────────────────

def _row_str(row):
    outcome  = 'SUCCESS' if row['undercut_success'] == 1 else 'FAILURE'
    pred_lbl = 'SUCCESS' if row['predicted']         == 1 else 'FAILURE'
    hit      = '✓' if row['correct'] == 1 else '✗'
    return (
        f"{hit}  {row['circuit'][:28]:28s}  "
        f"drv={row['driver']:3s}  ahead={row['car_ahead']:3s}  "
        f"gap={row['gap_ahead']:5.1f}s  "
        f"age={int(row['tire_age']):2d}vs{int(row['car_ahead_tire_age']):2d}  "
        f"p={row['prob_success']:.2f}  "
        f"pred={pred_lbl:7s}  actual={outcome}"
    )


# ── Mode: inspect ────────────────────────────────────────────────────────────

def mode_inspect():
    print(f"\n{'─'*100}")
    print(f"2024 holdout predictions  ({len(test_df)} attempts)  "
          f"| Model accuracy: {test_df['correct'].mean():.1%}")
    print(f"{'─'*100}")
    print(f"  {'circuit':28s}  {'drv':3s}  {'ahead':5s}  "
          f"{'gap':6s}  {'tyre ages':9s}  {'p':4s}  "
          f"{'predicted':9s}  actual")
    print(f"{'─'*100}")

    sorted_df = test_df.sort_values('prob_success', ascending=False)
    for _, row in sorted_df.iterrows():
        print(_row_str(row))

    acc  = test_df['correct'].mean()
    tp   = ((test_df['predicted'] == 1) & (test_df['undercut_success'] == 1)).sum()
    fp   = ((test_df['predicted'] == 1) & (test_df['undercut_success'] == 0)).sum()
    tn   = ((test_df['predicted'] == 0) & (test_df['undercut_success'] == 0)).sum()
    fn   = ((test_df['predicted'] == 0) & (test_df['undercut_success'] == 1)).sum()

    print(f"\n{'─'*100}")
    print(f"Accuracy: {acc:.1%}  |  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Precision (of predicted successes): {tp/(tp+fp):.1%}  |  "
          f"Recall (caught actual successes): {tp/(tp+fn):.1%}")


# ── Mode: errors ─────────────────────────────────────────────────────────────

def mode_errors():
    wrong = test_df[test_df['correct'] == 0].copy()
    wrong = wrong.sort_values('prob_success', ascending=False)

    false_pos = wrong[wrong['predicted'] == 1]   # model said success, was failure
    false_neg = wrong[wrong['predicted'] == 0]   # model said failure, was success

    print(f"\n{'─'*100}")
    print(f"FALSE POSITIVES ({len(false_pos)})  —  model predicted SUCCESS but actual FAILURE")
    print(f"These are undercuts the model thought would work but didn't")
    print(f"{'─'*100}")
    for _, row in false_pos.iterrows():
        print(_row_str(row))

    print(f"\n{'─'*100}")
    print(f"FALSE NEGATIVES ({len(false_neg)})  —  model predicted FAILURE but actual SUCCESS")
    print(f"These are undercuts the model missed / didn't believe in")
    print(f"{'─'*100}")
    for _, row in false_neg.sort_values('prob_success').iterrows():
        print(_row_str(row))


# ── Mode: driver ─────────────────────────────────────────────────────────────

def mode_driver(driver_code):
    driver_df = test_df[test_df['driver'] == driver_code.upper()].copy()
    if driver_df.empty:
        available = sorted(test_df['driver'].unique())
        print(f"Driver '{driver_code}' not found in 2024 holdout.")
        print(f"Available: {', '.join(available)}")
        return

    print(f"\n{'─'*100}")
    print(f"2024 predictions for driver {driver_code.upper()}  "
          f"({len(driver_df)} undercut attempts  |  "
          f"accuracy: {driver_df['correct'].mean():.1%})")
    print(f"{'─'*100}")
    for _, row in driver_df.sort_values('prob_success', ascending=False).iterrows():
        print(_row_str(row))


# ── Mode: scenario ───────────────────────────────────────────────────────────

def mode_scenario():
    """
    Build a custom scenario and predict undercut success probability.
    Sensible defaults are shown; just hit Enter to accept them.
    """
    print("\n── Custom Scenario Predictor ──")
    print("Press Enter to accept default values.\n")

    def ask(prompt, default, cast=float):
        raw = input(f"  {prompt} [{default}]: ").strip()
        return cast(raw) if raw else cast(default)

    gap_ahead          = ask("Gap to car ahead (seconds)",       3.0)
    tire_age           = ask("Your tire age (laps)",              18, int)
    car_ahead_tire_age = ask("Car-ahead tire age (laps)",         22, int)
    own_pace_raw       = ask("Your recent avg lap time (seconds)", 90.0)
    threat_pace_raw    = ask("Car-ahead recent avg lap time (s)",  90.5)
    deg_delta          = ask("Your deg rate (s/lap, ~0.05–0.15)",  0.08)
    ca_deg_delta       = ask("Car-ahead deg rate (s/lap)",         0.10)
    closing_rate       = ask("Closing rate (negative = closing)",  -0.3)
    pit_loss           = ask("Pit lane time loss (seconds)",       22.0)
    race_progress      = ask("Race progress (0=start, 1=end)",     0.5)

    print("\n  Compound options: SOFT / MEDIUM / HARD / INTERMEDIATE / WET")
    compound_raw = input("  Your compound [MEDIUM]: ").strip().upper() or 'MEDIUM'

    # Build feature vector
    tire_age_advantage = car_ahead_tire_age - tire_age
    pace_delta         = own_pace_raw - threat_pace_raw
    pit_loss_fraction  = pit_loss / own_pace_raw

    row = {f: 0.0 for f in FEATURES}
    row['gap_ahead']          = gap_ahead
    row['tire_age']           = float(tire_age)
    row['car_ahead_tire_age'] = float(car_ahead_tire_age)
    row['tire_age_advantage'] = float(tire_age_advantage)
    row['own_pace']           = own_pace_raw
    row['threat_pace']        = threat_pace_raw
    row['pace_delta']         = pace_delta
    row['deg_delta']          = deg_delta
    row['ca_deg_delta']       = ca_deg_delta
    row['closing_rate']       = closing_rate
    row['pit_loss']           = pit_loss
    row['pit_loss_fraction']  = pit_loss_fraction
    row['race_progress']      = race_progress

    compound_col = f'compound_{compound_raw}'
    if compound_col in row:
        row[compound_col] = 1.0

    X_scenario = np.array([[row[f] for f in FEATURES]])
    prob = model.predict_proba(X_scenario)[0, 1]

    # Sensitivity: what changes if gap halves / doubles?
    row_half = dict(row); row_half['gap_ahead'] = gap_ahead / 2
    row_double = dict(row); row_double['gap_ahead'] = gap_ahead * 2
    prob_half   = model.predict_proba(np.array([[row_half[f]   for f in FEATURES]]))[0, 1]
    prob_double = model.predict_proba(np.array([[row_double[f] for f in FEATURES]]))[0, 1]

    verdict = 'LIKELY SUCCESS' if prob >= 0.5 else 'LIKELY FAILURE'
    bar_len  = int(prob * 30)
    bar      = '█' * bar_len + '░' * (30 - bar_len)

    print(f"\n{'─'*60}")
    print(f"  P(undercut success) = {prob:.1%}  [{bar}]")
    print(f"  Verdict: {verdict}")
    print(f"{'─'*60}")
    print(f"  Scenario inputs:")
    print(f"    Gap ahead:        {gap_ahead:.1f}s")
    print(f"    Tire ages:        yours={tire_age}L  car_ahead={car_ahead_tire_age}L  "
          f"(advantage = {tire_age_advantage:+d} laps)")
    print(f"    Pace delta:       {pace_delta:+.3f}s (negative = you're faster)")
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
        driver = sys.argv[2] if len(sys.argv) > 2 else 'VER'
        mode_driver(driver)
    else:
        print(f"Unknown mode '{mode}'. Use: inspect | errors | scenario | driver <CODE>")
