# ml/train_model.py
#
# AI Surrogate Model — predicts Delta-V (m/s) from orbital geometry.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY PREDICT DELTA-V, NOT PROPELLANT?
# ─────────────────────────────────────────────────────────────────────────────
# Propellant = payload × (exp(ΔV / (Isp × g₀)) − 1)   [Tsiolkovsky equation]
#
# This is exact.  The model predicts ΔV from orbital geometry.  The rocket
# equation runs at inference time — ISP and payload never touch the model.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY NO BODY FLAG? — THE CORE PHYSICS ARGUMENT
# ─────────────────────────────────────────────────────────────────────────────
# For a Hohmann transfer, setting k = v_c1/v_c2 = sqrt(r2/r1):
#
#   ΔV₁ = v_c1 × (k × sqrt(2/(k²+1)) − 1)
#   ΔV₂ = v_c2 × (1 − sqrt(2/(k²+1)))
#
# The body's gravitational parameter μ and the individual radii r1, r2 do NOT
# appear — they cancel.  The same identity holds for Phasing and Bi-Elliptic.
#
# Consequence: a body one-hot flag is WRONG — it teaches the model that Earth
# and Moon at the same (v_c1, v_c2) pair have different ΔV, but they don't.
# Using log(μ) as a continuous body identifier is equally wrong: the model
# interpolates between Earth and Moon for Mars, but orbital mechanics are not
# a linear blend across bodies.
#
# The correct inputs are ONLY the circular velocities:
#   Hohmann    : (log_vc1, log_vc2)
#   Phasing    : (log_vc1, phase_angle)
#   Bi-Elliptic: (log_vc1, log_vc2, log_vcb)
#
# At inference time, main_window.py computes v_c = sqrt(μ/r) using the
# selected body's μ and converts altitude to total orbital radius — exactly
# the same transformation used in generate_data.py.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY LOG SPACE?
# ─────────────────────────────────────────────────────────────────────────────
# The mapping log(ΔV) ≈ log(v_c1) + smooth_function(log(v_c2/v_c1)) is nearly
# linear.  An MLP fits linear functions trivially.  With log1p on the target
# and log on the inputs, both sides are in log space → exceptional accuracy.
#
# Run to retrain:
#   python ml/train_model.py

import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_surrogate_model():
    print("=" * 60)
    print("  ORBITAL MECHANICS — AI SURROGATE MODEL TRAINING")
    print("  Target  : Delta-V (m/s) via log1p transform")
    print("  Features: log(vc1), log(vc2), log(vcb), phase_angle")
    print("  No body flag — ΔV = f(v_c) exactly, body-independent")
    print("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    current_folder = os.path.dirname(__file__)
    csv_path = os.path.join(current_folder, 'orbital_data.csv')

    print(f"\n[1/5] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).dropna()
    print(f"      Loaded {len(df):,} mission scenarios.")
    print(f"      Maneuver types: {df['Maneuver_Type'].value_counts().to_dict()}")
    print(f"      ΔV range:       {df['Delta_V_ms'].min():.2f} – {df['Delta_V_ms'].max():.2f} m/s")
    print(f"      log_vc1 range:  {df['log_vc1'].min():.3f} – {df['log_vc1'].max():.3f}")

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    print("\n[2/5] Preparing features and target...")

    # One-hot encode maneuver type only — NO body flag (see header above).
    # The maneuver flag tells the model which velocity slots are active.
    df = pd.get_dummies(df, columns=['Maneuver_Type'])

    y = df['Delta_V_ms']

    # Features:
    #   log_vc1  — log(v_c1) at parking orbit; always valid
    #   log_vc2  — log(v_c2) at target orbit;  0.0 sentinel for Phasing
    #   log_vcb  — log(v_cb) at intermediate apogee; 0.0 for Hohmann/Phasing
    #   Phase_Angle — degrees (Phasing only; 0 otherwise)
    #   Maneuver_Type_* — one-hot: activates the right velocity slot(s)
    X = df.drop(columns=['Delta_V_ms'])

    print(f"      Feature columns ({len(X.columns)}): {X.columns.tolist()}")

    # ── 3. Train / Test Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[3/5] Split: {len(X_train):,} training / {len(X_test):,} test samples.")

    # ── 4. Build Model Pipeline ───────────────────────────────────────────────
    print("\n[4/5] Training Neural Network...")

    # StandardScaler normalises all features to mean=0, std=1.
    # This is important because log_vc1 ≈ 5.6–9.0 while Phase_Angle ≈ 0–180 —
    # very different scales that would dominate Adam's gradient otherwise.
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nn', MLPRegressor(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            learning_rate='adaptive',
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            tol=1e-5,
            max_iter=3000,
            random_state=42,
        ))
    ])

    # log1p on ΔV compresses the 0–7784 m/s range to 0–9.0 in log1p.
    # Combined with log-space inputs, the full mapping is nearly linear.
    model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model.fit(X_train, y_train)
    print("      Training complete.")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on held-out test set...")
    dv_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, dv_pred)
    rmse = math.sqrt(mean_squared_error(y_test, dv_pred))
    r2   = r2_score(y_test, dv_pred)
    # Clip true values at 1 m/s so near-zero DV cases don't inflate MAPE
    mape = np.mean(np.abs((y_test - dv_pred) / np.clip(y_test, 1, None))) * 100

    print("\n" + "─" * 50)
    print("  DELTA-V PREDICTION — TEST SET")
    print("─" * 50)
    print(f"  Mean Absolute Error (MAE) : {mae:>10,.2f} m/s")
    print(f"  Root Mean Sq. Error (RMSE): {rmse:>10,.2f} m/s")
    print(f"  Mean Abs. % Error (MAPE)  : {mape:>9.2f} %")
    print(f"  Accuracy Score (R²)       : {r2:>10.4f}  (1.0 = perfect)")
    print("─" * 50)

    if r2 > 0.999:
        print("  ✓ Exceptional fit.")
    elif r2 > 0.99:
        print("  ✓ Excellent fit — ready for production use.")
    elif r2 > 0.95:
        print("  ✓ Good fit — suitable for mission planning estimates.")
    else:
        print("  ⚠ Moderate fit — consider more training data or features.")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    model_path   = os.path.join(current_folder, 'surrogate_model.pkl')
    columns_path = os.path.join(current_folder, 'model_columns.pkl')

    joblib.dump(model, model_path)
    # Column list lets main_window.py reindex its inference DataFrame to the
    # exact column order the model was trained on.
    joblib.dump(X.columns.tolist(), columns_path)

    print(f"\n  Model saved   → {model_path}")
    print(f"  Columns saved → {columns_path}")
    print("\n" + "=" * 60)
    print("  Done. Restart the dashboard to load the new model.")
    print("=" * 60)


if __name__ == "__main__":
    train_surrogate_model()
