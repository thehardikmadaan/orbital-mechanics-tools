# ml/train_model.py
#
# AI Surrogate Model — predicts Delta-V (m/s) from orbital geometry.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY PREDICT DELTA-V, NOT PROPELLANT?
# ─────────────────────────────────────────────────────────────────────────────
# Propellant = payload × (exp(ΔV / (Isp × g₀)) − 1)   [Tsiolkovsky equation]
#
# This is exact.  Having the model approximate it adds error.  Instead the
# model predicts ΔV from orbital geometry, then the exact rocket equation runs
# at inference time — engine type and payload mass never touch the model.
#
# ─────────────────────────────────────────────────────────────────────────────
# FEATURE DESIGN — CIRCULAR ORBITAL VELOCITIES
# ─────────────────────────────────────────────────────────────────────────────
# Every ΔV formula ultimately depends on circular orbital velocities:
#
#     v_c = sqrt(μ / r)
#
# For a Hohmann transfer the total ΔV can be written as f(v_c1, v_c2) with NO
# separate dependence on μ or r individually — the ratio μ/r is the only
# physics that matters and v_c already encodes it.  The same holds for
# Phasing (f(v_c1, phase_angle)) and Bi-Elliptic (f(v_c1, v_c2, v_cb)).
#
# In log space the relationship is approximately linear:
#
#     log(ΔV) ≈ log(v_c1) + smooth_function_of(v_c2 / v_c1)
#
# An MLP fits linear functions trivially.  By providing log(v_c) as features
# and applying log1p to the target, both sides of the mapping are in log
# space → the network learns a nearly linear function → near-perfect accuracy
# across Earth, Moon, and Mars without any body-specific parameters.
#
# Run to retrain after regenerating data:
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
    print("  Features: log(v_c1), log(v_c2), log(v_cb), phase")
    print("  Why     : ΔV = f(v_c) — linear in log velocity space")
    print("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    current_folder = os.path.dirname(__file__)
    csv_path = os.path.join(current_folder, 'orbital_data.csv')

    print(f"\n[1/5] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).dropna()
    print(f"      Loaded {len(df):,} mission scenarios.")
    print(f"      Maneuver types: {df['Maneuver_Type'].value_counts().to_dict()}")
    print(f"      ΔV range:       {df['Delta_V_ms'].min():.2f} – {df['Delta_V_ms'].max():.2f} m/s")
    print(f"      log_vc1 range:  {df['log_vc1'].min():.4f} – {df['log_vc1'].max():.4f}")
    print(f"      Bodies: {df['Body'].value_counts().to_dict()}")

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    print("\n[2/5] Preparing features and target...")

    # One-hot encode BOTH body and maneuver type.
    # Body one-hot is preferred over log(μ) because log(μ) is a continuous
    # scalar → the MLP would interpolate between Earth and Moon for Mars.
    # Orbital mechanics per body are NOT linear blends of each other.
    # A one-hot forces the model to learn a separate mapping per body.
    df = pd.get_dummies(df, columns=['Body', 'Maneuver_Type'])

    y = df['Delta_V_ms']

    # Features:
    #   Body_Earth / Body_Moon / Body_Mars — one-hot body selector; forces
    #              body-specific learning without cross-body interpolation
    #   log_vc1  — log(v_c1) — circular velocity at parking orbit; always valid
    #   log_vc2  — log(v_c2) — circular velocity at target orbit;
    #              0.0 for Phasing (model ignores via Phasing flag)
    #   log_vcb  — log(v_cb) — circular velocity at intermediate apogee;
    #              0.0 for Hohmann and Phasing (model ignores via their flags)
    #   Phase_Angle — degrees; 0 for Hohmann and Bi-Elliptic
    #   Maneuver_Type_* — one-hot: tells model which velocity slots are active
    X = df.drop(columns=['Delta_V_ms'])

    print(f"      Feature columns ({len(X.columns)}): {X.columns.tolist()}")

    # ── 3. Train / Test Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[3/5] Split: {len(X_train):,} training / {len(X_test):,} test samples.")

    # ── 4. Build Model Pipeline ───────────────────────────────────────────────
    print("\n[4/5] Training Neural Network...")

    # StandardScaler normalises features to mean=0, std=1.
    # Even in log velocity space the features still have different scales
    # (log_vc1 ≈ 6–9, Phase_Angle ≈ 0–180), so scaling is important for Adam.
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

    # log1p on ΔV: the target range 0–6000 m/s compresses to 0–8.7 in log1p.
    # Combined with log-space input features both sides of the mapping are
    # in log space → the network learns a nearly linear function.
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
    # Clip true values at 1 m/s to avoid divide-by-zero on the zero-ΔV cases
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
    # exact column order the model was trained on (important after pd.get_dummies).
    joblib.dump(X.columns.tolist(), columns_path)

    print(f"\n  Model saved   → {model_path}")
    print(f"  Columns saved → {columns_path}")
    print("\n" + "=" * 60)
    print("  Done. Restart the dashboard to load the new model.")
    print("=" * 60)


if __name__ == "__main__":
    train_surrogate_model()
