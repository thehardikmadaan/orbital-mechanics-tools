# ml/train_model.py
# AI Surrogate Model — predicts Delta-V from orbital geometry.
#
# Architecture decision: predict Delta_V_ms, NOT Propellant_kg.
#
# Why: Propellant = Payload × (exp(ΔV / (Isp × g₀)) − 1)
# The rocket equation is an exact, known formula. There is no reason to have
# a neural network approximate it — that only adds error. Instead, we train
# the model on the orbital geometry problem (ΔV from radii / angles / body),
# then apply the exact rocket equation at inference time.
#
# Benefits:
#   • Isp and payload drop out of training — simpler, more generalizable model
#   • R1 = R2 → model predicts ΔV ≈ 0 → propellant = 0 exactly (no phantom fuel)
#   • ΔV spans 0–6000 m/s (4 decades) vs propellant 0–700k kg (7 decades)
#   • Changing engine type at inference always gives the correct propellant
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
    print("  Target: Delta-V (m/s)  |  Propellant via exact eq.")
    print("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    current_folder = os.path.dirname(__file__)
    csv_path = os.path.join(current_folder, 'orbital_data.csv')

    print(f"\n[1/5] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).dropna()
    print(f"      Loaded {len(df):,} mission scenarios.")
    print(f"      Bodies:         {df['Body'].value_counts().to_dict()}")
    print(f"      Maneuver types: {df['Maneuver_Type'].value_counts().to_dict()}")
    print(f"      ΔV range:       {df['Delta_V_ms'].min():.2f} – {df['Delta_V_ms'].max():.2f} m/s")

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    print("\n[2/5] Preparing features and target...")
    df = pd.get_dummies(df, columns=['Body', 'Maneuver_Type'])

    # Target: Delta-V only — physics model predicts geometry, not thermodynamics
    y = df['Delta_V_ms']

    # Features: orbital geometry + body + maneuver type
    # Payload and ISP are deliberately excluded — they don't affect delta-V
    drop_cols = ['Delta_V_ms']
    X = df.drop(columns=drop_cols)

    print(f"      Feature columns ({len(X.columns)}): {X.columns.tolist()}")

    # ── 3. Train / Test Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[3/5] Split: {len(X_train):,} training / {len(X_test):,} test samples.")

    # ── 4. Build Model Pipeline ───────────────────────────────────────────────
    print("\n[4/5] Training Neural Network...")

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

    # Log transform on ΔV: the range spans 0–6000 m/s with many small values.
    # log1p maps this to 0–8.7, giving the network a manageable scale.
    # The near-zero ΔV cases (R1 ≈ R2) map to log1p(~0) ≈ 0, so the network
    # correctly learns "same orbit → no delta-V needed".
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
        print("  ⚠ Moderate fit — consider more training data.")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    model_path   = os.path.join(current_folder, 'surrogate_model.pkl')
    columns_path = os.path.join(current_folder, 'model_columns.pkl')

    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)

    print(f"\n  Model saved  → {model_path}")
    print(f"  Columns saved → {columns_path}")
    print("\n" + "=" * 60)
    print("  Done. Restart the dashboard to load the new model.")
    print("=" * 60)


if __name__ == "__main__":
    train_surrogate_model()
