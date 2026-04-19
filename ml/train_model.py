# ml/train_model.py
# AI Surrogate Model Training Pipeline
#
# A surrogate model is a fast approximation of an expensive calculation.
# Instead of running the full physics equations every time, the neural network
# learns to predict the answer directly from the input parameters.
#
# Run this script directly to retrain the model:
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
    print("=" * 60)

    # ── 1. Load dataset ───────────────────────────────────────────────────────
    current_folder = os.path.dirname(__file__)
    csv_path = os.path.join(current_folder, 'orbital_data.csv')

    print(f"\n[1/5] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path).dropna()
    print(f"      Loaded {len(df):,} complete mission scenarios.")
    print(f"      Bodies:          {df['Body'].value_counts().to_dict()}")
    print(f"      Maneuver types:  {df['Maneuver_Type'].value_counts().to_dict()}")

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    # One-hot encode categorical variables (body + maneuver type).
    # Neural networks need numeric inputs; get_dummies handles both columns at once.
    print("\n[2/5] Preparing features and targets...")
    df = pd.get_dummies(df, columns=['Body', 'Maneuver_Type'])

    y = df['Propellant_kg']
    X = df.drop(columns=['Propellant_kg', 'Delta_V_ms'])

    print(f"      Feature columns: {X.columns.tolist()}")
    print(f"      Propellant range: {y.min():.1f} kg — {y.max():.1f} kg")

    # ── 3. Train / Test Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n[3/5] Split: {len(X_train):,} training / {len(X_test):,} test samples.")

    # ── 4. Build the Neural Network Pipeline ──────────────────────────────────
    print("\n[4/5] Training Neural Network (this may take a few minutes)...")

    # StandardScaler normalises all features to zero mean / unit variance.
    # The wider architecture (512→256→128→64) captures planet-specific
    # orbital relationships that the smaller (256→128→64) network missed.
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nn', MLPRegressor(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            learning_rate='adaptive',   # Reduces LR when training plateaus
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,        # More patience for the larger network
            tol=1e-5,                   # Tighter convergence threshold
            max_iter=3000,
            random_state=42,
        ))
    ])

    # Log transform on the target variable:
    # Propellant follows an exponential (rocket equation), so log1p brings
    # small and large values to a comparable scale for the loss function.
    # expm1 is the exact inverse — no approximation error on the way out.
    model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model.fit(X_train, y_train)
    print("      Training complete.")

    # ── 5. Evaluate Performance ───────────────────────────────────────────────
    print("\n[5/5] Evaluating on held-out test set...")
    predictions = model.predict(X_test)

    mae  = mean_absolute_error(y_test, predictions)
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    r2   = r2_score(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / np.clip(y_test, 1, None))) * 100

    print("\n" + "─" * 45)
    print("  MODEL PERFORMANCE — TEST SET")
    print("─" * 45)
    print(f"  Mean Absolute Error (MAE) : {mae:>12,.2f} kg")
    print(f"  Root Mean Sq. Error (RMSE): {rmse:>12,.2f} kg")
    print(f"  Mean Abs. % Error (MAPE)  : {mape:>11.2f} %")
    print(f"  Accuracy Score (R²)       : {r2:>12.4f}  (1.0 = perfect)")
    print("─" * 45)

    if r2 > 0.99:
        print("  ✓ Excellent fit — ready for production use.")
    elif r2 > 0.95:
        print("  ✓ Good fit — suitable for mission planning estimates.")
    else:
        print("  ⚠ Moderate fit — consider generating more training data.")

    # ── Save the Trained Pipeline ─────────────────────────────────────────────
    model_path   = os.path.join(current_folder, 'surrogate_model.pkl')
    columns_path = os.path.join(current_folder, 'model_columns.pkl')

    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)

    print(f"\n  Surrogate model saved → {model_path}")
    print(f"  Feature columns saved → {columns_path}")
    print("\n" + "=" * 60)
    print("  Training complete. Restart the dashboard to load the new model.")
    print("=" * 60)


if __name__ == "__main__":
    train_surrogate_model()