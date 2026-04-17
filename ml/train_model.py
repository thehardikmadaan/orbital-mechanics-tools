import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


def train_surrogate_model():
    print("Initiating AI Surrogate Model Training...")

    current_folder = os.path.dirname(__file__)
    csv_path = os.path.join(current_folder, 'orbital_data.csv')

    df = pd.read_csv(csv_path)
    df = df.dropna()
    print(f"Loaded {len(df)} perfect rows.")

    df = pd.get_dummies(df, columns=['Maneuver_Type'])

    y = df['Propellant_kg']
    X = df.drop(columns=['Propellant_kg', 'Delta_V_ms'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Neural Network with Target Transformer... Please wait.")

    # 1. The Base Pipeline (Scales inputs)
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('nn', MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=2000, random_state=42))
    ])

    # 2. The Target Transformer (Fixes the "Whale Problem" using logs)
    model = TransformedTargetRegressor(
        regressor=base_pipeline,
        func=np.log1p,  # Flattens the exponential curve for training
        inverse_func=np.expm1  # Automatically un-flattens it when the UI asks for a prediction!
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- MODEL PERFORMANCE ---")
    print(f"Mean Absolute Error: {mae:.2f} kg")
    print(f"Accuracy Score (R2): {r2:.4f}")

    # Save the pipeline
    model_path = os.path.join(current_folder, 'surrogate_model.pkl')
    columns_path = os.path.join(current_folder, 'model_columns.pkl')
    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)
    print(f"\nModel Pipeline saved successfully!")


if __name__ == "__main__":
    train_surrogate_model()