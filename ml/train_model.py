import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from sklearn.model_selection import cross_val_score

def train_surrogate_model():
    print("Training surrogate model...")

    # 1. Load the data
    current_folder = os.path.dirname(__file__)
    csv_path = os.path.join(current_folder, 'orbital_data.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV.")

    if df.empty:
        raise ValueError("The CSV file is empty. Please provide valid training data.")

    # 2. Drop NaN rows and validate
    df = df.dropna()
    print(f"{len(df)} rows remaining after dropping NaN values.")

    if df.empty:
        raise ValueError(
            "All rows were dropped by dropna(). "
            "Your CSV likely contains NaN values in every row. "
            "Please inspect and clean 'orbital_data.csv'."
        )

    # 3. Validate required columns exist before transforming
    required_columns = ['Maneuver_Type', 'Propellant_kg', 'Delta_V_ms']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following required columns are missing from the CSV: {missing_cols}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # 4. Preprocess: One-Hot Encoding
    df = pd.get_dummies(df, columns=['Maneuver_Type'])

    # 5. Separate features (X) and target (y)
    y = df['Propellant_kg']
    X = df.drop(columns=['Propellant_kg', 'Delta_V_ms'])

    print(f"Training with {len(X)} samples and {X.shape[1]} features.")

    # 6. Split into 80% training / 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 7. Train
    print("Training Random Forest Regressor... Please wait.")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # CROSS-VALIDATION CHECK
    print("\nRunning 5-Fold Cross Validation... (This might take 10 seconds)")
    # cv=5 means it trains and tests 5 separate times on different data chunks
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Scores for each fold: {cv_scores}")
    print(f"True Average Accuracy (Cross-Validated): {cv_scores.mean():.4f}")

    # 8. Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nMODEL PERFORMANCE")
    print(f"Mean Absolute Error: {mae:.2f} kg (How far off the AI is on average)")
    print(f"Accuracy Score (R2): {r2:.4f} (1.0 is perfect)")

    # 9. Save the trained model
    model_path = os.path.join(current_folder, 'surrogate_model.pkl')
    columns_path = os.path.join(current_folder, 'model_columns.pkl')

    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)
    print(f"\nModel saved successfully as: {model_path}")


if __name__ == "__main__":
    train_surrogate_model()

