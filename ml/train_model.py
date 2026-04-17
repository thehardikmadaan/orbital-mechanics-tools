import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_surrogate_model():
    print("Training surrogate model...")
    #1. load the data
    current_folder = os.path.dirname(__file__)
    csv_path = os.path.join(current_folder, 'orbital_data.csv')
    df = pd.read_csv(csv_path)

    #2. preprocess the Data
    #converting the data more readable formal for ML, using one Hot encoding
    df = pd.get_dummies(df, columns=['Maneuver_Type'])

    # Separate Inputs (X) and the Target Output (y)
    # I want the AI to predict the Propellant_kg directly from the mission parameters.
    # drop Delta_V so the AI learns to skip that middle step entirely!
    y = df['Propellant_kg']
    X = df.drop(columns=['Propellant_kg', 'Delta_V_ms'])

    # Split into 80% Training data and 20% Testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3 Training
    print("Training Random Forest Regressor.... Please wait")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    #  4 Evaluate the Model (Testing it on the 20% it hasn't seen yet)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(" \n MODEL PERFORMANCE")
    print(f"Mean Absolute Error: {mae:.2f} kg (How far off the AI is on average)")
    print(f"Accuracy Score (R2): {r2:.4f} (1.0 is perfect)")

    # 5. Save the Trained Model to a file
    model_path = os.path.join(current_folder, 'surrogate_model.pkl')
    columns_path = os.path.join(current_folder, 'model_columns.pkl')

    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)
    print(f"\nModel saved successfully as: {model_path}")


if __name__ == "__main__":
    train_surrogate_model()

