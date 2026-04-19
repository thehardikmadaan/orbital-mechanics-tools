# Orbital Mechanics Tools & AI Surrogate Dashboard

## Overview
This repository contains a suite of orbital mechanics tools for calculating maneuvers, rocket propellant requirements, and visualizing orbital transfers. It features a modern GUI dashboard (HMI) built with PySide6 and an AI-powered surrogate model that predicts propellant requirements based on mission parameters.

## Features
- **Orbital Math Engine:** Precision calculations for Hohmann, Bi-Elliptic, and Phasing maneuvers.
- **Rocket Equation Solver:** Computes initial mass and propellant requirements using the Tsiolkovsky Rocket Equation.
- **Interactive Dashboard:** A high-fidelity GUI for mission planning and real-time calculation.
- **2D Orbital Visualization:** Scale-accurate rendering of trajectories using Matplotlib.
- **AI Surrogate Model:** A neural network (MLPRegressor) that provides fast approximations for mission fuel costs.

## Tech Stack
- **Language:** Python 3.10+
- **GUI Framework:** PySide6 (Qt for Python)
- **Data & Math:** NumPy, Pandas, Scikit-learn, Scipy
- **Visualization:** Matplotlib
- **Serialization:** Joblib

## Requirements
Ensure you have the following installed:
- Python 3.10 or higher
- `pip` (Python package manager)

### Dependencies
Currently, the dependencies are listed below (you can install them manually or populate `requirements.txt`):
- `PySide6`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `joblib`

## Setup & Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd orbital-mechanics-tools
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install PySide6 numpy pandas matplotlib scikit-learn joblib
   ```

## Usage
### Running the Dashboard
The main entry point for the application is `main.py`.
```bash
python main.py
```

### Machine Learning Pipeline
You can retrain the AI surrogate model using the provided scripts in the `ml/` directory.

1. **Generate synthetic data:**
   ```bash
   python ml/generate_data.py
   ```
   This creates `ml/orbital_data.csv` with 100,000 mission scenarios.

2. **Train the surrogate model:**
   ```bash
   python ml/train_model.py
   ```
   This trains a neural network and saves the model to `ml/surrogate_model.pkl`.

## Project Structure
- `main.py`: Application entry point.
- `core/`: Core physics and astrodynamics engines.
  - `astrodynamics.py`: Maneuver math (Hohmann, Bi-Elliptic, etc.).
  - `rocket_math.py`: Tsiolkovsky rocket equation and mass calculations.
- `ui/`: User interface components.
  - `main_window.py`: Primary PySide6 dashboard implementation.
- `visualization/`: Plotting and asset modules.
  - `plot_orbit.py`: Matplotlib-based orbital trajectory rendering.
- `ml/`: Machine learning pipeline.
  - `generate_data.py`: Synthetic dataset generation.
  - `train_model.py`: Neural network training script.
  - `*.pkl`: Saved model weights and scalers.

## Environment Variables
- TODO: Document any environment variables used by the system (none currently identified).

## Tests
- TODO: Implement automated tests (e.g., using `pytest` or `unittest`) to verify physics engine accuracy.

## License
- TODO: Add license information.
