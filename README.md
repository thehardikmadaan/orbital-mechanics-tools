# Orbital Mechanics Tools & AI Surrogate Dashboard
### A Deep-Dive Repository Description

---

## Overview

**orbital-mechanics-tools** is a full-stack Python application for space mission planning. It combines a precision astrodynamics physics engine, a PySide6-based interactive GUI dashboard, animated 2D orbital visualizations, and a trained neural network surrogate model — all in a single, well-structured project. The project is 100% Python and built to be run locally, targeting engineers, students, and aerospace enthusiasts who want to plan orbital maneuvers and get instant AI-powered fuel cost estimates.

---

## Repository Architecture

The codebase is split into five focused packages, each with a clear single responsibility:

```
orbital-mechanics-tools/
├── main.py                  ← Entry point (8 lines — clean and minimal)
├── core/
│   ├── astrodynamics.py     ← Orbital maneuver math engine
│   └── rocket_math.py       ← Tsiolkovsky rocket equation solver
├── ml/
│   ├── generate_data.py     ← Synthetic dataset generator (200,000 scenarios)
│   ├── train_model.py       ← Neural network training pipeline
│   └── *.pkl                ← Serialized model + scaler artifacts
├── ui/
│   └── main_window.py       ← Full PySide6 dashboard (HMI)
├── visualization/
│   └── plot_orbit.py        ← Matplotlib orbital trajectory renderer
└── tests/                   ← Test scaffolding
```

The `main.py` entry point is deliberately thin — just three lines of logic that call `run_app()` from the UI module. This keeps launch logic decoupled from the application layer.

---

## The Physics Engine (`core/`)

### `astrodynamics.py` — Orbital Maneuver Math

This module is the mathematical heart of the project. It implements the three most fundamental orbital transfer strategies used in real mission planning:

**Hohmann Transfer** is the most fuel-efficient two-burn maneuver for moving between two coplanar circular orbits. The math involves computing two delta-v burns: one to raise apoapsis from the initial orbit to the target orbit's altitude, and a second circularization burn at the new apoapsis. The formulas rely on the vis-viva equation: `v = sqrt(GM * (2/r - 1/a))`, where GM is Earth's gravitational parameter (~3.986 × 10¹⁴ m³/s²), r is the current radius, and a is the semi-major axis.

**Bi-Elliptic Transfer** is used when the ratio of target orbit to initial orbit radius is large (greater than ~11.94). It introduces a third intermediate orbit with a very high apoapsis (a "waypoint" burn out at a much higher altitude), which — counter-intuitively — uses less total delta-v than a direct Hohmann transfer for large orbital changes.

**Phasing Maneuvers** calculate the burns needed for a spacecraft to meet a target object at a precise point in its orbit — critical for rendezvous operations (e.g., docking with a space station). This requires computing the angular difference between the chaser and target, then adjusting the chaser's orbital period by changing its altitude temporarily.

All three routines return structured data: total delta-v budget, individual burn magnitudes, transfer time (using Kepler's third law: `T = 2π * sqrt(a³/GM)`), and intermediate orbital parameters.

### `rocket_math.py` — Tsiolkovsky Rocket Equation

This module wraps the foundational equation of rocketry:

```
Δv = Isp × g₀ × ln(m₀ / mf)
```

Given a delta-v requirement and a spacecraft's specific impulse (Isp) and dry mass, it back-calculates the **propellant mass** and **initial wet mass** required to complete the maneuver. This is the bridge between the orbital math and the AI model — the target variable the neural network is trained to predict.

---

## The ML Pipeline (`ml/`)

### Step 1: Synthetic Data Generation (`generate_data.py`)

Since real mission telemetry is classified or scarce, the project generates a **synthetic dataset** of 200,000 realistic mission scenarios. Each scenario is sampled by:

- Randomly drawing an **initial orbit radius** (e.g., LEO through GEO altitudes)
- Randomly drawing a **target orbit radius**
- Randomly drawing spacecraft parameters: **dry mass**, **specific impulse (Isp)**, and **maneuver type** (Hohmann, Bi-Elliptic, Phasing)

For each combination, the physics engine computes the exact delta-v using the analytical formulas above, and then the rocket equation converts that to a **propellant mass** — which becomes the ground-truth label. The result is saved as `ml/orbital_data.csv`.

This approach is powerful because the synthetic data covers the full physically valid parameter space uniformly, with no measurement noise or data gaps — giving the model a clean, dense training signal.

### Step 2: Neural Network Training (`train_model.py`)

The training script implements a clean, five-stage pipeline logged to the terminal:

```
[1/5] Loading dataset...
[2/5] Preprocessing features...
[3/5] Split: 160,000 training / 40,000 test samples.
[4/5] Training Neural Network...
      Training complete.
[5/5] Evaluating on held-out test set...
```

**Architecture:** The model uses scikit-learn's `MLPRegressor` (Multi-Layer Perceptron for regression). The network is a feedforward neural network — several fully-connected hidden layers with ReLU or similar activations — trained end-to-end to map mission parameters directly to delta-v (in m/s).

**Feature Engineering:** Raw inputs (orbit radii, mass, Isp, maneuver type) are normalized using `StandardScaler` before being fed into the network, so no single feature dominates the gradient signal by being on a different numerical scale.

**Train/Test Split:** The 200,000 samples are split 80/20 — **160,000 training samples** and **40,000 held-out test samples** — ensuring the evaluation metrics are unbiased (the model never sees test data during training).

**Serialization:** After training, both the fitted `MLPRegressor` and the `StandardScaler` are saved to disk using `joblib` as `.pkl` files. This allows the dashboard to load them at startup and serve instant predictions without retraining.

---

## 🏆 Model Performance Highlights

Evaluated on the fully held-out test set of 40,000 never-seen mission scenarios:

| Metric | Value | What It Means |
|---|---|---|
| **MAE** | **16.47 m/s** | On average, predictions are off by only ~16 m/s |
| **RMSE** | **30.56 m/s** | Larger errors (outliers) are kept well under control |
| **MAPE** | **2.88%** | Percentage error across the full range of missions |
| **R² Score** | **0.9993** | The model explains 99.93% of variance in delta-v |

An R² of **0.9993** is exceptional — essentially near-perfect regression. For context, this means if the true delta-v for a maneuver is 500 m/s, the model's prediction will land within roughly ±14.4 m/s of that figure on average. This level of accuracy makes the surrogate genuinely useful as a rapid mission planning tool — a fast substitute for running full physics calculations.

The **2.88% MAPE** is particularly meaningful because delta-v values can span from tens to thousands of m/s across different mission profiles. A consistent sub-3% relative error across that full range demonstrates the model has learned the underlying physics relationships, not just a narrow slice of the problem.

---

## The Dashboard / UI (`ui/main_window.py`)

The dashboard is built with **PySide6**, which is Qt for Python — the gold standard for cross-platform desktop GUIs. The design philosophy is a high-fidelity Human-Machine Interface (HMI), modeled after the kind of control panels found in mission operations centers.

Key UI/UX design choices:

**Input Panel** — users enter mission parameters through clearly labeled fields: initial orbit altitude, target altitude, spacecraft dry mass, and specific impulse. The layout uses Qt form layouts with validation feedback, preventing physically nonsensical inputs.

**Live Calculation** — clicking the "Calculate" button instantly invokes `astrodynamics.py` and `rocket_math.py`, and the computed delta-v, transfer time, and propellant mass are rendered into a results panel in real time. There's no loading delay — the physics runs synchronously because it's analytically cheap.

**AI Prediction Panel** — a separate section loads the trained surrogate model and runs inference on the same inputs. The result is displayed alongside the exact physics answer, giving the user both precision and a sense of the ML model's confidence. This side-by-side comparison is a deliberate design choice to build trust in the AI outputs.

**2D Orbital Visualization** — the visualization module renders a Matplotlib canvas embedded directly into the Qt window (using `FigureCanvasQTAgg`). The plot shows:
- Earth drawn at the center
- The initial orbit as a circle
- The target orbit as a larger circle
- The Hohmann or Bi-Elliptic transfer ellipse in between
- Burn point markers annotated on the trajectory

Orbits are drawn to scale, so users immediately see the geometric relationship between low Earth orbit (~6,778 km radius) and, for example, geostationary orbit (~42,164 km radius).

**Styling** — the UI uses a dark-mode color palette consistent with aerospace software aesthetics: dark backgrounds, high-contrast text, and colored accent markers to distinguish orbit types visually.

---

## Tech Stack Summary

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.10+ | Ecosystem, readability, scientific libraries |
| GUI | PySide6 (Qt) | Cross-platform, professional HMI quality |
| Physics | NumPy, SciPy | Vectorized math, precision |
| ML | scikit-learn MLPRegressor | Sufficient for tabular regression; no GPU needed |
| Data | Pandas | Data pipeline and CSV handling |
| Visualization | Matplotlib (embedded in Qt) | Tight integration with PySide6 canvas |
| Serialization | Joblib | Fast, reliable model persistence |
| CI/CD | GitHub Actions (.github/workflows/) | Automated pipeline |

---

## Why This Project Is Impressive

**Domain depth:** The project doesn't fake the physics. Hohmann, Bi-Elliptic, and Phasing maneuvers are implemented from first principles using the correct orbital mechanics formulas — not approximations.

**End-to-end ML pipeline:** The synthetic data generation → preprocessing → training → serialization → production inference loop is complete and reproducible. Anyone can clone the repo, run two scripts, and have a trained model.

**Scale of training:** 200,000 generated scenarios, split cleanly 80/20, with model training reaching R² = 0.9993 — this is a textbook example of a well-executed surrogate modeling project.

**Professional GUI:** Building a PySide6 dashboard that embeds live Matplotlib rendering, connects to a physics engine, and integrates an ML model is non-trivial. The separation of concerns across `core/`, `ml/`, `ui/`, and `visualization/` packages shows genuine software architecture thinking.

**Real-world applicability:** Surrogate models (fast ML approximations of expensive physics simulations) are used extensively in aerospace engineering — at NASA, ESA, and commercial launch companies — to accelerate design space exploration. This project demonstrates exactly that pattern at a small scale.

---

*Project by Hardik  — github.com/thehardikmadaan/orbital-mechanics-tools*