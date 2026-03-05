# Orbital Mechanics Calculator & Visualizer

## About the Project
This project is a Python-based simulation and calculation tool designed to solve fundamental orbital mechanics problems. It calculates the required Delta-v ($\Delta v$) for a Hohmann transfer orbit (e.g., from Low Earth Orbit to Geostationary Earth Orbit) and uses the Tsiolkovsky Rocket Equation to determine the necessary propellant mass. 

The project also includes a 2D visualization module to plot the Earth, the initial and final circular orbits, and the elliptical transfer trajectory.

## Features
* **Orbital Math Engine:** Calculates orbital velocities, semi-major/minor axes, and total $\Delta v$.
* **Rocket Equation Solver:** Computes initial mass and propellant requirements based on specific impulse ($I_{sp}$) and target payload mass using the formula:
  $$\Delta v = I_{sp} g_0 \ln\left(\frac{m_0}{m_f}\right)$$
* **Trajectory Visualization:** Renders scale-accurate 2D plots of the orbital transfer using Matplotlib.
* **Future Scope:** Planned integration of a graphical user interface (GUI) to make the tool interactive and user-friendly, applying Human-Machine Interaction (HMI) principles for a clean dashboard layout.

## Tech Stack
* **Python** (Core logic and math)
* **NumPy** (Array and trigonometric calculations)
* **Matplotlib** (Data visualization and orbital plotting)

## Getting Started
*(Instructions on how to install and run the program will go here once the code is finished!)*
