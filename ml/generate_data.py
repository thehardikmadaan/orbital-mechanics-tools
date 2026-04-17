# ml/generate_data.py

import csv
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.astrodynamics import hohmann_transfer, bi_elliptic_transfer, phasing_maneuver
from core.rocket_math import calculate_initial_mass

def generate_data(num_samples = 100000):
    # Mission Constraints
    mu = 3.986e14  # Earth's gravity
    r_earth = 6371000  # Radius in meters
    isp = 310  # Average engine efficiency

    file_path = os.path.join(os.path.dirname(__file__), 'orbital_data.csv')
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Features (Inputs) and Target (Output)
        writer.writerow(["Maneuver_Type", "R1_km", "R2_km", "Rb_km", "Phase_Angle", "Payload_kg", "Delta_V_ms", "Propellant_kg"])

        for _ in range(num_samples):
            # 1. Choose a random maneuver profile
            m_type = random.choice(["Hohmann", "Bi-Elliptic", "Phasing"])

            # 2. Randomize Inputs based on typical mission envelopes
            alt1 = random.uniform(200, 2000)  # LEO Parking (Up to Medium Earth Orbit)
            payload = random.uniform(100, 100000)  # From tiny CubeSats up to Starship!
            r1 = (alt1 * 1000) + r_earth

            # Initialize empty variables
            alt2, altb, phase, delta_v = 0, 0, 0, 0

            # 3. Apply Ground Truth Physics
            if m_type == "Hohmann":
                # Expand to Lunar distance!
                alt2 = random.uniform(2000, 400000)
                delta_v = hohmann_transfer(mu, r1, (alt2 * 1000) + r_earth)

            elif m_type == "Bi-Elliptic":
                # Target anywhere from LEO to Lunar
                alt2 = random.uniform(2000, 400000)
                # CRITICAL MATH RULE: altb MUST be larger than alt2.
                # We force it to be at least 10,000 km further out than the target!
                altb = random.uniform(alt2 + 10000, 500000)
                delta_v = bi_elliptic_transfer(mu, r1, (alt2 * 1000) + r_earth, (altb * 1000) + r_earth)

            elif m_type == "Phasing":
                # Phase angles from a tiny 1 degree shift up to half an orbit (180 degrees)
                phase = random.uniform(1, 180)
                delta_v = phasing_maneuver(mu, r1, phase)

            # 4. Calculate Fuel Mass
            wet_mass = calculate_initial_mass(delta_v, isp, payload)
            propellant = wet_mass - payload

            # 5. Save the row
            writer.writerow([m_type, round(alt1, 2), round(alt2, 2), round(altb, 2), round(phase, 2), round(payload, 2), round(delta_v, 2), round(propellant, 2)])

    print(f"Mission Complete! {num_samples} data points saved to {file_path}")


if __name__ == "__main__":
    generate_data()