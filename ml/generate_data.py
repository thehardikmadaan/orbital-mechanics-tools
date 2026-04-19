# ml/generate_data.py

import csv
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.astrodynamics import hohmann_transfer, bi_elliptic_transfer, phasing_maneuver
from core.rocket_math import calculate_initial_mass

# All engine ISP values from ENGINE_PRESETS in rocket_math.py.
# Sampled uniformly so the model learns the rocket equation across the full
# range of chemical and electric propulsion systems.
ENGINE_ISPS = [220, 260, 310, 450, 1600, 3000]

# Per-body physics constants and valid altitude ranges
BODY_CONFIGS = {
    "Earth": {
        "mu":        3.986004418e14,
        "r_body_m":  6_371_000,
        "min_alt_km": 200,
        "max_alt_km": 400_000,
        "max_rb_km":  600_000,
    },
    "Moon": {
        "mu":        4.9048695e12,
        "r_body_m":  1_737_400,
        "min_alt_km": 20,
        "max_alt_km": 70_000,
        "max_rb_km":  100_000,
    },
    "Mars": {
        "mu":        4.282837e13,
        "r_body_m":  3_389_500,
        "min_alt_km": 150,
        "max_alt_km": 400_000,
        "max_rb_km":  600_000,
    },
}

# Sampling weights — Earth dominates real-world missions
BODY_WEIGHTS = [0.60, 0.20, 0.20]
BODY_NAMES   = ["Earth", "Moon", "Mars"]


def generate_data(num_samples=150_000):
    file_path = os.path.join(os.path.dirname(__file__), 'orbital_data.csv')

    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Body", "Maneuver_Type",
            "R1_km", "R2_km", "Rb_km", "Phase_Angle",
            "Payload_kg", "Isp", "Delta_V_ms", "Propellant_kg"
        ])

        generated = 0
        while generated < num_samples:
            body_name = random.choices(BODY_NAMES, weights=BODY_WEIGHTS, k=1)[0]
            cfg       = BODY_CONFIGS[body_name]
            mu        = cfg["mu"]
            r_body_m  = cfg["r_body_m"]
            min_alt   = cfg["min_alt_km"]
            max_alt   = cfg["max_alt_km"]
            max_rb    = cfg["max_rb_km"]

            m_type  = random.choice(["Hohmann", "Bi-Elliptic", "Phasing"])
            alt1    = random.uniform(min_alt, max_alt)
            payload = random.uniform(100, 100_000)
            isp     = random.choice(ENGINE_ISPS)
            r1      = (alt1 * 1000) + r_body_m

            alt2, altb, phase, delta_v = 0.0, 0.0, 0.0, 0.0

            try:
                if m_type == "Hohmann":
                    alt2 = random.uniform(min_alt, max_alt)
                    # Require meaningful orbit separation: ≥500 km or ≥2% of alt1
                    if abs(alt2 - alt1) < max(500, alt1 * 0.02):
                        continue
                    delta_v = hohmann_transfer(mu, r1, (alt2 * 1000) + r_body_m)

                elif m_type == "Bi-Elliptic":
                    alt2 = random.uniform(min_alt, max_alt)
                    # Bi-Elliptic only makes physical sense for large orbit ratios.
                    # Requiring ratio ≥ 3 removes degenerate cases where Hohmann
                    # would always be cheaper, keeping the model's decision boundary clean.
                    ratio = max(alt2, alt1) / max(min(alt2, alt1), 1.0)
                    if ratio < 3.0:
                        continue
                    highest_orbit = max(alt1, alt2)
                    rb_lower = highest_orbit + 10_000
                    if rb_lower >= max_rb:
                        continue
                    altb = random.uniform(rb_lower, max_rb)
                    delta_v = bi_elliptic_transfer(
                        mu, r1,
                        (alt2 * 1000) + r_body_m,
                        (altb * 1000) + r_body_m
                    )

                elif m_type == "Phasing":
                    phase   = random.uniform(1, 180)
                    delta_v = phasing_maneuver(mu, r1, phase)

                if delta_v <= 0:
                    continue

                wet_mass   = calculate_initial_mass(delta_v, isp, payload)
                propellant = wet_mass - payload
                if propellant < 0:
                    continue

                writer.writerow([
                    body_name, m_type,
                    round(alt1, 2), round(alt2, 2), round(altb, 2), round(phase, 2),
                    round(payload, 2), isp, round(delta_v, 2), round(propellant, 2)
                ])
                generated += 1

            except (ValueError, ZeroDivisionError, OverflowError, ArithmeticError):
                continue

    print(f"Mission Complete! {generated} data points saved to {file_path}")


if __name__ == "__main__":
    generate_data()