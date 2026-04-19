# ml/generate_data.py
#
# Training data generator for the AI surrogate model.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE KEY PHYSICS INSIGHT — WHY WE USE CIRCULAR VELOCITIES AS FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# All three maneuver DV formulas can be expressed purely in terms of the
# circular orbital velocity  v_c = sqrt(μ / r)  at each relevant radius.
#
# Hohmann transfer:
#   v_c1 = sqrt(μ/r1),  v_c2 = sqrt(μ/r2)
#   DV = v_c1 × f(v_c2/v_c1) + v_c2 × g(v_c1/v_c2)    ← only v_c1, v_c2 matter
#
# Phasing maneuver:
#   v_c1 = sqrt(μ/r1)
#   DV = 2 × |v_phasing − v_c1|    ← only v_c1 and phase_angle matter
#
# Bi-Elliptic transfer:
#   v_c1 = sqrt(μ/r1),  v_c2 = sqrt(μ/r2),  v_cb = sqrt(μ/rb)
#   DV = f(v_c1, v_c2, v_cb)        ← all three velocities
#
# μ and r do NOT appear separately — they are always combined as v = sqrt(μ/r).
# This is the natural input space for the physics.  Giving the model (μ, r)
# separately forces it to learn the sqrt(μ/r) combination internally, which
# requires more layers and data.  Giving it v_c directly means the mapping
# from input to DV is nearly LINEAR in log space:
#
#     log(DV) ≈ log(v_c1) + f(v_c2 / v_c1)
#
# An MLP fits linear functions with a single layer, so this feature choice
# gives near-perfect accuracy with far fewer training samples.
#
# ─────────────────────────────────────────────────────────────────────────────
# ZERO SENTINELS FOR UNUSED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# log_vc2 = 0.0 for Phasing   (no target orbit — spacecraft returns to r1)
# log_vcb = 0.0 for Hohmann and Phasing   (no intermediate apogee)
# The Maneuver_Type one-hot flags tell the model which features are active.
#
# ─────────────────────────────────────────────────────────────────────────────
# TARGET: Delta_V_ms only.
# Propellant is computed at inference time with the exact Tsiolkovsky equation
# so ISP and payload never touch the model — zero approximation error.

import csv
import math
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.astrodynamics import hohmann_transfer, bi_elliptic_transfer, phasing_maneuver

# Per-body physics constants.
# Only mu and r_body_m are needed — they combine into v_c = sqrt(mu/r).
BODY_CONFIGS = {
    "Earth": {
        "mu":         3.986004418e14,   # m³/s²
        "r_body_m":   6_371_000,        # m
        "min_alt_km": 200,
        "max_alt_km": 400_000,
        "max_rb_km":  600_000,
    },
    "Moon": {
        "mu":         4.9048695e12,
        "r_body_m":   1_737_400,
        "min_alt_km": 20,
        "max_alt_km": 70_000,
        "max_rb_km":  100_000,
    },
    "Mars": {
        "mu":         4.282837e13,
        "r_body_m":   3_389_500,
        "min_alt_km": 150,
        "max_alt_km": 400_000,
        "max_rb_km":  600_000,
    },
}

BODY_WEIGHTS = [0.60, 0.20, 0.20]
BODY_NAMES   = ["Earth", "Moon", "Mars"]


def generate_data(num_samples=200_000):
    file_path = os.path.join(os.path.dirname(__file__), 'orbital_data.csv')

    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Feature columns — physics-informed velocities + categorical body flag:
        #
        #   Body        — categorical: "Earth", "Moon", or "Mars".
        #                 One-hot encoded at training time (Body_Earth etc.).
        #                 WHY one-hot instead of log(μ)?  log(μ) is a continuous
        #                 scalar, so the MLP interpolates between Earth and Moon
        #                 when it sees Mars — that interpolation is WRONG because
        #                 Mars physics is not intermediate between the two.
        #                 A one-hot flag forces the model to learn a completely
        #                 separate mapping for each body with NO cross-body bleed.
        #
        #   log_vc1   — log( sqrt(μ/r1) ) — circular velocity at parking orbit.
        #               This is the natural scale for ΔV calculations.  Together
        #               with the body one-hot the model has access to both the
        #               absolute orbital speed AND which body's physics to apply.
        #   log_vc2   — log( sqrt(μ/r2) ) — circular velocity at target orbit;
        #               0.0 for Phasing (spacecraft returns to r1)
        #   log_vcb   — log( sqrt(μ/rb) ) — circular velocity at intermediate
        #               apogee; 0.0 for Hohmann and Phasing
        #   Phase_Angle — degrees (Phasing only; 0 for Hohmann / Bi-Elliptic)
        #   Maneuver_Type — one-hot encoded at training time
        #
        # Target: Delta_V_ms (m/s)  — log1p transform applied in train_model.py
        writer.writerow([
            "Body", "Maneuver_Type",
            "log_vc1", "log_vc2", "log_vcb", "Phase_Angle",
            "Delta_V_ms"
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

            m_type = random.choice(["Hohmann", "Bi-Elliptic", "Phasing"])
            alt1   = random.uniform(min_alt, max_alt)
            r1_m   = (alt1 * 1000) + r_body_m

            # v_c1 = sqrt(μ/r1) — the fundamental speed unit for this orbit
            vc1 = math.sqrt(mu / r1_m)

            r2_m, rb_m, vc2, vcb, phase, delta_v = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            try:
                if m_type == "Hohmann":
                    # 8% same-orbit cases teach the model the zero-ΔV boundary.
                    # Without these, the model never sees ΔV=0 and extrapolates
                    # incorrectly for nearly identical orbits.
                    if random.random() < 0.08:
                        alt2 = alt1
                    else:
                        alt2 = random.uniform(min_alt, max_alt)
                    r2_m    = (alt2 * 1000) + r_body_m
                    vc2     = math.sqrt(mu / r2_m)
                    delta_v = hohmann_transfer(mu, r1_m, r2_m)
                    # vcb = 0.0 (no intermediate orbit)

                elif m_type == "Bi-Elliptic":
                    alt2 = random.uniform(min_alt, max_alt)
                    r2_m = (alt2 * 1000) + r_body_m

                    # Bi-Elliptic only beats Hohmann when the orbit ratio > 3×;
                    # below that it uses more fuel so it's not a realistic choice.
                    ratio = max(alt2, alt1) / max(min(alt2, alt1), 1.0)
                    if ratio < 3.0:
                        continue

                    highest_orbit = max(alt1, alt2)
                    rb_lower = highest_orbit + 10_000
                    if rb_lower >= max_rb:
                        continue

                    altb  = random.uniform(rb_lower, max_rb)
                    rb_m  = (altb * 1000) + r_body_m
                    vc2   = math.sqrt(mu / r2_m)
                    vcb   = math.sqrt(mu / rb_m)
                    delta_v = bi_elliptic_transfer(mu, r1_m, r2_m, rb_m)

                elif m_type == "Phasing":
                    phase   = random.uniform(1, 180)
                    # vc2 = 0.0 and vcb = 0.0 — spacecraft returns to r1
                    delta_v = phasing_maneuver(mu, r1_m, phase)

                if delta_v < 0:
                    continue

                # Log-transform the velocities.  For unused slots (vc2=0 for
                # Phasing, vcb=0 for Hohmann/Phasing) we write 0.0 as
                # sentinel — the maneuver one-hot flags tell the model when
                # to treat those slots as irrelevant.
                log_vc1 = math.log(vc1)
                log_vc2 = math.log(vc2) if vc2 > 0 else 0.0
                log_vcb = math.log(vcb) if vcb > 0 else 0.0

                writer.writerow([
                    body_name, m_type,   # one-hot encoded in train_model.py
                    round(log_vc1, 6),
                    round(log_vc2, 6),
                    round(log_vcb, 6),
                    round(phase,   2),
                    round(delta_v, 4),
                ])
                generated += 1

            except (ValueError, ZeroDivisionError, OverflowError, ArithmeticError):
                continue

    print(f"Mission Complete! {generated} data points saved to {file_path}")


if __name__ == "__main__":
    generate_data()
