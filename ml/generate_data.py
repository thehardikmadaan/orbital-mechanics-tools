# ml/generate_data.py
#
# Training data generator for the AI surrogate model.
#
# ─────────────────────────────────────────────────────────────────────────────
# BODY-INDEPENDENT DATA GENERATION — THE FUNDAMENTAL INSIGHT
# ─────────────────────────────────────────────────────────────────────────────
# Every delta-V formula is a PURE FUNCTION of circular orbital velocities.
# There is no separate body dependence:
#
#   Hohmann    ΔV = f(v_c1, v_c2)           — exact, provable algebraically
#   Phasing    ΔV = f(v_c1, phase_angle)     — exact
#   Bi-Elliptic ΔV = f(v_c1, v_c2, v_cb)   — exact
#
# Proof sketch for Hohmann (derivation from vis-viva, v_c = sqrt(μ/r)):
#   Let k = v_c1/v_c2.  Then r2/r1 = k².
#   v_perigee = v_c1 × k × sqrt(2/(k²+1))
#   ΔV₁ = v_perigee − v_c1 = v_c1 × (k × sqrt(2/(k²+1)) − 1)
#   v_apogee  = v_c2 × sqrt(2/(k²+1))
#   ΔV₂ = v_c2 − v_apogee = v_c2 × (1 − sqrt(2/(k²+1)))
#
# Both k and the final expression contain ONLY v_c1 and v_c2 — μ and r cancel.
# An equivalent statement: for the same (v_c1, v_c2) pair, Earth at 143 000 km
# and Moon at 50 km give IDENTICAL ΔV.  A body flag would mislead the model.
#
# CONSEQUENCE FOR TRAINING
# ─────────────────────────────────────────────────────────────────────────────
# We do NOT generate data by body.  Instead we sample (v_c1, v_c2) pairs
# UNIFORMLY across the full physically realizable velocity range:
#
#   v_min = 261 m/s  (Moon circular velocity at 70 000 km altitude)
#   v_max = 7784 m/s (Earth circular velocity at 200 km altitude)
#
# Using a reference μ (Earth's value) we recover r = μ/v_c² and call the
# exact physics functions.  The result is identical for any μ because the DV
# depends only on (v_c1, v_c2), not on μ or r individually.
#
# This eliminates all body-specific lookup tables, body weights, and body
# one-hot flags.  The model trains on the true 2D mathematical function
# f(log v_c1, log v_c2) → log ΔV, which is smooth and well-conditioned.
#
# TARGET: Delta_V_ms only.
# Propellant is computed at inference time with the exact Tsiolkovsky equation.

import csv
import math
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.astrodynamics import hohmann_transfer, bi_elliptic_transfer, phasing_maneuver

# Reference μ used only to convert v_c → r for the physics function calls.
# Any positive value works because the DV formulas are body-independent when
# expressed as functions of circular velocities.
MU_REF = 3.986004418e14   # m³/s²  (Earth, chosen as the most familiar value)

# Velocity range covering ALL physically realizable circular orbits across the
# three bodies used in the application:
#   Low end : Moon at 70 000 km altitude → v_c ≈ 261 m/s
#   High end: Earth at 200 km altitude   → v_c ≈ 7784 m/s
VC_MIN = 261.0    # m/s
VC_MAX = 7784.0   # m/s


def vc_to_r(vc):
    """Convert circular velocity to orbital radius using the reference μ."""
    return MU_REF / (vc ** 2)


def generate_data(num_samples=200_000):
    file_path = os.path.join(os.path.dirname(__file__), 'orbital_data.csv')

    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Feature columns — pure circular-velocity features, no body flag:
        #   log_vc1   — log(v_c1) — circular velocity at parking orbit
        #   log_vc2   — log(v_c2) — circular velocity at target orbit;
        #               0.0 for Phasing (spacecraft returns to r1)
        #   log_vcb   — log(v_cb) — circular velocity at intermediate apogee;
        #               0.0 for Hohmann and Phasing
        #   Phase_Angle — degrees (Phasing only; 0 for other maneuvers)
        #   Maneuver_Type — one-hot encoded at training time
        #
        # Target: Delta_V_ms (m/s) — log1p transform applied in train_model.py
        writer.writerow([
            "Maneuver_Type",
            "log_vc1", "log_vc2", "log_vcb", "Phase_Angle",
            "Delta_V_ms"
        ])

        generated = 0
        while generated < num_samples:
            m_type = random.choice(["Hohmann", "Bi-Elliptic", "Phasing"])
            delta_v = 0.0
            vc2, vcb, phase = 0.0, 0.0, 0.0

            try:
                # Sample circular velocity at the parking orbit uniformly in
                # [VC_MIN, VC_MAX].  Using log-uniform sampling gives equal
                # coverage per decade of velocity rather than biasing toward
                # the high-velocity (Earth LEO) region.
                log_vc1 = random.uniform(math.log(VC_MIN), math.log(VC_MAX))
                vc1 = math.exp(log_vc1)
                r1  = vc_to_r(vc1)

                if m_type == "Hohmann":
                    # 8% same-orbit cases (vc2 = vc1) teach the model the
                    # ΔV = 0 boundary.  Without these, the model never sees
                    # the zero case and extrapolates incorrectly for identical
                    # or nearly identical orbits.
                    if random.random() < 0.08:
                        vc2 = vc1
                    else:
                        log_vc2 = random.uniform(math.log(VC_MIN), math.log(VC_MAX))
                        vc2 = math.exp(log_vc2)
                    r2 = vc_to_r(vc2)
                    delta_v = hohmann_transfer(MU_REF, r1, r2)

                elif m_type == "Bi-Elliptic":
                    log_vc2 = random.uniform(math.log(VC_MIN), math.log(VC_MAX))
                    vc2 = math.exp(log_vc2)

                    # Bi-Elliptic orbit ratio constraint: max(r1,r2)/min(r1,r2) > 3.
                    # Since r = μ/v_c², ratio_r = (max_vc/min_vc)² → need
                    # max(vc1,vc2)/min(vc1,vc2) > sqrt(3) ≈ 1.732
                    vc_ratio = max(vc1, vc2) / min(vc1, vc2)
                    if vc_ratio < math.sqrt(3):
                        continue

                    # vcb must correspond to rb beyond BOTH r1 and r2 →
                    # v_cb < min(v_c1, v_c2).  We take 80% of the min vc
                    # as the upper bound so there's comfortable margin above rb.
                    vcb_max = min(vc1, vc2) * 0.80
                    if vcb_max <= VC_MIN:
                        continue
                    log_vcb = random.uniform(math.log(VC_MIN), math.log(vcb_max))
                    vcb = math.exp(log_vcb)

                    r2  = vc_to_r(vc2)
                    rb  = vc_to_r(vcb)
                    delta_v = bi_elliptic_transfer(MU_REF, r1, r2, rb)

                elif m_type == "Phasing":
                    phase   = random.uniform(1, 180)
                    # vc2 and vcb remain 0.0 — spacecraft returns to r1
                    delta_v = phasing_maneuver(MU_REF, r1, phase)

                if delta_v < 0:
                    continue

                # Log-transform velocities.  For unused slots (vc2=0 for Phasing,
                # vcb=0 for Hohmann/Phasing) we write 0.0 as sentinel — the
                # maneuver one-hot flags tell the model when to ignore them.
                log_vc2 = math.log(vc2) if vc2 > 0 else 0.0
                log_vcb = math.log(vcb) if vcb > 0 else 0.0

                writer.writerow([
                    m_type,
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
