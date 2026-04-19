# core/astrodynamics.py
# Orbital mechanics engine — all the physics that drives the mission planner.
# References: Bate, Mueller & White "Fundamentals of Astrodynamics" (NASA standard),
# ESA's GMAT User Guide, and NASA JPL Planetary Fact Sheet.

import math

# ─────────────────────────────────────────────────────────────────────────────
# PLANETARY CONSTANTS
# These are the standard gravitational parameters (μ = G × M) and mean radii
# used by ESA/NASA mission planners. μ is more precise than using G × M
# separately because it can be measured directly via spacecraft tracking.
# ─────────────────────────────────────────────────────────────────────────────
BODY_PARAMS = {
    "Earth": {
        "mu": 3.986004418e14,  # m³/s² — NASA/IERS 2010 standard value
        "radius_km": 6371.0,   # km    — Mean radius (not equatorial)
        "min_alt_km": 200,     # km    — Minimum safe orbit (above atmosphere)
        "max_alt_km": 400000,  # km    — Practical planning limit (~lunar dist)
    },
    "Moon": {
        "mu": 4.9048695e12,    # m³/s² — LRO/GRAIL mission measurements
        "radius_km": 1737.4,   # km    — Mean radius
        "min_alt_km": 20,      # km    — LRO flew at 50 km; 20 km is low
        "max_alt_km": 70000,   # km    — ~Hill sphere radius for planning
    },
    "Mars": {
        "mu": 4.282837e13,     # m³/s² — MRO navigation data
        "radius_km": 3389.5,   # km    — Mean radius (Mars Reconnaissance Orbiter)
        "min_alt_km": 150,     # km    — Just above the thin Martian atmosphere
        "max_alt_km": 400000,  # km    — Practical limit
    },
}


def circular_velocity(mu, radius):
    """
    Velocity of a spacecraft in a perfectly circular orbit.

    At this speed, gravity provides exactly the centripetal force needed to
    maintain the orbit — no thrust required. This is the reference velocity
    for all maneuver calculations.

    Formula: v = √(μ / r)   [derived from centripetal force = gravitational force]

    Inputs:
      mu     — Standard gravitational parameter of the central body (m³/s²)
      radius — Distance from the body's centre to the orbit (meters)

    Returns:
      Orbital velocity (m/s)
    """
    return math.sqrt(mu / radius)


def orbital_period(mu, semi_major_axis):
    """
    Orbital period of any Keplerian orbit using Kepler's Third Law.

    Kepler's Third Law says T² ∝ a³. The constant of proportionality depends
    on which body you're orbiting. ESA and NASA use this constantly — it's why
    GEO satellites take exactly 24 hours and GPS satellites take 12 hours.

    Formula: T = 2π × √(a³ / μ)

    Inputs:
      mu              — Standard gravitational parameter (m³/s²)
      semi_major_axis — Semi-major axis of the orbit (meters). For a circular
                        orbit this is just the orbital radius.

    Returns:
      Orbital period in seconds
    """
    return 2 * math.pi * math.sqrt((semi_major_axis ** 3) / mu)


def hohmann_transfer(mu, r1, r2):
    """
    Total Delta-v for a Hohmann transfer — the textbook two-burn maneuver.

    Invented by Walter Hohmann in 1925, this is the most fuel-efficient way
    to move between two coplanar circular orbits. It works by firing prograde
    at periapsis (Burn 1) to raise the apoapsis, then firing prograde again
    at apoapsis (Burn 2) to circularise at the target orbit.

    Used by: GPS satellite insertions, Hubble servicing missions (STS),
    ESA's BepiColombo Mercury transfer (combined with gravity assists).

    Step-by-step:
      1. Compute the semi-major axis of the transfer ellipse: a = (r1 + r2) / 2
      2. Find velocity at the ellipse's perigee (at r1) using the vis-viva equation
      3. Find velocity at the ellipse's apogee  (at r2) using the vis-viva equation
      4. Δv₁ = perigee velocity − circular velocity at r1
      5. Δv₂ = circular velocity at r2 − apogee velocity
      6. Total Δv = Δv₁ + Δv₂

    Inputs:
      mu — Standard gravitational parameter (m³/s²)
      r1 — Initial circular orbit radius (meters)
      r2 — Final circular orbit radius (meters)

    Returns:
      total_delta_v (m/s)
    """
    # Semi-major axis of the transfer ellipse (halfway between the two orbits)
    a = (r1 + r2) / 2

    # Vis-viva equation: v² = μ × (2/r − 1/a)
    # Gives the speed at any point on an elliptical orbit if you know r and a
    v_perigee = math.sqrt(mu * ((2 / r1) - (1 / a)))   # Speed at burn 1 point
    v_apogee  = math.sqrt(mu * ((2 / r2) - (1 / a)))   # Speed at burn 2 point

    # Circular orbit speeds before and after the transfer
    v_initial = circular_velocity(mu, r1)
    v_final   = circular_velocity(mu, r2)

    # Each burn accelerates the spacecraft from circular to elliptical (or vice versa)
    delta_v1 = abs(v_perigee - v_initial)
    delta_v2 = abs(v_apogee  - v_final)

    return delta_v1 + delta_v2


def hohmann_transfer_time(mu, r1, r2):
    """
    Transit time for a Hohmann transfer (half the ellipse period).

    The spacecraft travels exactly one half of the transfer ellipse, so the
    flight time is exactly T/2 of that ellipse. ESA reports this as 'Transfer
    Duration' in mission design documents.

    Inputs:
      mu — Standard gravitational parameter (m³/s²)
      r1 — Initial orbit radius (meters)
      r2 — Final orbit radius (meters)

    Returns:
      Transfer time in seconds
    """
    a = (r1 + r2) / 2
    # Full period of transfer ellipse ÷ 2 (we only fly the upper semicircle)
    return orbital_period(mu, a) / 2


def bi_elliptic_transfer(mu, r1, r2, rb):
    """
    Total Delta-v for a Bi-Elliptic transfer — three burns, two ellipses.

    Proposed independently by Shternfeld (1959) and shown to beat Hohmann when
    the target orbit is more than ~11.94× the initial orbit radius. NASA used
    this concept for deep-space staging; ESA considered it for GEO graveyard
    orbit disposals.

    How it works:
      Burn 1 at r1 → fly out to a deep-space apogee at rb
      Burn 2 at rb → redirect into a descent ellipse aimed at r2
      Burn 3 at r2 → circularise at the target orbit

    Inputs:
      mu — Standard gravitational parameter (m³/s²)
      r1 — Initial circular orbit radius (meters)
      r2 — Final circular orbit radius (meters)
      rb — Intermediate deep-space apogee radius (meters); MUST be > r1 and r2

    Returns:
      total_delta_v (m/s)
    """
    # ── Burn 1: Depart from r1 onto Ellipse 1 ────────────────────────────────
    a1   = (r1 + rb) / 2
    v_c1 = math.sqrt(mu / r1)
    v_p1 = math.sqrt(mu * ((2 / r1) - (1 / a1)))   # Perigee speed of Ellipse 1
    delta_v1 = abs(v_p1 - v_c1)

    # ── Burn 2: Transition at deep-space apogee rb ────────────────────────────
    a2   = (r2 + rb) / 2
    v_a1 = math.sqrt(mu * ((2 / rb) - (1 / a1)))   # Apogee speed of Ellipse 1
    v_a2 = math.sqrt(mu * ((2 / rb) - (1 / a2)))   # Apogee speed of Ellipse 2
    delta_v2 = abs(v_a2 - v_a1)

    # ── Burn 3: Arrive at r2 and circularise ─────────────────────────────────
    v_p2 = math.sqrt(mu * ((2 / r2) - (1 / a2)))   # Perigee speed of Ellipse 2
    v_c2 = math.sqrt(mu / r2)
    delta_v3 = abs(v_c2 - v_p2)

    return delta_v1 + delta_v2 + delta_v3


def bi_elliptic_transfer_time(mu, r1, r2, rb):
    """
    Total transit time for a Bi-Elliptic transfer (two half-ellipse periods).

    Inputs:
      mu — Standard gravitational parameter (m³/s²)
      r1 — Initial orbit radius (meters)
      r2 — Final orbit radius (meters)
      rb — Intermediate deep-space apogee (meters)

    Returns:
      Total transfer time in seconds
    """
    a1 = (r1 + rb) / 2   # First transfer ellipse semi-major axis
    a2 = (r2 + rb) / 2   # Second transfer ellipse semi-major axis
    return (orbital_period(mu, a1) / 2) + (orbital_period(mu, a2) / 2)


def phasing_maneuver(mu, r1, phase_angle_deg):
    """
    Delta-v for a coplanar phasing maneuver (catch-up / lead-ahead).

    Used constantly in rendezvous operations: ISS crew vehicle arrivals,
    Hubble repair missions, and ESA's ATV supply runs. If the target is
    ahead of you, you burn prograde into a smaller, faster orbit to lap the
    difference. If behind, you burn retrograde into a larger, slower orbit.

    This implementation handles the 'chase' case (target is ahead).

    Inputs:
      mu             — Standard gravitational parameter (m³/s²)
      r1             — Radius of the circular parking orbit (meters)
      phase_angle_deg — How far ahead the target is (0–180 degrees)

    Returns:
      total_delta_v (m/s) — Two equal burns: entry + exit
    """
    # Period of the original circular orbit (Kepler's 3rd Law)
    T_initial = orbital_period(mu, r1)

    # Time needed to close the phase gap (fraction of one full orbit)
    time_shift = (phase_angle_deg / 360.0) * T_initial

    # Target is ahead → we need a shorter period to catch it in one lap
    T_phasing = T_initial - time_shift

    # Compute the semi-major axis that produces this new period (inverse Kepler)
    a_phasing = (mu * (T_phasing / (2 * math.pi)) ** 2) ** (1 / 3)

    # Speed in the original circular orbit vs speed at r1 on the phasing ellipse
    v_circ  = math.sqrt(mu / r1)
    v_phase = math.sqrt(mu * ((2 / r1) - (1 / a_phasing)))

    # Same magnitude burn twice: enter phasing orbit, then exit back to r1
    delta_v_burn = abs(v_phase - v_circ)
    return 2 * delta_v_burn


def plane_change_dv(velocity, angle_deg):
    """
    Delta-v cost to change the orbital inclination (plane change maneuver).

    A pure plane change is extremely expensive — this is why ESA and NASA
    choose launch sites carefully to minimise the inclination penalty.
    For example, launching from Kourou (5°N) to GEO is much cheaper than
    from Baikonur (46°N). Formula from Fundamentals of Astrodynamics, §6.3.

    Formula: Δv = 2 × v × sin(Δi / 2)

    Inputs:
      velocity  — Current orbital speed at the point of the maneuver (m/s)
      angle_deg — Desired inclination change in degrees

    Returns:
      Delta-v required (m/s)
    """
    angle_rad = math.radians(angle_deg)
    return 2 * velocity * math.sin(angle_rad / 2)