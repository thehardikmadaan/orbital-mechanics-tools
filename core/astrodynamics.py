import math

def circular_velocity(mu, radius):
    """
        Calculates the velocity of a spacecraft in a circular orbit.

        Inputs:
        mu: Standard gravitational parameter of the central body (m^3/s^2)
        radius: Distance from the center of the body to the orbit (meters)

        Returns:
        velocity (m/s)
        """
    v= math.sqrt(mu/radius)
    return v

def hohmann_transfer(mu,r1,r2):
    """
        Calculates the total Delta-v required for a Hohmann transfer.

        Inputs:
        mu: Standard gravitational parameter (m^3/s^2)
        r1: Radius of the initial circular orbit (meters)
        r2: Radius of the final circular orbit (meters)

        Returns:
        total_delta_v (m/s)
        """
    #1/
    a = (r1+r2)/2
    #2/
    v_perigee = math.sqrt(mu * ((2 / r1) - (1 / a)))
    #3/
    v_apogee = math.sqrt(mu * ((2 / r2) - (1 / a)))
    #4/
    v_initial = circular_velocity(mu,r1)
    v_final = circular_velocity(mu,r2)
    #5/
    delta_v1 = abs(v_perigee - v_initial)
    delta_v2 = abs(v_apogee- v_final)
    #6/
    total_delta_v = delta_v1 + delta_v2
    return total_delta_v


def bi_elliptic_transfer(mu, r1, r2, rb):
    """
    Calculates the total Delta-v required for a Bi-Elliptic transfer.

    Inputs:
    mu: Standard gravitational parameter (m^3/s^2)
    r1: Initial circular orbit radius (meters)
    r2: Final circular orbit radius (meters)
    rb: Intermediate deep-space apogee radius (meters) - MUST be larger than r1 and r2!

    Returns:
    total_delta_v (m/s)
    """
    # Burn 1: Circular r1 to Ellipse 1 (perigee r1, apogee rb)
    a1 = (r1 + rb) / 2
    v_c1 = math.sqrt(mu / r1)
    v_p1 = math.sqrt(mu * ((2 / r1) - (1 / a1)))
    delta_v1 = abs(v_p1 - v_c1)

    # Burn 2: Ellipse 1 to Ellipse 2 (Out in deep space at rb)
    a2 = (r2 + rb) / 2
    v_a1 = math.sqrt(mu * ((2 / rb) - (1 / a1)))
    v_a2 = math.sqrt(mu * ((2 / rb) - (1 / a2)))
    delta_v2 = abs(v_a2 - v_a1)

    # Burn 3: Ellipse 2 to Circular r2 (Arriving at target r2)
    v_p2 = math.sqrt(mu * ((2 / r2) - (1 / a2)))
    v_c2 = math.sqrt(mu / r2)
    delta_v3 = abs(v_c2 - v_p2)

    total_delta_v = delta_v1 + delta_v2 + delta_v3
    return total_delta_v


ef
phasing_maneuver(mu, r1, phase_angle_deg):
"""
Calculates the Delta-v for a coplanar phasing maneuver.
Assumes a single phasing orbit to catch up to a target in the same parking orbit (r1).

Inputs:
mu: Standard gravitational parameter (m^3/s^2)
r1: Radius of the circular parking orbit (meters)
phase_angle_deg: How far ahead the target is (degrees)
"""
# 1. Original circular period (time it takes to do 1 orbit)
T_initial = 2 * math.pi * math.sqrt((r1 ** 3) / mu)

# 2. Time required to catch up the phase angle
time_shift = (phase_angle_deg / 360.0) * T_initial

# 3. New orbital period required to intercept the target
# If target is ahead (+ angle), we must do a FASTER (smaller) orbit: T - time_shift
T_phasing = T_initial - time_shift

# 4. Calculate the Semi-major axis of this new phasing ellipse
a_phasing = (mu * (T_phasing / (2 * math.pi)) ** 2) ** (1 / 3)

# 5. Velocity calculations
v_circ = math.sqrt(mu / r1)
# Velocity at perigee/apogee of phasing orbit (where it intersects r1)
v_phase = math.sqrt(mu * ((2 / r1) - (1 / a_phasing)))

# Burn 1: Enter phasing orbit
# Burn 2: Exit phasing orbit (re-circularize)
delta_v_burn = abs(v_phase - v_circ)
total_delta_v = 2 * delta_v_burn

return total_delta_v