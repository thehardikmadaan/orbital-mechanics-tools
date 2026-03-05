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