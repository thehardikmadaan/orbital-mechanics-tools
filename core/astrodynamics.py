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