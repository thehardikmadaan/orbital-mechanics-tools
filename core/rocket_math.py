import math

def calculate_inital_mass(delta_v, isp , final_mass):
    """ Calculates the required initial mass (wet mass) of a rocket.

    Inputs:
    delta_v: Total velocity change needed (m/s)
    isp: Specific impulse of the engine (seconds)
    final_mass: Mass of the payload + empty rocket stage (kg)

    Returns:
    initial_mass (kg) """