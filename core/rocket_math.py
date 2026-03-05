import math

def calculate_initial_mass(delta_v, isp , final_mass):
    """ Calculates the required initial mass (wet mass) of a rocket.

    Inputs:
    delta_v: Total velocity change needed (m/s)
    isp: Specific impulse of the engine (seconds)
    final_mass: Mass of the payload + empty rocket stage (kg)

    Returns:
    initial_mass (kg) """
    #$m_0 = m_f e^{\frac{\Delta v}{I_{sp} \cdot g_0}}$
    g0 = 9.81 # gravity- m/s_square
    initial_mass = (final_mass* (math.exp(delta_v/(isp*g0))))
    return initial_mass

# for how finding delta v based on fuel mass

def calculate_delta_v(initial_mass , final_mass, isp):
    """
        Calculates the maximum Delta-v a rocket stage can achieve.
        """
    g0 = 9.81
    delta_v = (isp * g0 * math.log(initial_mass/final_mass))
    return delta_v