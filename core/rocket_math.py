# core/rocket_math.py
# The Tsiolkovsky Rocket Equation — the fundamental constraint on all space travel.
# Published by Konstantin Tsiolkovsky in 1903. Without this equation, we couldn't
# plan any mission. It tells us how much fuel we need (or what Δv we can achieve).

import math

# ─────────────────────────────────────────────────────────────────────────────
# ENGINE / PROPELLANT PRESETS
# Isp (Specific Impulse) measures engine efficiency in seconds. Higher Isp =
# more Δv per kg of propellant. Electric drives are incredibly efficient but
# provide very low thrust; chemical rockets are less efficient but powerful.
#
# References:
#   NASA SP-8120 (Rocket Propulsion), ESA Propulsion Technology Review,
#   SpaceX/ULA engine performance datasheets.
# ─────────────────────────────────────────────────────────────────────────────
ENGINE_PRESETS = {
    "Chemical Bipropellant (310s)": {
        "isp": 310,
        "description": "Hydrazine/NTO — workhorse of satellite orbit raising. "
                       "Used on: ATV, Intelsat, most GEO satellites."
    },
    "H₂/LOX Cryogenic (450s)": {
        "isp": 450,
        "description": "Liquid hydrogen + liquid oxygen — highest performance "
                       "chemical engine. Used on: Ariane 5 upper stage (HM7B), "
                       "NASA RL-10 (Centaur), SpaceX Raptor Vacuum."
    },
    "Solid Rocket Motor (260s)": {
        "isp": 260,
        "description": "Solid propellant — simple, storable, no moving parts. "
                       "Used on: payload kick stages (PAM-D), sounding rockets, "
                       "ESA Vega lower stages."
    },
    "Monopropellant Hydrazine (220s)": {
        "isp": 220,
        "description": "Single-propellant system — lower Isp but simple and "
                       "reliable. Used for attitude control and small Δv burns "
                       "on most satellites."
    },
    "Electric Ion Drive (3000s)": {
        "isp": 3000,
        "description": "Xenon ion propulsion — extraordinary efficiency but "
                       "very low thrust (millinewtons). Used on: Dawn, Hayabusa, "
                       "ESA BepiColombo, Starlink satellites."
    },
    "Hall-Effect Thruster (1600s)": {
        "isp": 1600,
        "description": "Between ion and chemical — moderate thrust and efficiency. "
                       "Used on: SMART-1 (ESA lunar mission), OneWeb, Boeing 702SP."
    },
}

# Standard sea-level gravity — constant used in the rocket equation
# NIST value: 9.80665 m/s² (but 9.81 is the accepted engineering standard)
G0 = 9.81  # m/s²


def calculate_initial_mass(delta_v, isp, final_mass):
    """
    Required wet mass (fuel + spacecraft) to achieve a given Δv.

    This is the Tsiolkovsky Rocket Equation solved for the initial mass.
    The 'exponential tyranny': doubling the Δv doesn't double the fuel —
    it squares the mass ratio, which grows exponentially. This is why
    reaching orbit is so hard and why staging exists.

    Formula: m₀ = m_f × e^(Δv / (Isp × g₀))

    Inputs:
      delta_v    — Required velocity change (m/s)
      isp        — Specific impulse of the engine (seconds)
      final_mass — Payload + dry structure mass after the burn (kg)

    Returns:
      initial_mass (kg) — The wet mass needed at the start of the burn
    """
    return final_mass * math.exp(delta_v / (isp * G0))


def calculate_delta_v(initial_mass, final_mass, isp):
    """
    Maximum Δv achievable given the propellant loaded and dry mass.

    Inverse of calculate_initial_mass — answers 'how far can we go with this
    much fuel?' Used during mission feasibility checks.

    Formula: Δv = Isp × g₀ × ln(m₀ / m_f)

    Inputs:
      initial_mass — Wet mass at the start of the burn (kg)
      final_mass   — Dry mass at the end of the burn (kg)
      isp          — Specific impulse of the engine (seconds)

    Returns:
      delta_v (m/s)
    """
    return isp * G0 * math.log(initial_mass / final_mass)


def mass_fraction(propellant_mass, wet_mass):
    """
    Propellant mass fraction — how much of the spacecraft is fuel.

    A higher fraction means more of the vehicle is fuel rather than payload.
    Typical values: 70–90% for upper stages, 30–60% for satellite apogee engines.
    The Saturn V first stage was ~94% propellant by mass at liftoff.

    Returns: fraction as a percentage (0–100)
    """
    if wet_mass <= 0:
        return 0.0
    return (propellant_mass / wet_mass) * 100.0


def payload_fraction(payload_mass, wet_mass):
    """
    Payload mass fraction — the useful mass as a fraction of total wet mass.

    This is the inverse concern of mass_fraction. A rocket's job is to maximise
    this number. ESA's Ariane 6 targets ~2–4% payload fraction to GTO.

    Returns: fraction as a percentage (0–100)
    """
    if wet_mass <= 0:
        return 0.0
    return (payload_mass / wet_mass) * 100.0