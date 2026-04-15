from core.rocket_math import calculate_delta_v, calculate_initial_mass
from core.astrodynamics import circular_velocity,hohmann_transfer


def main():
    print(" XXX Orbital Transfer Calculations XXX")

    # Constraints
    mu = 3.986e14
    r_earth = 6371e3
    g0 = 9.81

    # Mission Parameters LEO TO GEO in meters
    r1 = 300000 + r_earth
    r2 = 35786000 + r_earth

    isp = 300 # Specific impulse
    final_mass = 2000 # Satellite + empty rocket stage (kg)

    # Perform Calculations
    delta_v = hohmann_transfer(mu, r1, r2)
    wet_mass = calculate_initial_mass(delta_v, isp, final_mass)
    propellant_mass = wet_mass - final_mass

    #results
    print("\n MISSION RESULTS :")
    print(f"Target Delta-v : {delta_v} m/s")
    print(f"Final Payload Mass : {final_mass} Kg")
    print(f"Required Wet Mass: {wet_mass:.2f} kg")
    print(f"Propellant Needed: {propellant_mass:.2f} kg")


#


if __name__ == "__main__":
    main()
    print("Mission Complete")

