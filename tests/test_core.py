"""Tests for the orbital mechanics core physics engine."""

import math
import pytest

from core.astrodynamics import (
    circular_velocity,
    orbital_period,
    hohmann_transfer,
    hohmann_transfer_time,
    bi_elliptic_transfer,
    bi_elliptic_transfer_time,
    phasing_maneuver,
    plane_change_dv,
    BODY_PARAMS,
)
from core.rocket_math import (
    calculate_initial_mass,
    calculate_delta_v,
    mass_fraction,
    payload_fraction,
    ENGINE_PRESETS,
)

# Standard gravitational parameter for Earth (m³/s²)
MU_EARTH = BODY_PARAMS["Earth"]["mu"]

# Earth radius in metres
R_EARTH = BODY_PARAMS["Earth"]["radius_km"] * 1000

# ISS orbit radius (~420 km altitude)
R_ISS = R_EARTH + 420e3

# GEO orbit radius
R_GEO = R_EARTH + 35_786e3


class TestCircularVelocity:
    def test_leo_velocity(self):
        """LEO circular velocity is approximately 7.66 km/s."""
        v = circular_velocity(MU_EARTH, R_ISS)
        assert 7600 < v < 7700

    def test_geo_velocity(self):
        """GEO circular velocity is approximately 3.07 km/s."""
        v = circular_velocity(MU_EARTH, R_GEO)
        assert 3060 < v < 3080

    def test_higher_orbit_slower(self):
        """Circular velocity decreases as orbital radius increases."""
        v1 = circular_velocity(MU_EARTH, R_ISS)
        v2 = circular_velocity(MU_EARTH, R_GEO)
        assert v1 > v2


class TestOrbitalPeriod:
    def test_geo_period_is_one_day(self):
        """GEO period is approximately 86 164 seconds (one sidereal day)."""
        T = orbital_period(MU_EARTH, R_GEO)
        assert abs(T - 86_164) < 60  # within 1 minute

    def test_period_increases_with_radius(self):
        """Higher orbits have longer periods (Kepler's Third Law)."""
        T1 = orbital_period(MU_EARTH, R_ISS)
        T2 = orbital_period(MU_EARTH, R_GEO)
        assert T2 > T1


class TestHohmannTransfer:
    def test_same_orbit_zero_dv(self):
        """Transferring to the same orbit requires zero delta-v."""
        dv = hohmann_transfer(MU_EARTH, R_ISS, R_ISS)
        assert dv == pytest.approx(0.0, abs=1e-6)

    def test_leo_to_geo_dv(self):
        """LEO→GEO Hohmann delta-v is approximately 3930 m/s."""
        dv = hohmann_transfer(MU_EARTH, R_ISS, R_GEO)
        assert 3800 < dv < 4100

    def test_transfer_time_positive(self):
        """Transfer time must be positive."""
        t = hohmann_transfer_time(MU_EARTH, R_ISS, R_GEO)
        assert t > 0

    def test_transfer_time_leo_to_geo(self):
        """LEO→GEO Hohmann transfer takes roughly 5 hours."""
        t = hohmann_transfer_time(MU_EARTH, R_ISS, R_GEO)
        assert 4 * 3600 < t < 6 * 3600

    def test_symmetric(self):
        """Delta-v is the same regardless of transfer direction."""
        dv_up = hohmann_transfer(MU_EARTH, R_ISS, R_GEO)
        dv_down = hohmann_transfer(MU_EARTH, R_GEO, R_ISS)
        assert dv_up == pytest.approx(dv_down, rel=1e-9)


class TestBiEllipticTransfer:
    def test_dv_positive(self):
        """Bi-Elliptic delta-v must be positive."""
        r_b = R_EARTH + 100_000e3
        dv = bi_elliptic_transfer(MU_EARTH, R_ISS, R_GEO, r_b)
        assert dv > 0

    def test_transfer_time_positive(self):
        """Bi-Elliptic transfer time must be positive."""
        r_b = R_EARTH + 100_000e3
        t = bi_elliptic_transfer_time(MU_EARTH, R_ISS, R_GEO, r_b)
        assert t > 0

    def test_large_ratio_cheaper_than_hohmann(self):
        """For very large orbit ratios, Bi-Elliptic beats Hohmann."""
        r1 = R_EARTH + 200e3
        r2 = R_EARTH + 500_000e3
        r_b = R_EARTH + 800_000e3
        dv_hohmann = hohmann_transfer(MU_EARTH, r1, r2)
        dv_bi = bi_elliptic_transfer(MU_EARTH, r1, r2, r_b)
        assert dv_bi < dv_hohmann


class TestPhasingManeuver:
    def test_dv_positive(self):
        """Phasing delta-v must be positive."""
        dv = phasing_maneuver(MU_EARTH, R_ISS, 45)
        assert dv > 0

    def test_larger_angle_more_dv(self):
        """A larger phase angle requires more delta-v."""
        dv_small = phasing_maneuver(MU_EARTH, R_ISS, 10)
        dv_large = phasing_maneuver(MU_EARTH, R_ISS, 90)
        assert dv_large > dv_small


class TestPlaneChange:
    def test_zero_angle_zero_dv(self):
        """A zero-degree inclination change costs nothing."""
        dv = plane_change_dv(7660, 0)
        assert dv == pytest.approx(0.0, abs=1e-9)

    def test_positive_angle_positive_dv(self):
        """A positive inclination change has a positive cost."""
        dv = plane_change_dv(7660, 28.5)
        assert dv > 0


class TestRocketMath:
    def test_round_trip_delta_v(self):
        """calculate_delta_v is the inverse of calculate_initial_mass."""
        isp = 450
        final_mass = 1000
        delta_v_target = 3000
        wet_mass = calculate_initial_mass(delta_v_target, isp, final_mass)
        delta_v_recovered = calculate_delta_v(wet_mass, final_mass, isp)
        assert delta_v_recovered == pytest.approx(delta_v_target, rel=1e-9)

    def test_wet_mass_greater_than_dry_mass(self):
        """Wet mass must exceed dry mass for any positive delta-v."""
        wet = calculate_initial_mass(1000, 310, 500)
        assert wet > 500

    def test_mass_fraction_sum(self):
        """Mass fraction + payload fraction should equal 100% for full payload."""
        propellant = 600
        wet = 1000
        payload = wet - propellant
        mf = mass_fraction(propellant, wet)
        pf = payload_fraction(payload, wet)
        assert mf + pf == pytest.approx(100.0, abs=1e-9)

    def test_mass_fraction_zero_wet_mass(self):
        """mass_fraction returns 0 for zero wet mass (guard against division)."""
        assert mass_fraction(0, 0) == 0.0

    def test_payload_fraction_zero_wet_mass(self):
        """payload_fraction returns 0 for zero wet mass."""
        assert payload_fraction(0, 0) == 0.0

    def test_engine_presets_have_isp(self):
        """Every engine preset must define a positive Isp."""
        for name, params in ENGINE_PRESETS.items():
            assert params["isp"] > 0, f"{name} has non-positive Isp"


class TestBodyParams:
    def test_all_bodies_present(self):
        """Earth, Moon, and Mars must all be defined."""
        for body in ("Earth", "Moon", "Mars"):
            assert body in BODY_PARAMS

    def test_mu_positive(self):
        """Gravitational parameter must be positive for all bodies."""
        for body, params in BODY_PARAMS.items():
            assert params["mu"] > 0, f"{body} has non-positive mu"

    def test_radius_positive(self):
        """Body radius must be positive."""
        for body, params in BODY_PARAMS.items():
            assert params["radius_km"] > 0, f"{body} has non-positive radius"
