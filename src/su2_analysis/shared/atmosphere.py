"""ISA atmosphere model and Reynolds/Mach utilities."""
from __future__ import annotations
import math
from su2_analysis.settings import (
    GAMMA, R_AIR, T0_ISA, P0_ISA, LAPSE_RATE,
    MU_REF, T_MU_REF, S_SUTHERLAND,
)


def isa_temperature(altitude_m: float) -> float:
    """Return static temperature [K] at given altitude (troposphere only)."""
    return T0_ISA - LAPSE_RATE * altitude_m


def isa_pressure(altitude_m: float) -> float:
    """Return static pressure [Pa] at given altitude."""
    T = isa_temperature(altitude_m)
    return P0_ISA * (T / T0_ISA) ** (9.80665 / (LAPSE_RATE * R_AIR))


def isa_density(altitude_m: float) -> float:
    """Return air density [kg/m³] at given altitude."""
    return isa_pressure(altitude_m) / (R_AIR * isa_temperature(altitude_m))


def sutherland_viscosity(T: float) -> float:
    """Dynamic viscosity [Pa·s] via Sutherland's law."""
    return MU_REF * (T / T_MU_REF) ** 1.5 * (T_MU_REF + S_SUTHERLAND) / (T + S_SUTHERLAND)


def speed_of_sound(altitude_m: float) -> float:
    """Speed of sound [m/s] at given altitude."""
    return math.sqrt(GAMMA * R_AIR * isa_temperature(altitude_m))


def reynolds_number(velocity_ms: float, chord_m: float, altitude_m: float) -> float:
    """Chord-based Reynolds number."""
    rho = isa_density(altitude_m)
    mu  = sutherland_viscosity(isa_temperature(altitude_m))
    return rho * velocity_ms * chord_m / mu


def blade_velocity(radius_m: float, rpm: float) -> float:
    """Blade tangential velocity [m/s]."""
    return (2.0 * math.pi * rpm / 60.0) * radius_m


def relative_velocity(axial_ms: float, blade_ms: float) -> float:
    """Resultant relative velocity seen by the blade section [m/s]."""
    return math.sqrt(axial_ms**2 + blade_ms**2)


def wall_spacing_for_yplus(
    yplus: float,
    velocity_ms: float,
    chord_m: float,
    altitude_m: float,
) -> float:
    """Estimate first-cell wall distance [m] to achieve target y+.

    Uses the flat-plate estimate: u_τ ≈ 0.026 * U_inf / Re_c^(1/7).
    """
    rho = isa_density(altitude_m)
    mu  = sutherland_viscosity(isa_temperature(altitude_m))
    Re  = rho * velocity_ms * chord_m / mu
    cf  = 0.026 / Re ** (1.0 / 7.0)      # skin friction (Prandtl power law)
    tau_w = 0.5 * cf * rho * velocity_ms**2
    u_tau = math.sqrt(tau_w / rho)
    return yplus * mu / (rho * u_tau)
