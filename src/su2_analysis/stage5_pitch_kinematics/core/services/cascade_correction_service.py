"""Cascade corrections: Weinig solidity factor and Carter deviation rule."""
from __future__ import annotations
import math
import pandas as pd
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.settings import CARTER_M_COEFFICIENT


def weinig_factor(solidity: float) -> float:
    """Weinig lift-reduction factor for blade-to-blade cascade interference.

    K = arctan(π·σ/2) / (π·σ/2)
    Approaches 1 for isolated foil (σ→0) and decreases for denser cascades.
    Valid for σ ∈ [0.5, 2.5].  Reference: Dixon & Hall, 'Fluid Mechanics and
    Thermodynamics of Turbomachinery', 7th ed., eq. 3.46.
    """
    arg = math.pi * solidity / 2.0
    return math.atan(arg) / arg


def carter_deviation(camber_angle_deg: float, solidity: float) -> float:
    """Carter's rule for blade outlet angle deviation [degrees].

    δ = m · θ / √σ,  m = CARTER_M_COEFFICIENT = 0.23
    """
    return CARTER_M_COEFFICIENT * camber_angle_deg / math.sqrt(solidity)


def build_cascade_table(cfg: AnalysisConfig, camber_angle_deg: float = 20.0) -> pd.DataFrame:
    """Return cascade correction table for all blade sections.

    Parameters
    ----------
    camber_angle_deg : representative blade camber angle [°]
    """
    rows = []
    for name, sec in cfg.fan_geometry.sections.items():
        sigma = sec.solidity
        K_w   = weinig_factor(sigma)
        delta = carter_deviation(camber_angle_deg, sigma)
        rows.append({
            "section":           name,
            "radius_m":          sec.radius,
            "chord_m":           sec.chord,
            "solidity":          sigma,
            "weinig_factor":     K_w,
            "cl_scale":          K_w,
            "carter_deviation_deg": delta,
        })
    return pd.DataFrame(rows)
