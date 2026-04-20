"""Velocity triangles and 3D kinematics for the fan stage."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.shared.atmosphere import blade_velocity, speed_of_sound, isa_temperature


def build_velocity_triangles(
    cfg: AnalysisConfig,
    pitch_map: pd.DataFrame,
) -> pd.DataFrame:
    """Compute inlet/outlet velocity triangles for each condition/section.

    Convention
    ----------
    Va : axial velocity [m/s]
    U  : blade speed   [m/s]
    W  : relative velocity = √(Va² + U²)  [m/s]
    β1 : relative inlet angle = arctan(Va/U)  [°]
    Wθ : swirl component of relative velocity = U - Vθ
    Vθ : absolute swirl = Va·tan(α_opt)  [m/s]
    """
    rpm = cfg.fan_geometry.rpm
    rows = []

    for cond_name, fc in cfg.flight_conditions.items():
        Va  = fc.axial_velocity
        alt = fc.altitude
        a   = speed_of_sound(alt)
        T   = isa_temperature(alt)

        for section_name, sec in cfg.fan_geometry.sections.items():
            U = blade_velocity(sec.radius, rpm)
            W = math.sqrt(Va**2 + U**2)
            M_rel = W / a

            pm_row = pitch_map[
                (pitch_map["condition"] == cond_name) &
                (pitch_map["section"]   == section_name)
            ]
            alpha = float(pm_row["alpha_opt"].iloc[0]) if not pm_row.empty else 4.0

            beta1    = math.degrees(math.atan2(Va, U))          # inlet relative angle
            Vtheta   = Va * math.tan(math.radians(alpha))        # absolute swirl
            Wtheta   = U - Vtheta                                # relative swirl
            beta2    = math.degrees(math.atan2(Va, Wtheta)) if Wtheta != 0 else 90.0
            delta_Vu = Vtheta                                    # swirl addition

            rows.append({
                "condition":   cond_name,
                "section":     section_name,
                "radius_m":    sec.radius,
                "Va_ms":       Va,
                "U_ms":        U,
                "W_ms":        W,
                "M_rel":       M_rel,
                "alpha_deg":   alpha,
                "beta1_deg":   beta1,
                "Vtheta_ms":   Vtheta,
                "Wtheta_ms":   Wtheta,
                "beta2_deg":   beta2,
                "delta_Vu_ms": delta_Vu,
            })

    return pd.DataFrame(rows)
