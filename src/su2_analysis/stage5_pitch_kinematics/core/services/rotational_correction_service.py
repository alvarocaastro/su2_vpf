"""3D rotational corrections: Snel and Du-Selig models."""
from __future__ import annotations
import pandas as pd
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.settings import SNEL_COEFFICIENT


def snel_cl_increment(cl_2d: float, chord: float, radius: float) -> float:
    """Snel (1994) centrifugal + Coriolis CL augmentation.

    ΔCL_rot = a · (c/r)² · CL_2D,   a = SNEL_COEFFICIENT = 3.0
    """
    return SNEL_COEFFICIENT * (chord / radius) ** 2 * cl_2d


def build_rotational_table(
    cfg: AnalysisConfig,
    pitch_map: pd.DataFrame,
) -> pd.DataFrame:
    """Apply Snel rotation correction at alpha_opt for each section/condition."""
    rows = []
    for _, pm_row in pitch_map.iterrows():
        cond    = pm_row["condition"]
        section = pm_row["section"]
        cl_2d   = pm_row.get("cl_opt", 0.8)
        if pd.isna(cl_2d):
            cl_2d = 0.8

        sec   = cfg.fan_geometry.sections[section]
        c     = sec.chord
        r     = sec.radius
        cr2   = (c / r) ** 2

        delta_cl = snel_cl_increment(cl_2d, c, r)
        cl_3d    = cl_2d + delta_cl
        rot_pct  = (delta_cl / cl_2d * 100.0) if cl_2d != 0 else 0.0

        rows.append({
            "condition":     cond,
            "section":       section,
            "radius_m":      r,
            "chord_m":       c,
            "c_over_r_sq":   cr2,
            "cl_2d":         cl_2d,
            "delta_cl_snel": delta_cl,
            "cl_3d":         cl_3d,
            "rotation_pct":  rot_pct,
        })
    return pd.DataFrame(rows)
