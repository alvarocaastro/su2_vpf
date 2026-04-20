"""Pitch adjustment: required Δβ per flight phase for the single actuator."""
from __future__ import annotations
import math
import pandas as pd
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.shared.atmosphere import blade_velocity


def compute_pitch_adjustment(
    cfg: AnalysisConfig,
    pitch_map: pd.DataFrame,
    design_condition: str = "cruise",
    reference_section: str = "mid",
) -> pd.DataFrame:
    """Compute the blade pitch angle change Δβ required at each flight phase.

    The actuator moves the blade metal angle β by Δβ from the cruise position.
    Using the mid-span section as representative (single actuator moves all sections together).
    """
    rpm = cfg.fan_geometry.rpm
    sec = cfg.fan_geometry.sections[reference_section]
    U   = blade_velocity(sec.radius, rpm)

    # Cruise reference metal angle
    pm_cruise = pitch_map[
        (pitch_map["condition"] == design_condition) &
        (pitch_map["section"]   == reference_section)
    ]
    alpha_cruise = float(pm_cruise["alpha_opt"].iloc[0]) if not pm_cruise.empty else 4.0
    Va_cruise    = cfg.flight_conditions[design_condition].axial_velocity
    beta_flow_cruise = math.degrees(math.atan2(Va_cruise, U))
    beta_metal_cruise = beta_flow_cruise + alpha_cruise

    rows = []
    for cond_name, fc in cfg.flight_conditions.items():
        Va = fc.axial_velocity
        pm_row = pitch_map[
            (pitch_map["condition"] == cond_name) &
            (pitch_map["section"]   == reference_section)
        ]
        alpha_opt = float(pm_row["alpha_opt"].iloc[0]) if not pm_row.empty else alpha_cruise

        beta_flow = math.degrees(math.atan2(Va, U))
        beta_metal_required = beta_flow + alpha_opt
        delta_beta = beta_metal_required - beta_metal_cruise

        rows.append({
            "condition":           cond_name,
            "section":             reference_section,
            "Va_ms":               Va,
            "U_ms":                U,
            "beta_flow_deg":       beta_flow,
            "alpha_opt_deg":       alpha_opt,
            "beta_metal_req_deg":  beta_metal_required,
            "beta_metal_cruise_deg": beta_metal_cruise,
            "delta_beta_deg":      delta_beta,
        })

    return pd.DataFrame(rows)
