"""Blade twist design for cruise condition (single-actuator constraint)."""
from __future__ import annotations
import math
import pandas as pd
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.shared.atmosphere import blade_velocity


def compute_blade_twist(
    cfg: AnalysisConfig,
    pitch_map: pd.DataFrame,
    design_condition: str = "cruise",
) -> pd.DataFrame:
    """Compute the blade metal angle β at each section for the design condition.

    β = arctan(Va / U)  (flow angle) + α_opt  (incidence)

    With a single actuator the hub pitch Δβ is applied uniformly.
    The twist table shows what off-design incidence results at other conditions.

    Returns
    -------
    DataFrame with columns: section, radius, U, beta_flow_deg, alpha_opt_design,
    beta_metal_deg, twist_total_deg (= β_root - β_tip).
    """
    rpm = cfg.fan_geometry.rpm
    rows = []

    for section_name, sec in cfg.fan_geometry.sections.items():
        U = blade_velocity(sec.radius, rpm)

        # Design condition (cruise) flow angle
        design_fc = cfg.flight_conditions.get(design_condition)
        Va = design_fc.axial_velocity if design_fc else 150.0
        beta_flow_design = math.degrees(math.atan2(Va, U))

        pm_row = pitch_map[
            (pitch_map["condition"] == design_condition) &
            (pitch_map["section"]   == section_name)
        ]
        alpha_opt = float(pm_row["alpha_opt"].iloc[0]) if not pm_row.empty else 4.0

        beta_metal = beta_flow_design + alpha_opt
        rows.append({
            "section":           section_name,
            "radius_m":          sec.radius,
            "U_ms":              U,
            "Va_design_ms":      Va,
            "beta_flow_deg":     beta_flow_design,
            "alpha_opt_design":  alpha_opt,
            "beta_metal_deg":    beta_metal,
        })

    df = pd.DataFrame(rows)
    if len(df) >= 2:
        beta_root = df.loc[df["section"] == "root", "beta_metal_deg"].values[0]
        beta_tip  = df.loc[df["section"] == "tip",  "beta_metal_deg"].values[0]
        df["twist_total_deg"] = beta_root - beta_tip
    else:
        df["twist_total_deg"] = float("nan")

    return df
