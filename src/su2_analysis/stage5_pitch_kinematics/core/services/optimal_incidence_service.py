"""Optimal incidence and off-design penalty for the single-actuator VPF."""
from __future__ import annotations
import numpy as np
import pandas as pd
from su2_analysis.config_loader import AnalysisConfig


def compute_optimal_incidence(pitch_map: pd.DataFrame) -> pd.DataFrame:
    """Pivot pitch_map to show α_opt for each condition/section combination."""
    return pitch_map[["condition", "section", "alpha_opt", "ld_opt"]].copy()


def compute_off_design(
    pitch_map: pd.DataFrame,
    design_condition: str = "cruise",
) -> pd.DataFrame:
    """Compute incidence mismatch and efficiency penalty at off-design conditions.

    For each section the hub pitch is set to the design-condition α_opt.
    At other conditions the actual flow angle differs → Δα mismatch.
    Efficiency penalty estimated as ΔCL/CD ≈ −0.05 × Δα² (quadratic approximation).
    """
    rows = []
    sections = pitch_map["section"].unique()
    conditions = pitch_map["condition"].unique()

    for section in sections:
        pm_sec = pitch_map[pitch_map["section"] == section]
        # Design-point alpha (cruise)
        design_row = pm_sec[pm_sec["condition"] == design_condition]
        alpha_design = float(design_row["alpha_opt"].iloc[0]) if not design_row.empty else 4.0
        ld_design    = float(design_row["ld_opt"].iloc[0])    if not design_row.empty else 30.0

        for cond in conditions:
            row = pm_sec[pm_sec["condition"] == cond]
            alpha_free = float(row["alpha_opt"].iloc[0]) if not row.empty else alpha_design
            ld_free    = float(row["ld_opt"].iloc[0])    if not row.empty else ld_design

            delta_alpha = alpha_free - alpha_design    # mismatch [°]
            # Quadratic ld penalty coefficient from typical fan polar curvature
            PENALTY_COEFF = 0.05
            delta_ld = -PENALTY_COEFF * delta_alpha**2

            rows.append({
                "section":         section,
                "condition":       cond,
                "alpha_design_deg": alpha_design,
                "alpha_free_deg":  alpha_free,
                "delta_alpha_deg": delta_alpha,
                "ld_free":         ld_free,
                "ld_constrained":  ld_design + delta_ld,
                "ld_penalty":      delta_ld,
            })

    return pd.DataFrame(rows)
