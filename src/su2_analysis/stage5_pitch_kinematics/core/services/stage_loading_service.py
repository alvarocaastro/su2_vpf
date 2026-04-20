"""Stage loading analysis: Euler work equation, φ-ψ diagram."""
from __future__ import annotations
import math
import pandas as pd
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.settings import PHI_MIN, PHI_MAX, PSI_MIN, PSI_MAX
from su2_analysis.shared.atmosphere import blade_velocity


def _flow_coefficient(Va: float, U: float) -> float:
    return Va / U


def _work_coefficient(Va: float, U: float, alpha_deg: float) -> float:
    """ψ = Va/U · tan(α) — simplified axial fan Euler work."""
    return (Va / U) * math.tan(math.radians(alpha_deg))


def build_stage_loading(
    cfg: AnalysisConfig,
    pitch_map: pd.DataFrame,
    scenario: str = "ideal",
) -> pd.DataFrame:
    """Compute φ, ψ, and specific work for each condition/section.

    Parameters
    ----------
    scenario : 'ideal'  → each section uses its own α_opt (free pitch)
               'real'   → all sections share the hub α_opt (single actuator)
    """
    rpm = cfg.fan_geometry.rpm
    rows = []

    # In the real scenario, the hub pitch sets the pitch angle for all sections
    hub_alpha: dict[str, float] = {}
    if scenario == "real":
        for cond in cfg.flight_conditions:
            pm_root = pitch_map[
                (pitch_map["condition"] == cond) &
                (pitch_map["section"] == "root")
            ]
            hub_alpha[cond] = float(pm_root["alpha_opt"].iloc[0]) if not pm_root.empty else 4.0

    for cond_name, fc in cfg.flight_conditions.items():
        Va = fc.axial_velocity
        for section_name, sec in cfg.fan_geometry.sections.items():
            U = blade_velocity(sec.radius, rpm)

            if scenario == "ideal":
                pm_row = pitch_map[
                    (pitch_map["condition"] == cond_name) &
                    (pitch_map["section"]   == section_name)
                ]
                alpha = float(pm_row["alpha_opt"].iloc[0]) if not pm_row.empty else 4.0
            else:
                alpha = hub_alpha.get(cond_name, 4.0)

            phi = _flow_coefficient(Va, U)
            psi = _work_coefficient(Va, U, alpha)
            W_spec = U * Va * math.tan(math.radians(alpha))    # Euler work [J/kg]

            in_design_zone = (
                PHI_MIN <= phi <= PHI_MAX and PSI_MIN <= psi <= PSI_MAX
            )

            rows.append({
                "scenario":      scenario,
                "condition":     cond_name,
                "section":       section_name,
                "radius_m":      sec.radius,
                "U_ms":          U,
                "Va_ms":         Va,
                "alpha_deg":     alpha,
                "phi":           phi,
                "psi":           psi,
                "W_spec_J_kg":   W_spec,
                "in_design_zone": in_design_zone,
            })

    return pd.DataFrame(rows)
