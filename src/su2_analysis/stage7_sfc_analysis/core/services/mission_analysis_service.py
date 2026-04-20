"""Mission fuel burn integration over all flight phases."""
from __future__ import annotations
import pandas as pd
from su2_analysis.config_loader import EngineParameters


def compute_mission_fuel_burn(
    sfc_df: pd.DataFrame,
    engine: EngineParameters,
) -> pd.DataFrame:
    """Integrate fuel burn over the mission profile.

    Fuel burn = SFC × thrust_fraction × design_thrust × duration [hours]

    Returns
    -------
    DataFrame with per-phase fuel burn (baseline and VPF) and totals.
    """
    design_thrust_lbf = engine.design_thrust_kN * 1000 / 4.44822   # kN → lbf

    rows = []
    for phase_name, phase in engine.mission.items():
        duration_h = phase.duration_min / 60.0
        thrust_lbf = phase.thrust_fraction * design_thrust_lbf

        sfc_row = sfc_df[sfc_df["condition"] == phase_name]
        if sfc_row.empty:
            sfc_row = sfc_df.iloc[0:1]    # fallback to first available

        sfc_base = float(sfc_row["sfc_base"].iloc[0])
        sfc_vpf  = float(sfc_row["sfc_new"].iloc[0])

        fuel_base = sfc_base * thrust_lbf * duration_h    # lb
        fuel_vpf  = sfc_vpf  * thrust_lbf * duration_h
        saving_lb = fuel_base - fuel_vpf
        saving_kg = saving_lb * 0.453592

        rows.append({
            "phase":             phase_name,
            "duration_min":      phase.duration_min,
            "thrust_fraction":   phase.thrust_fraction,
            "thrust_lbf":        thrust_lbf,
            "sfc_base":          sfc_base,
            "sfc_vpf":           sfc_vpf,
            "fuel_base_lb":      fuel_base,
            "fuel_vpf_lb":       fuel_vpf,
            "fuel_saving_lb":    saving_lb,
            "fuel_saving_kg":    saving_kg,
        })

    df = pd.DataFrame(rows)

    # Totals row
    total = df.select_dtypes("number").sum()
    total_row = {col: float(total[col]) if col in total else "" for col in df.columns}
    total_row["phase"] = "TOTAL"
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    return df
