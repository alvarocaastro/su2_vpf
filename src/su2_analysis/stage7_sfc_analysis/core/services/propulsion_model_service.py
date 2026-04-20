"""Propulsion model: 2D-to-3D efficiency transfer and fan efficiency gain."""
from __future__ import annotations
import numpy as np
import pandas as pd
from su2_analysis.config_loader import EngineParameters
from su2_analysis.settings import TAU_TRANSFER


def compute_epsilon(
    metrics: pd.DataFrame,
    reference_condition: str = "cruise",
    reference_section: str = "mid",
) -> pd.DataFrame:
    """Compute ε = (CL/CD)_vpf / (CL/CD)_fixed_ref for each section/condition.

    The fixed-pitch reference is the cruise/mid condition.
    """
    ref = metrics[
        (metrics["condition"] == reference_condition) &
        (metrics["section"]   == reference_section)
    ]
    ld_ref = float(ref["ld_max"].iloc[0]) if not ref.empty else 1.0

    df = metrics.copy()
    df["ld_ref"]   = ld_ref
    df["epsilon"]  = df["ld_max"] / ld_ref
    return df


def compute_delta_eta(
    epsilon_df: pd.DataFrame,
    engine: EngineParameters,
) -> pd.DataFrame:
    """Compute fan efficiency gain Δη_fan and updated SFC for each condition.

    Δη_fan = τ · (ε̄ − 1) · η_fan,base
    SFC_new = SFC_base / (1 + Δη_fan / η_fan,base)
    """
    tau      = engine.tau
    eta_base = engine.fan_efficiency
    sfc_base = engine.baseline_sfc

    rows = []
    for cond in epsilon_df["condition"].unique():
        sub = epsilon_df[epsilon_df["condition"] == cond]
        eps_mean = float(sub["epsilon"].mean())

        delta_eta = tau * (eps_mean - 1.0) * eta_base
        eta_new   = eta_base + delta_eta
        sfc_new   = sfc_base / (1.0 + delta_eta / eta_base)
        delta_sfc = (1.0 - sfc_new / sfc_base) * 100.0

        rows.append({
            "condition":    cond,
            "epsilon_mean": eps_mean,
            "delta_eta":    delta_eta,
            "eta_new":      eta_new,
            "sfc_base":     sfc_base,
            "sfc_new":      sfc_new,
            "delta_sfc_pct": delta_sfc,
            "tau":          tau,
        })

    return pd.DataFrame(rows)
