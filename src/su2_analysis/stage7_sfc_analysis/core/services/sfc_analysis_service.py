"""SFC sensitivity analysis: sweep τ ∈ [0.3, 0.7]."""
from __future__ import annotations
import numpy as np
import pandas as pd
from su2_analysis.config_loader import EngineParameters


def compute_sfc_sensitivity(
    epsilon_df: pd.DataFrame,
    engine: EngineParameters,
    tau_range: tuple[float, float] = (0.3, 0.7),
    n_points: int = 9,
) -> pd.DataFrame:
    """Vary τ and compute resulting ΔSFC for each condition.

    Returns DataFrame with columns: tau, condition, delta_sfc_pct.
    """
    eta_base = engine.fan_efficiency
    sfc_base = engine.baseline_sfc
    taus     = np.linspace(*tau_range, n_points)

    rows = []
    for tau in taus:
        for cond in epsilon_df["condition"].unique():
            sub = epsilon_df[epsilon_df["condition"] == cond]
            eps_mean  = float(sub["epsilon"].mean())
            delta_eta = tau * (eps_mean - 1.0) * eta_base
            sfc_new   = sfc_base / (1.0 + delta_eta / eta_base)
            delta_sfc = (1.0 - sfc_new / sfc_base) * 100.0
            rows.append({
                "tau":          tau,
                "condition":    cond,
                "delta_sfc_pct": delta_sfc,
            })

    return pd.DataFrame(rows)
