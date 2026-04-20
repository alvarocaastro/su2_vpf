"""Compute the optimal pitch map across flight conditions and blade sections."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def compute_pitch_map(polars: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a pitch map from polar data.

    Parameters
    ----------
    polars : dict keyed by "{condition}_{section}" with polar DataFrames
             containing columns [alpha, cl, cd, ld, converged].

    Returns
    -------
    DataFrame with columns: condition, section, alpha_opt, cl_opt,
    cd_opt, ld_opt.
    """
    rows = []
    for key, polar in polars.items():
        parts = key.split("_", 1)
        condition = parts[0]
        section   = parts[1] if len(parts) > 1 else "unknown"

        df = polar[polar["converged"] & (polar["alpha"] >= 1.0)]
        if df.empty:
            df = polar[polar["converged"]]
        if df.empty:
            rows.append({
                "condition": condition, "section": section,
                "alpha_opt": float("nan"), "cl_opt": float("nan"),
                "cd_opt": float("nan"), "ld_opt": float("nan"),
            })
            continue

        idx = df["ld"].idxmax()
        best = df.loc[idx]
        rows.append({
            "condition": condition,
            "section":   section,
            "alpha_opt": float(best["alpha"]),
            "cl_opt":    float(best["cl"]),
            "cd_opt":    float(best["cd"]),
            "ld_opt":    float(best["ld"]),
        })

    return pd.DataFrame(rows)
