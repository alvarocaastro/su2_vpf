"""Score airfoil polars to select the best candidate for VPF application."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _second_peak_ld(polar: pd.DataFrame) -> float:
    """Return CL/CD at the second local maximum of the lift-to-drag curve.

    The second peak is defined as the highest (CL/CD) at alpha >= 1°,
    which avoids the flat-plate peak near zero incidence.
    """
    df = polar[polar["alpha"] >= 1.0].dropna(subset=["ld"])
    if df.empty:
        return float("nan")
    return float(df["ld"].max())


def _alpha_at_second_peak(polar: pd.DataFrame) -> float:
    """Return the angle of attack at the second (CL/CD) peak."""
    df = polar[polar["alpha"] >= 1.0].dropna(subset=["ld"])
    if df.empty:
        return float("nan")
    idx = df["ld"].idxmax()
    return float(df.loc[idx, "alpha"])


def _stall_margin(polar: pd.DataFrame) -> float:
    """Estimated stall margin [degrees] = alpha_stall - alpha_opt."""
    df = polar.dropna(subset=["cl"])
    if df.empty:
        return float("nan")
    cl_max = df["cl"].max()
    alpha_stall = float(df.loc[df["cl"].idxmax(), "alpha"])
    alpha_opt = _alpha_at_second_peak(polar)
    return alpha_stall - alpha_opt


def _cl_max(polar: pd.DataFrame) -> float:
    df = polar.dropna(subset=["cl"])
    return float(df["cl"].max()) if not df.empty else float("nan")


def score_airfoils(polars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute a ranking table for all candidate airfoils.

    Parameters
    ----------
    polars : mapping of airfoil name → polar DataFrame
             (columns: alpha, cl, cd, cm, ld, converged)

    Returns
    -------
    DataFrame with columns: airfoil, ld_2nd_peak, alpha_opt, cl_max,
    stall_margin, score, rank.
    """
    rows = []
    for name, polar in polars.items():
        ld_peak     = _second_peak_ld(polar)
        alpha_opt   = _alpha_at_second_peak(polar)
        cl_max_val  = _cl_max(polar)
        stall_mg    = _stall_margin(polar)
        rows.append({
            "airfoil":      name,
            "ld_2nd_peak":  ld_peak,
            "alpha_opt":    alpha_opt,
            "cl_max":       cl_max_val,
            "stall_margin": stall_mg,
        })

    df = pd.DataFrame(rows)

    # Composite score: 60% CL/CD, 25% stall margin, 15% CL_max
    # (all normalised to [0,1])
    def _normalise(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - lo) / (hi - lo)

    df["score"] = (
        0.60 * _normalise(df["ld_2nd_peak"])
        + 0.25 * _normalise(df["stall_margin"])
        + 0.15 * _normalise(df["cl_max"])
    )
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    return df
