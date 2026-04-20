"""Parse SU2 output files into pandas DataFrames.

SU2 writes two key output files:
- history.csv  : residuals + integrated forces per iteration
- surface_flow.csv : surface field data (Cp, Mach, x, y) at converged state
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class SU2RunResult:
    """Aerodynamic coefficients from a converged SU2 RANS run."""
    aoa: float
    cl: float
    cd: float
    cm: float
    converged: bool
    n_iter: int


class SU2ParseError(ValueError):
    """Raised when expected columns are absent from SU2 output."""


# Column name map: SU2 history header → our standard names
_HISTORY_COL_MAP = {
    "\"CL\"":   "cl",
    "\"CD\"":   "cd",
    "\"CMz\"":  "cm",
    "CL":       "cl",
    "CD":       "cd",
    "CMz":      "cm",
    "LIFT":     "cl",
    "DRAG":     "cd",
    "MOMENT_Z": "cm",
}


def parse_history(history_csv: Path, aoa: float) -> SU2RunResult:
    """Extract converged CL, CD, CM from SU2 history.csv.

    Returns the values from the last iteration row.
    """
    df = _read_csv_flexible(history_csv)

    # Normalise column names
    df.columns = [c.strip().strip('"') for c in df.columns]
    col_map = {
        c: _HISTORY_COL_MAP[c]
        for c in df.columns
        if c in _HISTORY_COL_MAP
    }
    if not col_map:
        raise SU2ParseError(
            f"No recognised force columns in {history_csv}. "
            f"Available: {list(df.columns)}"
        )
    df = df.rename(columns=col_map)

    # Use last converged row
    last = df.iloc[-1]
    cl = float(last.get("cl", float("nan")))
    cd = float(last.get("cd", float("nan")))
    cm = float(last.get("cm", 0.0))
    n_iter = len(df)
    converged = not (np.isnan(cl) or np.isnan(cd) or cd < 0)

    return SU2RunResult(aoa=aoa, cl=cl, cd=cd, cm=cm, converged=converged, n_iter=n_iter)


def parse_surface_flow(surface_csv: Path) -> pd.DataFrame:
    """Parse surface_flow.csv into a DataFrame with columns [x, y, cp, mach].

    Returns empty DataFrame if the file does not exist.
    """
    if not surface_csv.exists():
        return pd.DataFrame()

    df = _read_csv_flexible(surface_csv)
    df.columns = [c.strip().strip('"').lower() for c in df.columns]

    # Possible column names for coordinates and Cp
    rename = {}
    for col in df.columns:
        if col in ("x", "x_coord", "\"x\""):
            rename[col] = "x"
        elif col in ("y", "y_coord", "\"y\""):
            rename[col] = "y"
        elif "pressure_coefficient" in col or col == "cp" or col == "\"cp\"":
            rename[col] = "cp"
        elif "mach" in col:
            rename[col] = "mach"

    df = df.rename(columns=rename)

    keep = [c for c in ("x", "y", "cp", "mach") if c in df.columns]
    df = df[keep].dropna()

    # Sort by x for plotting
    if "x" in df.columns:
        df = df.sort_values("x").reset_index(drop=True)

    return df


def build_polar(results: list[SU2RunResult]) -> pd.DataFrame:
    """Assemble a list of SU2RunResult into a polar DataFrame.

    Columns: alpha, cl, cd, cm, ld (lift-to-drag), converged.
    """
    rows = []
    for r in results:
        ld = r.cl / r.cd if r.cd and r.cd > 0 else float("nan")
        rows.append({
            "alpha": r.aoa,
            "cl": r.cl,
            "cd": r.cd,
            "cm": r.cm,
            "ld": ld,
            "converged": r.converged,
        })
    df = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    return df


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read a CSV that may use comma or tab separators and may have a comment line."""
    text = path.read_text(errors="replace")
    # Skip lines starting with '%' (SU2 comment marker)
    lines = [l for l in text.splitlines() if not l.strip().startswith("%")]
    from io import StringIO
    cleaned = "\n".join(lines)
    try:
        return pd.read_csv(StringIO(cleaned))
    except Exception:
        return pd.read_csv(StringIO(cleaned), sep=r"\s+", engine="python")
