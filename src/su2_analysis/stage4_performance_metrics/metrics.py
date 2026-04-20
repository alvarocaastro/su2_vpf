"""Stage 4 — Performance Metrics: CL/CD_max, α_opt, stall margin, VPF benefit."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from su2_analysis.config import STAGE_DIRS
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.pipeline.contracts import Stage2Result, Stage4Result
from su2_analysis.settings import DPI, FIGURE_FORMAT
from su2_analysis.shared.plot_style import apply_style, CONDITION_COLORS, SECTION_LINESTYLES

log = logging.getLogger(__name__)

CONDITIONS = ["takeoff", "climb", "cruise", "descent"]
SECTIONS   = ["root", "mid", "tip"]


def run_stage4(cfg: AnalysisConfig, stage2: Stage2Result) -> Stage4Result:
    apply_style()
    out_dir = STAGE_DIRS["stage4"]
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    # Cruise mid as fixed-pitch reference
    ref_key = "cruise_mid"
    ref_polar = stage2.polars.get(ref_key, pd.DataFrame())
    ref_ld = _ld_max_second(ref_polar)

    for key, polar in stage2.polars.items():
        cond, section = key.split("_", 1)
        df = polar[polar["converged"]] if "converged" in polar.columns else polar
        if df.empty:
            continue

        ld_max     = _ld_max_second(df)
        alpha_opt  = _alpha_at_ld_max_second(df)
        cl_max     = float(df["cl"].max())
        alpha_stall = float(df.loc[df["cl"].idxmax(), "alpha"])
        stall_mg   = alpha_stall - alpha_opt if not np.isnan(alpha_opt) else float("nan")
        cd_min     = float(df["cd"].min())
        cm_at_opt  = _cm_at_alpha(df, alpha_opt)
        vpf_benefit = (ld_max / ref_ld - 1.0) * 100.0 if ref_ld and not np.isnan(ref_ld) else float("nan")

        rows.append({
            "condition":     cond,
            "section":       section,
            "ld_max":        ld_max,
            "alpha_opt_deg": alpha_opt,
            "cl_max":        cl_max,
            "alpha_stall_deg": alpha_stall,
            "stall_margin_deg": stall_mg,
            "cd_min":        cd_min,
            "cm_at_opt":     cm_at_opt,
            "vpf_benefit_pct": vpf_benefit,
        })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "tables" / "metrics_summary.csv", index=False)

    _plot_heatmap(metrics, "ld_max",          "CL/CD max",     out_dir)
    _plot_heatmap(metrics, "stall_margin_deg", "Stall margin [°]", out_dir)
    _plot_heatmap(metrics, "vpf_benefit_pct",  "VPF benefit [%]",  out_dir)
    _plot_efficiency_gain(metrics, out_dir)

    return Stage4Result(metrics=metrics, output_dir=out_dir)


def _ld_max_second(df: pd.DataFrame) -> float:
    sub = df[df["alpha"] >= 1.0] if not df.empty and "alpha" in df.columns else df
    if sub.empty or "ld" not in sub.columns:
        return float("nan")
    return float(sub["ld"].max())


def _alpha_at_ld_max_second(df: pd.DataFrame) -> float:
    sub = df[df["alpha"] >= 1.0] if not df.empty and "alpha" in df.columns else df
    if sub.empty or "ld" not in sub.columns:
        return float("nan")
    return float(sub.loc[sub["ld"].idxmax(), "alpha"])


def _cm_at_alpha(df: pd.DataFrame, alpha: float) -> float:
    if np.isnan(alpha) or "cm" not in df.columns:
        return float("nan")
    idx = (df["alpha"] - alpha).abs().idxmin()
    return float(df.loc[idx, "cm"])


def _plot_heatmap(metrics: pd.DataFrame, col: str, label: str, out_dir: Path) -> None:
    try:
        pivot = metrics.pivot(index="section", columns="condition", values=col)
        pivot = pivot.reindex(index=SECTIONS, columns=CONDITIONS)
        fig, ax = plt.subplots(figsize=(7, 3))
        im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, rotation=20)
        ax.set_yticks(range(len(SECTIONS)))
        ax.set_yticklabels(SECTIONS)
        plt.colorbar(im, ax=ax, label=label)
        for r in range(len(SECTIONS)):
            for c in range(len(CONDITIONS)):
                v = pivot.values[r, c]
                if not np.isnan(float(v)):
                    ax.text(c, r, f"{float(v):.1f}", ha="center", va="center", fontsize=9)
        ax.set_title(f"Stage 4 — {label}")
        fig.tight_layout()
        safe = col.replace("/", "_")
        fig.savefig(out_dir / "figures" / f"heatmap_{safe}.{FIGURE_FORMAT}", dpi=DPI)
        plt.close(fig)
    except Exception as exc:
        log.warning("Heatmap failed for %s: %s", col, exc)


def _plot_efficiency_gain(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(metrics))
    labels = [f"{r.condition}/{r.section}" for _, r in metrics.iterrows()]
    colors = [CONDITION_COLORS.get(r.condition, "#888") for _, r in metrics.iterrows()]
    ax.bar(x, metrics["vpf_benefit_pct"], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set(ylabel="VPF benefit vs. fixed-pitch cruise [%]",
           title="Stage 4 — Efficiency Gain with Variable Pitch")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"efficiency_gain.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)
