"""Stage 7 — SFC Analysis & Mission Integration orchestrator."""
from __future__ import annotations
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from su2_analysis.config import STAGE_DIRS
from su2_analysis.config_loader import AnalysisConfig, EngineParameters
from su2_analysis.pipeline.contracts import Stage4Result, Stage6Result, Stage7Result
from su2_analysis.settings import DPI, FIGURE_FORMAT
from su2_analysis.shared.plot_style import apply_style, CONDITION_COLORS, PALETTE
from su2_analysis.stage7_sfc_analysis.core.services.propulsion_model_service import (
    compute_delta_eta, compute_epsilon,
)
from su2_analysis.stage7_sfc_analysis.core.services.sfc_analysis_service import (
    compute_sfc_sensitivity,
)
from su2_analysis.stage7_sfc_analysis.core.services.mission_analysis_service import (
    compute_mission_fuel_burn,
)

log = logging.getLogger(__name__)


def run_stage7(
    cfg: AnalysisConfig,
    engine: EngineParameters,
    stage4: Stage4Result,
    stage6: Stage6Result,
) -> Stage7Result:
    apply_style()
    out_dir = STAGE_DIRS["stage7"]
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ── Efficiency ratio ε ────────────────────────────────────────────────────
    epsilon_df = compute_epsilon(stage4.metrics)
    epsilon_df.to_csv(out_dir / "tables" / "epsilon.csv", index=False)

    # ── ΔSFC per condition ────────────────────────────────────────────────────
    sfc_df = compute_delta_eta(epsilon_df, engine)
    sfc_df.to_csv(out_dir / "tables" / "sfc_analysis.csv", index=False)

    # ── Section breakdown ──────────────────────────────────────────────────────
    section_df = epsilon_df[["condition", "section", "ld_max", "epsilon"]].copy()
    section_df.to_csv(out_dir / "tables" / "sfc_section_breakdown.csv", index=False)

    # ── τ sensitivity ──────────────────────────────────────────────────────────
    sensitivity = compute_sfc_sensitivity(epsilon_df, engine)
    sensitivity.to_csv(out_dir / "tables" / "sfc_sensitivity.csv", index=False)

    # ── Mission fuel burn ──────────────────────────────────────────────────────
    mission = compute_mission_fuel_burn(sfc_df, engine)
    mission.to_csv(out_dir / "tables" / "mission_fuel_burn.csv", index=False)

    # ── Weight saving contribution ─────────────────────────────────────────────
    weight_saving_kg = 0.0
    if not stage6.weight_table.empty:
        save_row = stage6.weight_table[
            stage6.weight_table["mechanism"].str.startswith("Net saving")
        ]
        weight_saving_kg = float(save_row["total_kg_2engines"].iloc[0]) if not save_row.empty else 0.0

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_sfc_breakdown(sfc_df, out_dir)
    _plot_epsilon_spanwise(section_df, out_dir)
    _plot_sensitivity(sensitivity, out_dir)
    _plot_mission_fuel(mission, out_dir)

    summary = _build_summary(sfc_df, mission, weight_saving_kg, engine)
    (out_dir / "sfc_analysis_summary.txt").write_text(summary)

    return Stage7Result(
        sfc_table=sfc_df,
        section_breakdown=section_df,
        sensitivity_table=sensitivity,
        mission_fuel_burn=mission,
        summary_text=summary,
        output_dir=out_dir,
    )


def _plot_sfc_breakdown(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = [CONDITION_COLORS.get(c, "#888") for c in df["condition"]]
    axes[0].bar(df["condition"], df["delta_sfc_pct"], color=colors)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set(ylabel="ΔSFC [%]", title="SFC Reduction with VPF")

    x = np.arange(len(df))
    axes[1].bar(x - 0.2, df["sfc_base"], 0.35, label="Fixed pitch (base)", color=PALETTE[1])
    axes[1].bar(x + 0.2, df["sfc_new"],  0.35, label="VPF", color=PALETTE[0])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["condition"])
    axes[1].set(ylabel="SFC [lb/(lbf·h)]", title="Baseline vs VPF SFC")
    axes[1].legend()
    fig.suptitle("Stage 7 — SFC Analysis")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"sfc_breakdown.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_epsilon_spanwise(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, cond in enumerate(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        ax.plot(sub["section"], sub["epsilon"], "o-",
                color=CONDITION_COLORS.get(cond, PALETTE[i]), label=cond, lw=2)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set(xlabel="Blade section", ylabel="ε = (CL/CD)_vpf / (CL/CD)_ref",
           title="Stage 7 — Efficiency Ratio ε Spanwise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"epsilon_spanwise.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_sensitivity(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, cond in enumerate(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        ax.plot(sub["tau"], sub["delta_sfc_pct"],
                color=CONDITION_COLORS.get(cond, PALETTE[i]), label=cond, lw=2)
    ax.set(xlabel="Transfer factor τ", ylabel="ΔSFC [%]",
           title="Stage 7 — SFC Sensitivity to τ (2D→3D damping)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"sfc_sensitivity.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_mission_fuel(df: pd.DataFrame, out_dir: Path) -> None:
    df_phases = df[df["phase"] != "TOTAL"].copy()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df_phases))
    ax.bar(x - 0.2, df_phases["fuel_base_lb"], 0.35, label="Fixed pitch", color=PALETTE[1])
    ax.bar(x + 0.2, df_phases["fuel_vpf_lb"],  0.35, label="VPF",          color=PALETTE[0])
    ax.set_xticks(x)
    ax.set_xticklabels(df_phases["phase"])
    ax.set(ylabel="Fuel burn [lb]", title="Stage 7 — Mission Fuel Burn: Baseline vs VPF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"mission_fuel_burn.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _build_summary(
    sfc_df: pd.DataFrame,
    mission: pd.DataFrame,
    weight_saving_kg: float,
    engine: EngineParameters,
) -> str:
    total_row = mission[mission["phase"] == "TOTAL"]
    fuel_save_kg = float(total_row["fuel_saving_kg"].iloc[0]) if not total_row.empty else 0.0

    lines = [
        "=" * 60,
        "STAGE 7 — SFC ANALYSIS SUMMARY",
        "=" * 60,
        "",
        "SFC REDUCTION PER FLIGHT PHASE (τ = {:.2f})".format(engine.tau),
    ]
    for _, r in sfc_df.iterrows():
        lines.append(
            f"  {r['condition']:10s}: ΔSFC = {r['delta_sfc_pct']:+.2f}%"
            f"  (ε̄ = {r['epsilon_mean']:.3f}, Δη = {r['delta_eta']:.4f})"
        )
    lines += [
        "",
        "MISSION FUEL BURN (per flight, 2 engines)",
        f"  Fixed-pitch total:  {float(total_row['fuel_base_lb'].iloc[0]):.0f} lb",
        f"  VPF total:          {float(total_row['fuel_vpf_lb'].iloc[0]):.0f} lb",
        f"  Fuel saving:        {fuel_save_kg:.1f} kg",
        "",
        "MECHANISM WEIGHT SAVING",
        f"  VPF vs cascade reverser: {weight_saving_kg:.0f} kg (2-engine installation)",
        "",
        "COMBINED BENEFIT",
        f"  Fuel saving/flight + {weight_saving_kg:.0f} kg lighter airframe",
    ]
    return "\n".join(lines)
