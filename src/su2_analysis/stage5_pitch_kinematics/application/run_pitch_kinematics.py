"""Stage 5 — Pitch Kinematics orchestrator."""
from __future__ import annotations
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from su2_analysis.config import STAGE_DIRS
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.pipeline.contracts import Stage4Result, Stage5Result
from su2_analysis.settings import DPI, FIGURE_FORMAT, PHI_MIN, PHI_MAX, PSI_MIN, PSI_MAX
from su2_analysis.shared.plot_style import apply_style, CONDITION_COLORS, PALETTE
from su2_analysis.stage5_pitch_kinematics.core.services.cascade_correction_service import (
    build_cascade_table,
)
from su2_analysis.stage5_pitch_kinematics.core.services.rotational_correction_service import (
    build_rotational_table,
)
from su2_analysis.stage5_pitch_kinematics.core.services.blade_twist_service import (
    compute_blade_twist,
)
from su2_analysis.stage5_pitch_kinematics.core.services.stage_loading_service import (
    build_stage_loading,
)
from su2_analysis.stage5_pitch_kinematics.core.services.optimal_incidence_service import (
    compute_optimal_incidence, compute_off_design,
)
from su2_analysis.stage5_pitch_kinematics.core.services.pitch_adjustment_service import (
    compute_pitch_adjustment,
)
from su2_analysis.stage5_pitch_kinematics.core.services.kinematics_service import (
    build_velocity_triangles,
)

log = logging.getLogger(__name__)


def run_stage5(cfg: AnalysisConfig, stage4: Stage4Result, pitch_map: pd.DataFrame) -> Stage5Result:
    apply_style()
    out_dir = STAGE_DIRS["stage5"]
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ── [A] Cascade corrections ───────────────────────────────────────────────
    cascade = build_cascade_table(cfg)
    cascade.to_csv(out_dir / "tables" / "cascade_corrections.csv", index=False)

    # ── [B] Rotational corrections ────────────────────────────────────────────
    rotational = build_rotational_table(cfg, pitch_map)
    rotational.to_csv(out_dir / "tables" / "rotational_corrections.csv", index=False)

    # ── [C] Blade twist ───────────────────────────────────────────────────────
    blade_twist = compute_blade_twist(cfg, pitch_map)
    blade_twist.to_csv(out_dir / "tables" / "blade_twist.csv", index=False)

    # ── Optimal incidence & off-design ────────────────────────────────────────
    opt_incidence = compute_optimal_incidence(pitch_map)
    opt_incidence.to_csv(out_dir / "tables" / "optimal_incidence.csv", index=False)

    off_design = compute_off_design(pitch_map)
    off_design.to_csv(out_dir / "tables" / "off_design.csv", index=False)

    # ── [D] Pitch adjustment ──────────────────────────────────────────────────
    pitch_adj = compute_pitch_adjustment(cfg, pitch_map)
    pitch_adj.to_csv(out_dir / "tables" / "pitch_adjustment.csv", index=False)

    # ── Velocity triangles / kinematics ──────────────────────────────────────
    kinematics = build_velocity_triangles(cfg, pitch_map)
    kinematics.to_csv(out_dir / "tables" / "kinematics.csv", index=False)

    # ── [D] Stage loading — ideal & real ──────────────────────────────────────
    sl_ideal = build_stage_loading(cfg, pitch_map, scenario="ideal")
    sl_ideal.to_csv(out_dir / "tables" / "stage_loading_ideal.csv", index=False)

    sl_real  = build_stage_loading(cfg, pitch_map, scenario="real")
    sl_real.to_csv(out_dir / "tables" / "stage_loading_real.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_cascade_bars(cascade, out_dir)
    _plot_rotational_bars(rotational, out_dir)
    _plot_blade_twist(blade_twist, out_dir)
    _plot_phi_psi_map(sl_ideal, sl_real, out_dir)
    _plot_pitch_adjustment(pitch_adj, out_dir)
    _plot_velocity_triangles(kinematics, out_dir)

    summary = _build_summary(cascade, rotational, blade_twist, pitch_adj, sl_ideal, sl_real)
    (out_dir / "pitch_kinematics_summary.txt").write_text(summary)

    return Stage5Result(
        cascade_table=cascade,
        rotational_table=rotational,
        optimal_incidence_table=opt_incidence,
        pitch_adjustment_table=pitch_adj,
        blade_twist_table=blade_twist,
        off_design_table=off_design,
        kinematics_table=kinematics,
        stage_loading_ideal=sl_ideal,
        stage_loading_real=sl_real,
        summary_text=summary,
        output_dir=out_dir,
    )


def _plot_cascade_bars(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(df))
    axes[0].bar(x, df["weinig_factor"], color=PALETTE[:len(df)])
    axes[0].set(xticks=x, xticklabels=df["section"], ylabel="Weinig factor K",
                title="Cascade CL scale factor")
    axes[1].bar(x, df["carter_deviation_deg"], color=PALETTE[:len(df)])
    axes[1].set(xticks=x, xticklabels=df["section"], ylabel="δ [°]",
                title="Carter deviation angle")
    fig.suptitle("Stage 5A — Cascade Corrections")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"cascade_corrections.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_rotational_bars(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, section in enumerate(["root", "mid", "tip"]):
        sub = df[df["section"] == section]
        x = np.arange(len(sub))
        ax.bar(x + i * 0.25, sub["rotation_pct"], width=0.25,
               label=section, color=PALETTE[i])
    ax.set(ylabel="ΔCL_rot / CL_2D [%]",
           title="Stage 5B — Snel Rotational Correction by Condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"rotational_corrections.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_blade_twist(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["radius_m"], df["beta_metal_deg"], "o-", color=PALETTE[0], lw=2)
    ax.set(xlabel="Radius [m]", ylabel="β_metal [°]",
           title="Stage 5C — Blade Metal Angle Spanwise Distribution")
    if "twist_total_deg" in df.columns and not df["twist_total_deg"].isna().all():
        twist = df["twist_total_deg"].iloc[0]
        ax.text(0.05, 0.95, f"Total twist: {twist:.1f}°",
                transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"blade_twist.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_phi_psi_map(ideal: pd.DataFrame, real: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    # Design zone rectangle
    rect = mpatches.FancyBboxPatch(
        (PHI_MIN, PSI_MIN), PHI_MAX - PHI_MIN, PSI_MAX - PSI_MIN,
        boxstyle="square,pad=0", facecolor="lightgreen", alpha=0.2,
        edgecolor="green", lw=1.5, label="Dixon & Hall design zone",
    )
    ax.add_patch(rect)

    for cond in ideal["condition"].unique():
        sub_i = ideal[ideal["condition"] == cond]
        sub_r = real[real["condition"] == cond]
        c = CONDITION_COLORS.get(cond, "k")
        ax.scatter(sub_i["phi"], sub_i["psi"], color=c, marker="o", s=60,
                   label=f"{cond} (ideal)")
        ax.scatter(sub_r["phi"], sub_r["psi"], color=c, marker="x", s=60,
                   label=f"{cond} (real)")

    ax.set(xlabel="Flow coefficient φ = Va/U",
           ylabel="Work coefficient ψ = ΔVθ/U",
           title="Stage 5D — Stage Loading: Ideal vs Real (single actuator)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"phi_psi_map.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_pitch_adjustment(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [CONDITION_COLORS.get(c, "#888") for c in df["condition"]]
    ax.bar(df["condition"], df["delta_beta_deg"], color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(ylabel="Δβ [°]", title="Stage 5 — Required Pitch Adjustment per Flight Phase")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"pitch_adjustment.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_velocity_triangles(df: pd.DataFrame, out_dir: Path) -> None:
    conds = df["condition"].unique()
    fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 4))
    if len(conds) == 1:
        axes = [axes]
    for ax, cond in zip(axes, conds):
        sub = df[df["condition"] == cond]
        for _, row in sub.iterrows():
            Va, U, Vt = row["Va_ms"], row["U_ms"], row["Vtheta_ms"]
            # Inlet triangle: (0,0) → (Va,0) → (Va, -U)
            ax.annotate("", (Va, 0), (0, 0), arrowprops=dict(arrowstyle="->"))
            ax.annotate("", (Va, -U), (0, 0), arrowprops=dict(arrowstyle="->", color="blue"))
            ax.annotate("", (Va, 0), (Va, -U), arrowprops=dict(arrowstyle="->", color="red"))
        ax.set(title=cond, xlabel="Va [m/s]", ylabel="U [m/s]")
        ax.set_aspect("equal")
    fig.suptitle("Stage 5 — Velocity Triangles")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / f"velocity_triangles.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _build_summary(cascade, rotational, twist, pitch_adj, sl_ideal, sl_real) -> str:
    lines = ["=" * 60, "STAGE 5 — PITCH KINEMATICS SUMMARY", "=" * 60, ""]

    lines.append("A) CASCADE CORRECTIONS (Weinig + Carter)")
    for _, r in cascade.iterrows():
        lines.append(f"  {r['section']:6s}: Weinig K={r['weinig_factor']:.3f}"
                     f"  Carter δ={r['carter_deviation_deg']:.2f}°")
    lines.append("")

    lines.append("B) ROTATIONAL CORRECTIONS (Snel, a=3.0)")
    for _, r in rotational.iterrows():
        lines.append(f"  {r['condition']:8s} / {r['section']:4s}: "
                     f"ΔCL_rot = {r['delta_cl_snel']:.4f} "
                     f"({r['rotation_pct']:.1f}%)")
    lines.append("")

    lines.append("C) BLADE TWIST (cruise design)")
    for _, r in twist.iterrows():
        lines.append(f"  {r['section']:6s}: β_metal = {r['beta_metal_deg']:.2f}°"
                     f"  twist = {r.get('twist_total_deg', float('nan')):.1f}°")
    lines.append("")

    lines.append("D) PITCH ADJUSTMENT (mid-span hub control)")
    for _, r in pitch_adj.iterrows():
        lines.append(f"  {r['condition']:10s}: Δβ = {r['delta_beta_deg']:+.2f}°")
    lines.append("")

    in_zone_ideal = sl_ideal["in_design_zone"].sum()
    in_zone_real  = sl_real["in_design_zone"].sum()
    total = len(sl_ideal)
    lines.append(f"STAGE LOADING — ideal: {in_zone_ideal}/{total} in Dixon & Hall zone")
    lines.append(f"STAGE LOADING — real:  {in_zone_real}/{total} in Dixon & Hall zone")

    return "\n".join(lines)
