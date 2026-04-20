"""Stage 2 — Compressible SU2 RANS polars for all flight conditions × blade sections."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from su2_analysis.adapters.su2.config_writer import write_su2_config
from su2_analysis.adapters.su2.mesh_generator import generate_cgrid_mesh
from su2_analysis.adapters.su2.su2_parser import (
    SU2RunResult, build_polar, parse_history, parse_surface_flow,
)
from su2_analysis.adapters.su2.su2_runner import run_su2
from su2_analysis.config import AIRFOIL_DIR, STAGE_DIRS
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.pipeline.contracts import Stage2Result
from su2_analysis.settings import TARGET_YPLUS, DPI, FIGURE_FORMAT
from su2_analysis.shared.atmosphere import (
    blade_velocity, isa_temperature, relative_velocity, reynolds_number,
    speed_of_sound, wall_spacing_for_yplus,
)
from su2_analysis.shared.plot_style import (
    apply_style, CONDITION_COLORS, SECTION_LINESTYLES,
)
from su2_analysis.stage2_su2_simulations.pitch_map import compute_pitch_map

log = logging.getLogger(__name__)

SECTIONS  = ["root", "mid", "tip"]
CONDITIONS = ["takeoff", "climb", "cruise", "descent"]


def run_stage2(cfg: AnalysisConfig, selected_airfoil: str) -> Stage2Result:
    apply_style()
    out_dir = STAGE_DIRS["stage2"]
    out_dir.mkdir(parents=True, exist_ok=True)

    dat_file = AIRFOIL_DIR / f"{selected_airfoil}.dat"
    alpha_list = cfg.alpha_sweep_simulations.to_list()
    polars: Dict[str, pd.DataFrame] = {}

    for cond_name, fc in cfg.flight_conditions.items():
        for section_name in SECTIONS:
            key = f"{cond_name}_{section_name}"
            log.info("Stage 2 — running SU2: %s", key)

            section = cfg.fan_geometry.sections[section_name]
            chord   = section.chord
            radius  = section.radius
            rpm     = cfg.fan_geometry.rpm
            alt     = fc.altitude

            # Blade relative velocity and Mach
            U   = blade_velocity(radius, rpm)
            W   = relative_velocity(fc.axial_velocity, U)
            a   = speed_of_sound(alt)
            mach_rel = W / a
            T_inf = isa_temperature(alt)
            Re = reynolds_number(W, chord, alt)
            ws = wall_spacing_for_yplus(TARGET_YPLUS, W, chord, alt)

            run_base = out_dir / cond_name / section_name
            mesh_dir = run_base / "mesh"
            mesh_dir.mkdir(parents=True, exist_ok=True)

            mesh_file = mesh_dir / f"{selected_airfoil}_{section_name}.su2"
            if not mesh_file.exists():
                log.info("  Generating mesh for %s / %s", cond_name, section_name)
                generate_cgrid_mesh(
                    airfoil_dat=dat_file,
                    output_mesh=mesh_file,
                    chord=chord,
                    wall_spacing=ws,
                    farfield_radius_chords=cfg.su2.mesh_farfield_radius,
                    n_airfoil_points=cfg.su2.mesh_airfoil_points,
                    n_radial_layers=cfg.su2.mesh_radial_layers,
                    growth_rate=cfg.su2.mesh_growth_rate,
                )

            results: List[SU2RunResult] = []
            for alpha in alpha_list:
                aoa_dir = run_base / f"aoa_{alpha:+.1f}"
                aoa_dir.mkdir(exist_ok=True)
                cfg_file = aoa_dir / "config.cfg"
                write_su2_config(
                    output_path=cfg_file,
                    mesh_file=mesh_file,
                    mach=mach_rel,
                    aoa=alpha,
                    reynolds=Re,
                    chord=chord,
                    T_inf=T_inf,
                    max_iter=cfg.su2.max_iter,
                    conv_residual=cfg.su2.convergence_residual,
                    cfl=cfg.su2.cfl_number,
                    turb_model=cfg.su2.turbulence_model,
                )
                try:
                    history = run_su2(
                        su2_exe=cfg.su2.executable,
                        cfg_file=cfg_file,
                        work_dir=aoa_dir,
                        timeout=cfg.su2.timeout_seconds,
                        max_retries=cfg.su2.max_retries,
                    )
                    r = parse_history(history, alpha)
                except Exception as exc:
                    log.warning("  %s AoA %+.1f° failed: %s", key, alpha, exc)
                    r = SU2RunResult(aoa=alpha, cl=float("nan"),
                                     cd=float("nan"), cm=float("nan"),
                                     converged=False, n_iter=0)
                results.append(r)

            polar_df = build_polar(results)
            polar_df["condition"] = cond_name
            polar_df["section"]   = section_name
            polar_df["mach"]      = mach_rel
            polar_df["reynolds"]  = Re
            polar_csv = run_base / "polar.csv"
            polar_df.to_csv(polar_csv, index=False)
            polars[key] = polar_df
            log.info("  %s polar saved (%d points)", key, len(polar_df))

    pitch_map = compute_pitch_map(polars)
    pitch_map.to_csv(out_dir / "pitch_map.csv", index=False)

    _plot_polars(polars, out_dir)
    _plot_pitch_map_heatmap(pitch_map, out_dir)

    return Stage2Result(
        selected_airfoil=selected_airfoil,
        polars=polars,
        pitch_map=pitch_map,
        output_dir=out_dir,
    )


def _plot_polars(polars: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    cond_ax = dict(zip(CONDITIONS, axes.flatten()))

    for key, polar in polars.items():
        cond, section = key.split("_", 1)
        ax = cond_ax[cond]
        df = polar[polar["converged"]]
        ax.plot(
            df["alpha"], df["ld"],
            color=CONDITION_COLORS[cond],
            ls=SECTION_LINESTYLES.get(section, "-"),
            label=section,
        )

    for cond, ax in cond_ax.items():
        ax.set(title=cond.capitalize(), xlabel="α [°]", ylabel="CL/CD")
        ax.legend(title="Section", fontsize=8)

    fig.suptitle("Stage 2 — Compressible RANS Polars (SU2)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"polars_overview.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)


def _plot_pitch_map_heatmap(pitch_map: pd.DataFrame, out_dir: Path) -> None:
    try:
        pivot = pitch_map.pivot(index="section", columns="condition", values="alpha_opt")
        pivot = pivot.reindex(index=SECTIONS, columns=CONDITIONS)
        fig, ax = plt.subplots(figsize=(7, 3))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, rotation=20)
        ax.set_yticks(range(len(SECTIONS)))
        ax.set_yticklabels(SECTIONS)
        plt.colorbar(im, ax=ax, label="α_opt [°]")
        for r in range(len(SECTIONS)):
            for c in range(len(CONDITIONS)):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    ax.text(c, r, f"{val:.1f}°", ha="center", va="center", fontsize=9)
        ax.set_title("Stage 2 — Optimal Pitch Map (α_opt) [degrees]")
        fig.tight_layout()
        fig.savefig(out_dir / f"alpha_opt_heatmap.{FIGURE_FORMAT}", dpi=DPI)
        plt.close(fig)
    except Exception as exc:
        log.warning("Could not generate pitch map heatmap: %s", exc)
