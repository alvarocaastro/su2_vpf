"""Stage 1 — Airfoil Selection using SU2 RANS at reference condition."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from su2_analysis.adapters.su2.config_writer import write_su2_config
from su2_analysis.adapters.su2.mesh_generator import generate_cgrid_mesh
from su2_analysis.adapters.su2.su2_parser import (
    SU2RunResult, build_polar, parse_history,
)
from su2_analysis.adapters.su2.su2_runner import run_su2
from su2_analysis.config import AIRFOIL_DIR, STAGE_DIRS
from su2_analysis.config_loader import AnalysisConfig
from su2_analysis.pipeline.contracts import Stage1Result
from su2_analysis.settings import (
    TARGET_YPLUS, DPI, FIGURE_FORMAT,
    CL_MAX_MIN_ACCEPTABLE, CD_MIN_MAX_ACCEPTABLE,
)
from su2_analysis.shared.atmosphere import (
    wall_spacing_for_yplus, isa_temperature,
)
from su2_analysis.shared.plot_style import apply_style, PALETTE
from su2_analysis.stage1_airfoil_selection.scoring import score_airfoils

log = logging.getLogger(__name__)


def run_stage1(cfg: AnalysisConfig) -> Stage1Result:
    """Run airfoil selection and return a Stage1Result."""
    apply_style()
    out_dir = STAGE_DIRS["stage1"]
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = cfg.reference_condition
    mach    = float(ref["mach"])
    reynolds = float(ref["reynolds"])
    alpha_list = list(np.arange(
        float(ref["alpha_sweep"][0]),
        float(ref["alpha_sweep"][1]) + 0.5 * float(ref["alpha_sweep"][2]),
        float(ref["alpha_sweep"][2]),
    ))

    # Use sea-level temperature for reference condition
    T_inf = isa_temperature(0.0)

    # Reference chord = mid-section chord
    chord = cfg.fan_geometry.sections["mid"].chord
    # Approximate freestream velocity from Re and sea-level properties
    from su2_analysis.shared.atmosphere import isa_density, sutherland_viscosity
    rho = isa_density(0.0)
    mu  = sutherland_viscosity(T_inf)
    velocity = reynolds * mu / (rho * chord)
    ws = wall_spacing_for_yplus(TARGET_YPLUS, velocity, chord, 0.0)

    polars: Dict[str, pd.DataFrame] = {}

    for airfoil_name in cfg.airfoil_candidates:
        log.info("Stage 1 — running SU2 polar for %s", airfoil_name)
        dat_file = AIRFOIL_DIR / f"{airfoil_name}.dat"
        if not dat_file.exists():
            log.warning("Airfoil file not found: %s, skipping", dat_file)
            continue

        foil_dir = out_dir / airfoil_name
        mesh_dir = foil_dir / "mesh"
        run_dir  = foil_dir / "runs"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)

        mesh_file = mesh_dir / f"{airfoil_name}.su2"
        log.info("  Generating mesh → %s", mesh_file.name)
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
            aoa_dir = run_dir / f"aoa_{alpha:+.1f}"
            aoa_dir.mkdir(exist_ok=True)
            cfg_file = aoa_dir / "config.cfg"
            write_su2_config(
                output_path=cfg_file,
                mesh_file=mesh_file,
                mach=mach,
                aoa=alpha,
                reynolds=reynolds,
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
                run_result = parse_history(history, alpha)
            except Exception as exc:
                log.warning("  AoA %+.1f° failed: %s", alpha, exc)
                run_result = SU2RunResult(aoa=alpha, cl=float("nan"),
                                          cd=float("nan"), cm=float("nan"),
                                          converged=False, n_iter=0)
            results.append(run_result)
            log.debug("    AoA %+.1f° → CL=%.4f CD=%.5f", alpha, run_result.cl, run_result.cd)

        polar_df = build_polar(results)
        polar_df.to_csv(foil_dir / "polar.csv", index=False)
        polars[airfoil_name] = polar_df

    # ── Scoring ───────────────────────────────────────────────────────────────
    ranking = score_airfoils(polars)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    selected = ranking.iloc[0]["airfoil"]
    log.info("Stage 1 — selected airfoil: %s", selected)

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_polar_comparison(polars, selected, out_dir)

    return Stage1Result(
        selected_airfoil=selected,
        ranking=ranking,
        polars=polars,
        output_dir=out_dir,
    )


def _plot_polar_comparison(
    polars: Dict[str, pd.DataFrame],
    selected: str,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, (name, polar) in enumerate(polars.items()):
        color = PALETTE[i % len(PALETTE)]
        lw = 2.5 if name == selected else 1.2
        ls = "-" if name == selected else "--"
        label = f"{name} ★" if name == selected else name
        df = polar[polar["converged"]]
        axes[0].plot(df["alpha"], df["cl"], color=color, lw=lw, ls=ls, label=label)
        axes[1].plot(df["alpha"], df["cd"] * 1e4, color=color, lw=lw, ls=ls)
        axes[2].plot(df["alpha"], df["ld"], color=color, lw=lw, ls=ls)

    axes[0].set(xlabel="α [°]", ylabel="CL", title="Lift Curve")
    axes[1].set(xlabel="α [°]", ylabel="CD × 10⁴", title="Drag Polar")
    axes[2].set(xlabel="α [°]", ylabel="CL/CD", title="Glide Ratio")
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage 1 — Airfoil Comparison (SU2 RANS, M=0.3, Re=3M)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"polar_comparison.{FIGURE_FORMAT}", dpi=DPI)
    plt.close(fig)
