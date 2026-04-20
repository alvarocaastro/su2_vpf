"""SU2-VPF Pipeline — main entry point.

Runs all 7 analysis stages sequentially for a Variable Pitch Fan analysis
using SU2 8.4.0 RANS as the aerodynamic solver.

Usage
-----
    python run_analysis.py                  # full pipeline
    python run_analysis.py --stages 1 2     # only stages 1 and 2
    python run_analysis.py --from-stage 4   # resume from stage 4
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from su2_analysis.config import CONFIG_FILE, ENGINE_PARAMS_FILE, STAGE_DIRS, STAGE_NAMES
from su2_analysis.config_loader import load_analysis_config, load_engine_parameters
from su2_analysis.shared.progress import (
    banner, section, step, ok, warn, info, stage_banner, stage_done,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-40s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_analysis")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SU2-VPF aerodynamic pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stages",     type=int, nargs="+", metavar="N")
    group.add_argument("--from-stage", type=int, metavar="N")
    return parser.parse_args()


def _should_run(n: int, args: argparse.Namespace) -> bool:
    if args.stages:     return n in args.stages
    if args.from_stage: return n >= args.from_stage
    return True


def main() -> None:
    args    = _parse_args()
    t_start = time.monotonic()

    banner("SU2-VPF  ·  Variable Pitch Fan Aerodynamic Pipeline  ·  v1.0")

    # ── Configuration ─────────────────────────────────────────────────────────
    section("Loading configuration")
    step("Reading analysis_config.yaml")
    cfg = load_analysis_config(CONFIG_FILE)
    ok(f"Fan geometry loaded — {cfg.fan_geometry.n_blades} blades, "
       f"{cfg.fan_geometry.rpm} RPM")

    step("Reading engine_parameters.yaml")
    engine = load_engine_parameters(ENGINE_PARAMS_FILE)
    ok(f"Engine parameters loaded — baseline SFC {engine.baseline_sfc} lb/(lbf·h)")

    info(f"SU2 executable : {cfg.su2.executable}")
    info(f"Turbulence model: {cfg.su2.turbulence_model}  |  "
         f"max_iter: {cfg.su2.max_iter}  |  CFL: {cfg.su2.cfl_number} (adaptive)")
    info(f"Airfoil candidates: {', '.join(cfg.airfoil_candidates)}")
    info(f"Flight conditions : {', '.join(cfg.flight_conditions.keys())}")
    info(f"Alpha sweep (Stage 2): {cfg.alpha_sweep_simulations.start}° → "
         f"{cfg.alpha_sweep_simulations.end}°  step {cfg.alpha_sweep_simulations.step}°")

    results: dict = {}

    # ── Stage 1 — Airfoil Selection ───────────────────────────────────────────
    if _should_run(1, args):
        stage_banner(1, STAGE_NAMES["stage1"])
        info(f"Candidates: {', '.join(cfg.airfoil_candidates)}")
        info("Running restart-chained SU2 RANS polars at M=0.3, Re=3M, α step=2°")
        t0 = time.monotonic()
        from su2_analysis.stage1_airfoil_selection.airfoil_selection_service import run_stage1
        results["stage1"] = run_stage1(cfg)
        stage_done(1, STAGE_NAMES["stage1"], time.monotonic() - t0)
        ok(f"Selected airfoil: {results['stage1'].selected_airfoil}")

    # ── Stage 2 — SU2 RANS Simulations ───────────────────────────────────────
    if _should_run(2, args):
        stage_banner(2, STAGE_NAMES["stage2"])
        selected = (results.get("stage1") and results["stage1"].selected_airfoil) \
                   or "naca_65-410"
        n_cond   = len(cfg.flight_conditions)
        n_sec    = 3
        n_alpha  = len(cfg.alpha_sweep_simulations.to_list())
        info(f"Airfoil  : {selected}")
        info(f"Runs     : {n_cond} conditions × {n_sec} sections = {n_cond*n_sec} polars")
        info(f"AoA pts  : {n_alpha} per polar  (restart-chained)")
        info(f"Total SU2 runs (est.): {n_cond * n_sec * n_alpha}  "
             f"(1st cold, rest warm-started)")
        t0 = time.monotonic()
        from su2_analysis.stage2_su2_simulations.final_analysis_service import run_stage2
        results["stage2"] = run_stage2(cfg, selected)
        stage_done(2, STAGE_NAMES["stage2"], time.monotonic() - t0)
        ok(f"{len(results['stage2'].polars)} polars generated")

    # ── Stage 3 — CFD Post-Processing ────────────────────────────────────────
    if _should_run(3, args):
        stage_banner(3, STAGE_NAMES["stage3"])
        info("Extracting Cp distributions and Mach surface from SU2 solution files")
        info("Detecting shock location via Korn drag-divergence equation")
        t0 = time.monotonic()
        s2 = results.get("stage2") or _load_stage2_fallback(cfg)
        from su2_analysis.stage3_cfd_postprocessing.postprocessing_service import run_stage3
        results["stage3"] = run_stage3(cfg, s2)
        stage_done(3, STAGE_NAMES["stage3"], time.monotonic() - t0)
        ok(f"Mach summary written — "
           f"{results['stage3'].mach_summary['wave_drag'].sum()} cases with wave drag")

    # ── Stage 4 — Performance Metrics ────────────────────────────────────────
    if _should_run(4, args):
        stage_banner(4, STAGE_NAMES["stage4"])
        info("Computing CL/CD_max, α_opt, stall margin, VPF vs fixed-pitch benefit")
        t0 = time.monotonic()
        s2 = results.get("stage2") or _load_stage2_fallback(cfg)
        from su2_analysis.stage4_performance_metrics.metrics import run_stage4
        results["stage4"] = run_stage4(cfg, s2)
        stage_done(4, STAGE_NAMES["stage4"], time.monotonic() - t0)
        m = results["stage4"].metrics
        ok(f"{len(m)} metric rows — "
           f"max VPF benefit: {m['vpf_benefit_pct'].max():.1f}%")

    # ── Stage 5 — Pitch Kinematics ────────────────────────────────────────────
    if _should_run(5, args):
        stage_banner(5, STAGE_NAMES["stage5"])
        info("[A] Cascade corrections  — Weinig factor + Carter deviation")
        info("[B] Rotational corrections — Snel model (a=3.0)")
        info("[C] Blade twist design   — single-actuator cruise optimum")
        info("[D] Stage loading        — ideal (free pitch) vs real (hub control)")
        t0 = time.monotonic()
        s4        = results.get("stage4") or _load_stage4_fallback()
        pitch_map = _get_pitch_map(results, cfg)
        from su2_analysis.stage5_pitch_kinematics.application.run_pitch_kinematics import run_stage5
        results["stage5"] = run_stage5(cfg, s4, pitch_map)
        stage_done(5, STAGE_NAMES["stage5"], time.monotonic() - t0)
        ok("10 tables + 20 figures written")

    # ── Stage 6 — Reverse Thrust ──────────────────────────────────────────────
    if _should_run(6, args):
        stage_banner(6, STAGE_NAMES["stage6"])
        rt = engine.reverse_thrust
        info(f"Sweeping Δβ from {rt.delta_beta_sweep_start}° to "
             f"{rt.delta_beta_sweep_end}°  ({rt.delta_beta_sweep_points} points)")
        info(f"Operating point: N1={rt.rpm_fraction*100:.0f}%,  "
             f"Va={rt.axial_velocity} m/s  (ground roll)")
        info("Comparing VPF actuator weight vs conventional cascade reverser")
        t0 = time.monotonic()
        s5 = results.get("stage5") or _load_stage5_fallback()
        from su2_analysis.stage6_reverse_thrust.application.run_reverse_thrust import run_stage6
        results["stage6"] = run_stage6(cfg, engine, s5)
        stage_done(6, STAGE_NAMES["stage6"], time.monotonic() - t0)
        if not results["stage6"].optimal_table.empty:
            opt = results["stage6"].optimal_table.iloc[0]
            ok(f"Optimal Δβ = {opt['delta_beta_deg']:+.1f}°  →  "
               f"T_rev = {opt['T_reverse_N']/1000:.1f} kN")

    # ── Stage 7 — SFC Analysis ────────────────────────────────────────────────
    if _should_run(7, args):
        stage_banner(7, STAGE_NAMES["stage7"])
        info(f"Transfer factor τ = {engine.tau}  (2D RANS → 3D fan damping)")
        info("Integrating fuel burn over takeoff / climb / cruise / descent")
        info("Comparing mission fuel: fixed-pitch baseline vs VPF")
        t0 = time.monotonic()
        s4 = results.get("stage4") or _load_stage4_fallback()
        s6 = results.get("stage6") or _load_stage6_fallback()
        from su2_analysis.stage7_sfc_analysis.application.run_sfc_analysis import run_stage7
        results["stage7"] = run_stage7(cfg, engine, s4, s6)
        stage_done(7, STAGE_NAMES["stage7"], time.monotonic() - t0)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    banner(f"Pipeline complete — {elapsed:.1f} s  ({elapsed/60:.1f} min)")

    if "stage7" in results:
        section("SFC Results Summary")
        sfc = results["stage7"].sfc_table
        for _, r in sfc.iterrows():
            sign  = "+" if r["delta_sfc_pct"] > 0 else ""
            color = "\033[92m" if r["delta_sfc_pct"] > 0 else "\033[91m"
            print(f"  {color}{r['condition']:10s}  ΔSFC = "
                  f"{sign}{r['delta_sfc_pct']:.2f}%\033[0m")

        mission = results["stage7"].mission_fuel_burn
        total   = mission[mission["phase"] == "TOTAL"]
        if not total.empty:
            saving_kg = float(total["fuel_saving_kg"].iloc[0])
            ok(f"Total fuel saving per flight: {saving_kg:.1f} kg")

    section("Output directories")
    for key, d in STAGE_DIRS.items():
        info(f"{STAGE_NAMES[key]:30s} → {d}")


# ── Fallback loaders ──────────────────────────────────────────────────────────

def _load_stage2_fallback(cfg) -> "Stage2Result":
    import pandas as pd
    from su2_analysis.pipeline.contracts import Stage2Result
    section("Loading Stage 2 results from disk")
    out_dir = STAGE_DIRS["stage2"]
    polars  = {}
    for csv_file in out_dir.rglob("polar.csv"):
        parts = csv_file.parts
        try:
            section_ = parts[-2]; cond = parts[-3]
            polars[f"{cond}_{section_}"] = pd.read_csv(csv_file)
        except Exception:
            pass
    pm_csv = out_dir / "pitch_map.csv"
    pm     = pd.read_csv(pm_csv) if pm_csv.exists() else pd.DataFrame()
    info(f"Loaded {len(polars)} polars from {out_dir}")
    return Stage2Result(selected_airfoil="naca_65-410",
                        polars=polars, pitch_map=pm, output_dir=out_dir)


def _get_pitch_map(results: dict, cfg) -> "pd.DataFrame":
    import pandas as pd
    if "stage2" in results:
        return results["stage2"].pitch_map
    pm_csv = STAGE_DIRS["stage2"] / "pitch_map.csv"
    if pm_csv.exists():
        return pd.read_csv(pm_csv)
    warn("Pitch map not found — using default α_opt = 4° for all conditions")
    rows = []
    for cond in cfg.flight_conditions:
        for sec in ["root", "mid", "tip"]:
            rows.append({"condition": cond, "section": sec,
                         "alpha_opt": 4.0, "cl_opt": 0.8,
                         "cd_opt": 0.015, "ld_opt": 53.0})
    return pd.DataFrame(rows)


def _load_stage4_fallback() -> "Stage4Result":
    import pandas as pd
    from su2_analysis.pipeline.contracts import Stage4Result
    out_dir = STAGE_DIRS["stage4"]
    csv     = out_dir / "tables" / "metrics_summary.csv"
    info(f"Loading Stage 4 metrics from {csv}")
    return Stage4Result(metrics=pd.read_csv(csv) if csv.exists() else pd.DataFrame(),
                        output_dir=out_dir)


def _load_stage5_fallback() -> "Stage5Result":
    import pandas as pd
    from su2_analysis.pipeline.contracts import Stage5Result
    out_dir = STAGE_DIRS["stage5"]
    def _l(n): p = out_dir/"tables"/n; return pd.read_csv(p) if p.exists() else pd.DataFrame()
    info("Loading Stage 5 tables from disk")
    return Stage5Result(
        cascade_table=_l("cascade_corrections.csv"),
        rotational_table=_l("rotational_corrections.csv"),
        optimal_incidence_table=_l("optimal_incidence.csv"),
        pitch_adjustment_table=_l("pitch_adjustment.csv"),
        blade_twist_table=_l("blade_twist.csv"),
        off_design_table=_l("off_design.csv"),
        kinematics_table=_l("kinematics.csv"),
        stage_loading_ideal=_l("stage_loading_ideal.csv"),
        stage_loading_real=_l("stage_loading_real.csv"),
        summary_text="", output_dir=out_dir,
    )


def _load_stage6_fallback() -> "Stage6Result":
    import pandas as pd
    from su2_analysis.pipeline.contracts import Stage6Result
    out_dir = STAGE_DIRS["stage6"]
    def _l(n): p = out_dir/"tables"/n; return pd.read_csv(p) if p.exists() else pd.DataFrame()
    info("Loading Stage 6 tables from disk")
    return Stage6Result(
        sweep_table=_l("reverse_thrust_sweep.csv"),
        optimal_table=_l("reverse_thrust_optimal.csv"),
        kinematics_table=_l("reverse_kinematics.csv"),
        weight_table=_l("mechanism_weight.csv"),
        summary_text="", output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
