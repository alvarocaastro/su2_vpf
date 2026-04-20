"""SU2-VPF Pipeline — main entry point.

Runs all 7 analysis stages sequentially for a Variable Pitch Fan analysis
using SU2 8.4.0 RANS as the aerodynamic solver (replacing XFOIL).

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

# ── ensure src/ is on the Python path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from su2_analysis.config import CONFIG_FILE, ENGINE_PARAMS_FILE, STAGE_DIRS, STAGE_NAMES
from su2_analysis.config_loader import load_analysis_config, load_engine_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_analysis")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SU2-VPF aerodynamic pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--stages", type=int, nargs="+", metavar="N",
        help="Run only these stage numbers (e.g. 1 2 3)",
    )
    group.add_argument(
        "--from-stage", type=int, metavar="N",
        help="Resume from stage N (runs N through 7)",
    )
    return parser.parse_args()


def _should_run(stage_num: int, args: argparse.Namespace) -> bool:
    if args.stages:
        return stage_num in args.stages
    if args.from_stage:
        return stage_num >= args.from_stage
    return True


def main() -> None:
    args = _parse_args()
    t_start = time.monotonic()
    log.info("=" * 60)
    log.info("SU2-VPF Aerodynamic Pipeline")
    log.info("=" * 60)

    # ── Load configuration ────────────────────────────────────────────────────
    cfg    = load_analysis_config(CONFIG_FILE)
    engine = load_engine_parameters(ENGINE_PARAMS_FILE)
    log.info("Configuration loaded. SU2 executable: %s", cfg.su2.executable)

    results: dict = {}

    # ── Stage 1 — Airfoil Selection ───────────────────────────────────────────
    if _should_run(1, args):
        log.info("── Stage 1: %s ──", STAGE_NAMES["stage1"])
        t0 = time.monotonic()
        from su2_analysis.stage1_airfoil_selection.airfoil_selection_service import run_stage1
        results["stage1"] = run_stage1(cfg)
        log.info("Stage 1 done in %.1f s — selected: %s",
                 time.monotonic() - t0, results["stage1"].selected_airfoil)

    # ── Stage 2 — SU2 RANS Simulations ───────────────────────────────────────
    if _should_run(2, args):
        log.info("── Stage 2: %s ──", STAGE_NAMES["stage2"])
        t0 = time.monotonic()
        selected = (results.get("stage1") and results["stage1"].selected_airfoil) or "naca_65-410"
        from su2_analysis.stage2_su2_simulations.final_analysis_service import run_stage2
        results["stage2"] = run_stage2(cfg, selected)
        log.info("Stage 2 done in %.1f s — %d polars",
                 time.monotonic() - t0, len(results["stage2"].polars))

    # ── Stage 3 — CFD Post-Processing ────────────────────────────────────────
    if _should_run(3, args):
        log.info("── Stage 3: %s ──", STAGE_NAMES["stage3"])
        t0 = time.monotonic()
        s2 = results.get("stage2") or _load_stage2_fallback(cfg)
        from su2_analysis.stage3_cfd_postprocessing.postprocessing_service import run_stage3
        results["stage3"] = run_stage3(cfg, s2)
        log.info("Stage 3 done in %.1f s", time.monotonic() - t0)

    # ── Stage 4 — Performance Metrics ────────────────────────────────────────
    if _should_run(4, args):
        log.info("── Stage 4: %s ──", STAGE_NAMES["stage4"])
        t0 = time.monotonic()
        s2 = results.get("stage2") or _load_stage2_fallback(cfg)
        from su2_analysis.stage4_performance_metrics.metrics import run_stage4
        results["stage4"] = run_stage4(cfg, s2)
        log.info("Stage 4 done in %.1f s — %d metrics rows",
                 time.monotonic() - t0, len(results["stage4"].metrics))

    # ── Stage 5 — Pitch Kinematics ────────────────────────────────────────────
    if _should_run(5, args):
        log.info("── Stage 5: %s ──", STAGE_NAMES["stage5"])
        t0 = time.monotonic()
        s4 = results.get("stage4") or _load_stage4_fallback()
        pitch_map = _get_pitch_map(results, cfg)
        from su2_analysis.stage5_pitch_kinematics.application.run_pitch_kinematics import run_stage5
        results["stage5"] = run_stage5(cfg, s4, pitch_map)
        log.info("Stage 5 done in %.1f s", time.monotonic() - t0)

    # ── Stage 6 — Reverse Thrust ──────────────────────────────────────────────
    if _should_run(6, args):
        log.info("── Stage 6: %s ──", STAGE_NAMES["stage6"])
        t0 = time.monotonic()
        s5 = results.get("stage5") or _load_stage5_fallback()
        from su2_analysis.stage6_reverse_thrust.application.run_reverse_thrust import run_stage6
        results["stage6"] = run_stage6(cfg, engine, s5)
        log.info("Stage 6 done in %.1f s", time.monotonic() - t0)

    # ── Stage 7 — SFC Analysis ────────────────────────────────────────────────
    if _should_run(7, args):
        log.info("── Stage 7: %s ──", STAGE_NAMES["stage7"])
        t0 = time.monotonic()
        s4 = results.get("stage4") or _load_stage4_fallback()
        s6 = results.get("stage6") or _load_stage6_fallback()
        from su2_analysis.stage7_sfc_analysis.application.run_sfc_analysis import run_stage7
        results["stage7"] = run_stage7(cfg, engine, s4, s6)
        log.info("Stage 7 done in %.1f s", time.monotonic() - t0)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    log.info("=" * 60)
    log.info("Pipeline complete in %.1f s (%.1f min)", elapsed, elapsed / 60)
    log.info("Results written to: %s", STAGE_DIRS["stage1"].parent)

    if "stage7" in results:
        sfc = results["stage7"].sfc_table
        log.info("SFC improvement summary:")
        for _, r in sfc.iterrows():
            log.info("  %-10s ΔSFC = %+.2f%%", r["condition"], r["delta_sfc_pct"])


# ── Fallback loaders (when resuming from a later stage) ───────────────────────

def _load_stage2_fallback(cfg) -> "Stage2Result":
    """Load Stage 2 results from disk when not computed in this run."""
    import pandas as pd
    from su2_analysis.pipeline.contracts import Stage2Result
    from su2_analysis.config import STAGE_DIRS

    out_dir = STAGE_DIRS["stage2"]
    polars = {}
    for csv_file in out_dir.rglob("polar.csv"):
        parts = csv_file.parts
        # reconstruct key from directory structure: .../condition/section/polar.csv
        try:
            section = parts[-2]
            cond    = parts[-3]
            key     = f"{cond}_{section}"
            polars[key] = pd.read_csv(csv_file)
        except Exception:
            pass

    pitch_map_csv = out_dir / "pitch_map.csv"
    pitch_map = pd.read_csv(pitch_map_csv) if pitch_map_csv.exists() else pd.DataFrame()

    return Stage2Result(
        selected_airfoil="naca_65-410",
        polars=polars,
        pitch_map=pitch_map,
        output_dir=out_dir,
    )


def _get_pitch_map(results: dict, cfg) -> "pd.DataFrame":
    import pandas as pd
    if "stage2" in results:
        return results["stage2"].pitch_map
    from su2_analysis.config import STAGE_DIRS
    pm_csv = STAGE_DIRS["stage2"] / "pitch_map.csv"
    if pm_csv.exists():
        return pd.read_csv(pm_csv)
    # Generate synthetic pitch map from config defaults
    rows = []
    for cond in cfg.flight_conditions:
        for section in ["root", "mid", "tip"]:
            rows.append({"condition": cond, "section": section,
                         "alpha_opt": 4.0, "cl_opt": 0.8,
                         "cd_opt": 0.015, "ld_opt": 53.0})
    return pd.DataFrame(rows)


def _load_stage4_fallback() -> "Stage4Result":
    import pandas as pd
    from su2_analysis.pipeline.contracts import Stage4Result
    from su2_analysis.config import STAGE_DIRS

    out_dir = STAGE_DIRS["stage4"]
    csv = out_dir / "tables" / "metrics_summary.csv"
    metrics = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    return Stage4Result(metrics=metrics, output_dir=out_dir)


def _load_stage5_fallback() -> "Stage5Result":
    import pandas as pd
    from su2_analysis.pipeline.contracts import Stage5Result
    from su2_analysis.config import STAGE_DIRS

    out_dir = STAGE_DIRS["stage5"]

    def _load(name: str) -> pd.DataFrame:
        p = out_dir / "tables" / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    return Stage5Result(
        cascade_table=_load("cascade_corrections.csv"),
        rotational_table=_load("rotational_corrections.csv"),
        optimal_incidence_table=_load("optimal_incidence.csv"),
        pitch_adjustment_table=_load("pitch_adjustment.csv"),
        blade_twist_table=_load("blade_twist.csv"),
        off_design_table=_load("off_design.csv"),
        kinematics_table=_load("kinematics.csv"),
        stage_loading_ideal=_load("stage_loading_ideal.csv"),
        stage_loading_real=_load("stage_loading_real.csv"),
        summary_text="",
        output_dir=out_dir,
    )


def _load_stage6_fallback() -> "Stage6Result":
    import pandas as pd
    from su2_analysis.pipeline.contracts import Stage6Result
    from su2_analysis.config import STAGE_DIRS

    out_dir = STAGE_DIRS["stage6"]

    def _load(name: str) -> pd.DataFrame:
        p = out_dir / "tables" / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    return Stage6Result(
        sweep_table=_load("reverse_thrust_sweep.csv"),
        optimal_table=_load("reverse_thrust_optimal.csv"),
        kinematics_table=_load("reverse_kinematics.csv"),
        weight_table=_load("mechanism_weight.csv"),
        summary_text="",
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
