"""Canonical paths and stage names for the su2_vpf pipeline."""
from pathlib import Path

# ── Root directories ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # su2_vpf/
SRC_ROOT     = Path(__file__).resolve().parent        # su2_analysis/

DATA_DIR     = PROJECT_ROOT / "data"
AIRFOIL_DIR  = DATA_DIR / "airfoils"
CONFIG_DIR   = PROJECT_ROOT / "config"
RESULTS_DIR  = PROJECT_ROOT / "results"

# ── Per-stage result directories ──────────────────────────────────────────────
STAGE_DIRS = {
    "stage1": RESULTS_DIR / "stage1_airfoil_selection",
    "stage2": RESULTS_DIR / "stage2_su2_simulations",
    "stage3": RESULTS_DIR / "stage3_cfd_postprocessing",
    "stage4": RESULTS_DIR / "stage4_performance_metrics",
    "stage5": RESULTS_DIR / "stage5_pitch_kinematics",
    "stage6": RESULTS_DIR / "stage6_reverse_thrust",
    "stage7": RESULTS_DIR / "stage7_sfc_analysis",
}

STAGE_NAMES = {
    "stage1": "Airfoil Selection",
    "stage2": "SU2 RANS Simulations",
    "stage3": "CFD Post-Processing",
    "stage4": "Performance Metrics",
    "stage5": "Pitch Kinematics",
    "stage6": "Reverse Thrust",
    "stage7": "SFC Analysis",
}

CONFIG_FILE          = CONFIG_DIR / "analysis_config.yaml"
ENGINE_PARAMS_FILE   = CONFIG_DIR / "engine_parameters.yaml"
