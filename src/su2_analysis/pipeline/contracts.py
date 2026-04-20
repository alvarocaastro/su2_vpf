"""Typed output contracts for each pipeline stage."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class Stage1Result:
    selected_airfoil: str
    ranking: pd.DataFrame
    polars: Dict[str, pd.DataFrame]          # airfoil_name → polar DataFrame
    output_dir: Path


@dataclass
class Stage2Result:
    selected_airfoil: str
    polars: Dict[str, pd.DataFrame]          # "condition_section" → polar
    pitch_map: pd.DataFrame
    output_dir: Path


@dataclass
class Stage3Result:
    cp_data: Dict[str, pd.DataFrame]         # "condition_section" → Cp vs x/c
    mach_summary: pd.DataFrame
    output_dir: Path


@dataclass
class Stage4Result:
    metrics: pd.DataFrame                    # one row per condition/section
    output_dir: Path


@dataclass
class Stage5Result:
    cascade_table: pd.DataFrame
    rotational_table: pd.DataFrame
    optimal_incidence_table: pd.DataFrame
    pitch_adjustment_table: pd.DataFrame
    blade_twist_table: pd.DataFrame
    off_design_table: pd.DataFrame
    kinematics_table: pd.DataFrame
    stage_loading_ideal: pd.DataFrame
    stage_loading_real: pd.DataFrame
    summary_text: str
    output_dir: Path


@dataclass
class Stage6Result:
    sweep_table: pd.DataFrame
    optimal_table: pd.DataFrame
    kinematics_table: pd.DataFrame
    weight_table: pd.DataFrame
    summary_text: str
    output_dir: Path


@dataclass
class Stage7Result:
    sfc_table: pd.DataFrame
    section_breakdown: pd.DataFrame
    sensitivity_table: pd.DataFrame
    mission_fuel_burn: pd.DataFrame
    summary_text: str
    output_dir: Path
