"""Load and validate YAML configuration files into typed structures."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import yaml


@dataclass
class SectionGeometry:
    radius: float
    chord: float
    solidity: float


@dataclass
class FanGeometry:
    rpm: float
    n_blades: int
    hub_to_tip_ratio: float
    sections: Dict[str, SectionGeometry]


@dataclass
class SU2Config:
    executable: Path
    max_iter: int
    convergence_residual: float
    turbulence_model: str
    cfl_number: float
    mesh_farfield_radius: float
    mesh_airfoil_points: int
    mesh_radial_layers: int
    mesh_growth_rate: float
    target_yplus: float
    timeout_seconds: int
    max_retries: int


@dataclass
class FlightCondition:
    mach_relative: float
    axial_velocity: float
    altitude: float
    ncrit: float


@dataclass
class AlphaSweep:
    start: float
    end: float
    step: float

    def to_list(self) -> List[float]:
        import numpy as np
        return list(np.arange(self.start, self.end + self.step * 0.5, self.step))


@dataclass
class AtmosphereConfig:
    sea_level_temperature: float
    sea_level_pressure: float
    lapse_rate: float
    gamma: float
    R_air: float
    mu_ref: float
    T_ref: float
    S_suth: float


@dataclass
class AnalysisConfig:
    fan_geometry: FanGeometry
    su2: SU2Config
    reference_condition: dict
    flight_conditions: Dict[str, FlightCondition]
    alpha_sweep_simulations: AlphaSweep
    airfoil_candidates: List[str]
    atmosphere: AtmosphereConfig


@dataclass
class MissionPhase:
    duration_min: float
    thrust_fraction: float


@dataclass
class ReverseThrustConfig:
    rpm_fraction: float
    axial_velocity: float
    delta_beta_sweep_start: float
    delta_beta_sweep_end: float
    delta_beta_sweep_points: int
    min_stall_margin_deg: float


@dataclass
class MechanismWeightConfig:
    vpf_actuator_fraction: float
    cascade_reverser_fraction: float


@dataclass
class EngineParameters:
    name: str
    baseline_sfc: float
    fan_efficiency: float
    bypass_ratio: float
    design_thrust_kN: float
    dry_weight_kg: float
    fan_diameter_m: float
    tau: float
    mission: Dict[str, MissionPhase]
    reverse_thrust: ReverseThrustConfig
    mechanism_weight: MechanismWeightConfig


def load_analysis_config(path: Path) -> AnalysisConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    geom_raw = raw["fan_geometry"]
    sections = {
        name: SectionGeometry(**vals)
        for name, vals in geom_raw["sections"].items()
    }
    fan_geom = FanGeometry(
        rpm=geom_raw["rpm"],
        n_blades=geom_raw["n_blades"],
        hub_to_tip_ratio=geom_raw["hub_to_tip_ratio"],
        sections=sections,
    )

    su2_raw = raw["su2"]
    su2_cfg = SU2Config(
        executable=Path(su2_raw["executable"]),
        max_iter=su2_raw["max_iter"],
        convergence_residual=su2_raw["convergence_residual"],
        turbulence_model=su2_raw["turbulence_model"],
        cfl_number=su2_raw["cfl_number"],
        mesh_farfield_radius=su2_raw["mesh_farfield_radius"],
        mesh_airfoil_points=su2_raw["mesh_airfoil_points"],
        mesh_radial_layers=su2_raw["mesh_radial_layers"],
        mesh_growth_rate=su2_raw["mesh_growth_rate"],
        target_yplus=su2_raw["target_yplus"],
        timeout_seconds=su2_raw["timeout_seconds"],
        max_retries=su2_raw["max_retries"],
    )

    flight_conditions = {
        name: FlightCondition(**vals)
        for name, vals in raw["flight_conditions"].items()
    }

    atm_raw = raw["atmosphere"]
    atm = AtmosphereConfig(**atm_raw)

    alpha_sweep = AlphaSweep(**raw["alpha_sweep_simulations"])

    return AnalysisConfig(
        fan_geometry=fan_geom,
        su2=su2_cfg,
        reference_condition=raw["reference_condition"],
        flight_conditions=flight_conditions,
        alpha_sweep_simulations=alpha_sweep,
        airfoil_candidates=raw["airfoil_candidates"],
        atmosphere=atm,
    )


def load_engine_parameters(path: Path) -> EngineParameters:
    with open(path) as f:
        raw = yaml.safe_load(f)

    eng = raw["engine"]
    mission = {
        name: MissionPhase(**vals)
        for name, vals in raw["mission"]["phases"].items()
    }
    rt = ReverseThrustConfig(**raw["reverse_thrust"])
    mw = MechanismWeightConfig(**raw["mechanism_weight"])

    return EngineParameters(
        name=eng["name"],
        baseline_sfc=eng["baseline_sfc"],
        fan_efficiency=eng["fan_efficiency"],
        bypass_ratio=eng["bypass_ratio"],
        design_thrust_kN=eng["design_thrust_kN"],
        dry_weight_kg=eng["dry_weight_kg"],
        fan_diameter_m=eng["fan_diameter_m"],
        tau=raw["efficiency_transfer"]["tau"],
        mission=mission,
        reverse_thrust=rt,
        mechanism_weight=mw,
    )
