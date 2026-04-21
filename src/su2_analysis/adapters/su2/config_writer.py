"""Write SU2 .cfg configuration files for RANS turbofan blade polar runs."""
from __future__ import annotations
from pathlib import Path


# Solution/restart use CSV format so restart chaining works without binary I/O
_CFG_TEMPLATE = """\
% ─────────────────────────────────────────────────────────────────
%  SU2 Configuration – Turbofan Blade RANS Polar (auto-generated)
% ─────────────────────────────────────────────────────────────────

% Problem definition
SOLVER= RANS
KIND_TURB_MODEL= {turb_model}
{sst_options}\
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
READ_BINARY_RESTART= NO

% Compressible flow — relative frame quantities for turbofan blade section
MACH_NUMBER= {mach}
AOA= {aoa}
SIDESLIP_ANGLE= 0.0
FREESTREAM_OPTION= TEMPERATURE_FS
FREESTREAM_TEMPERATURE= {T_inf}
REYNOLDS_NUMBER= {reynolds}
REYNOLDS_LENGTH= {chord}
% Turbulent inlet conditions (fan face: Tu~5%, short mixing length)
FREESTREAM_TURBULENCEINTENSITY= {tu_intensity}
FREESTREAM_TURB2FREESTREAM_RATIO= {turb_ratio}
REF_ORIGIN_MOMENT_X= 0.25
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH= {chord}
REF_AREA= {chord}

% Boundary conditions
% MARKER_HEATFLUX: adiabatic no-slip wall (correct for viscous RANS)
MARKER_HEATFLUX= ( airfoil, 0.0 )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( airfoil )
MARKER_MONITORING= ( airfoil )

% Multigrid
MGLEVEL= 3
MGCYCLE= W_CYCLE
MG_PRE_SMOOTH= ( 1, 2, 3, 3 )
MG_POST_SMOOTH= ( 0, 0, 0, 0 )
MG_CORRECTION_SMOOTH= ( 0, 0, 0, 0 )
MG_DAMP_RESTRICTION= 0.75
MG_DAMP_PROLONGATION= 0.75

% Numerical scheme
NUM_METHOD_GRAD= GREEN_GAUSS
CFL_NUMBER= {cfl}
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.5, 1.5, 1.0, {cfl_max} )
MAX_DELTA_TIME= 1E6

% Convective scheme — ROE+MUSCL handles transonic blade passages well
CONV_NUM_METHOD_FLOW= ROE
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN
VENKAT_LIMITER_COEFF= 0.03

% Turbulence transport
CONV_NUM_METHOD_TURB= SCALAR_UPWIND
MUSCL_TURB= NO

% Convergence — residual drop OR Cauchy plateau (whichever triggers first)
% Monitor both flow and turbulence equations
CONV_FIELD= {conv_field}
CONV_RESIDUAL_MINVAL= {conv_residual}
CONV_STARTITER= 25
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1E-5

% Iteration
ITER= {max_iter}
OUTPUT_WRT_FREQ= {max_iter}

% Input / output
MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2
SOLUTION_FILENAME= solution_flow
RESTART_FILENAME= restart_flow
OUTPUT_FILES= ( RESTART_ASCII, CSV )
VOLUME_FILENAME= volume_flow
SURFACE_FILENAME= surface_flow
% TOTAL_PRESSURE_LOSS: key turbomachinery efficiency metric (total-to-total)
HISTORY_OUTPUT= ( ITER, RMS_RHO, RMS_RHO_E, LIFT, DRAG, MOMENT_Z, TOTAL_PRESSURE_LOSS )
WRT_ZONE_HIST= NO
SCREEN_OUTPUT= ( INNER_ITER, RMS_DENSITY, LIFT, DRAG, TOTAL_PRESSURE_LOSS )
"""

# SST variant: V2003m is the Menter 2003 correction with improved separation prediction
_SST_OPTIONS_LINE = "SST_OPTIONS= V2003m\n"


def write_su2_config(
    output_path: Path,
    mesh_file: Path,
    mach: float,
    aoa: float,
    reynolds: float,
    chord: float,
    T_inf: float,
    max_iter: int = 2000,
    conv_residual: float = -7.0,
    cfl: float = 5.0,
    turb_model: str = "SST",
    tu_intensity: float = 0.05,
    turb_ratio: float = 10.0,
) -> Path:
    """Write a SU2 .cfg file for a turbofan blade RANS polar run.

    Parameters
    ----------
    output_path   : destination .cfg file
    mesh_file     : absolute path to the .su2 mesh
    mach          : relative Mach number seen by the blade section
    aoa           : incidence angle [degrees]
    reynolds      : chord-based Reynolds number
    chord         : reference chord [m]
    T_inf         : freestream static temperature [K]
    max_iter      : max solver iterations
    conv_residual : log10 residual drop target
    cfl           : initial CFL number (adaptive CFL is enabled)
    turb_model    : SU2 turbulence model keyword (SST recommended for blades)
    tu_intensity  : freestream turbulence intensity (0–1); fan face ~0.05
    turb_ratio    : freestream mu_t/mu ratio; controls turbulence length scale
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_sst = turb_model.upper().startswith("SST")
    # SST monitors TKE; SA monitors the modified viscosity
    conv_field   = "RMS_DENSITY, RMS_TKE" if is_sst else "RMS_DENSITY, RMS_NU_TILDE"
    sst_options  = _SST_OPTIONS_LINE if is_sst else ""

    content = _CFG_TEMPLATE.format(
        turb_model=turb_model,
        sst_options=sst_options,
        mach=mach,
        aoa=aoa,
        T_inf=T_inf,
        reynolds=reynolds,
        chord=chord,
        tu_intensity=tu_intensity,
        turb_ratio=turb_ratio,
        cfl=cfl,
        cfl_max=cfl * 20,
        conv_residual=conv_residual,
        conv_field=conv_field,
        max_iter=max_iter,
        mesh_file=str(mesh_file).replace("\\", "/"),
    )
    output_path.write_text(content)
    return output_path
