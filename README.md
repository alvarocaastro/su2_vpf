# su2_vpf

Aerodynamic analysis pipeline for a **Variable Pitch Fan (VPF)** applied to a GE9X-class turbofan engine, using **SU2 8.4.0 compressible RANS** as the CFD solver.

Unlike XFOIL-based approaches, SU2 solves the compressible Navier-Stokes equations natively — no Prandtl-Glauert or Kármán-Tsien corrections needed. Transonic effects at M=0.85–0.93 are captured directly by the solver.

---

## Pipeline

| Stage | Description | Output |
|-------|-------------|--------|
| 1 | Airfoil selection via SU2 RANS polars | ranking.csv, polar_comparison.png |
| 2 | Compressible RANS simulations (4 conditions × 3 sections) | 12 polars, pitch_map.csv |
| 3 | CFD post-processing: Cp distributions, Mach contours, shock detection | cp_*.png, mach_summary.csv |
| 4 | Performance metrics: CL/CD_max, stall margin, VPF efficiency gain | metrics_summary.csv, heatmaps |
| 5 | Pitch kinematics: cascade (Weinig+Carter), rotation (Snel), blade twist, stage loading | 10 CSV + 20 figures |
| 6 | Reverse thrust via negative pitch — no cascade doors needed | sweep + weight comparison |
| 7 | SFC reduction and mission fuel burn integration | ΔSFC per phase, fuel saving |

## Requirements

```
pip install gmsh numpy pandas matplotlib pyyaml scipy
```

SU2 8.4.0 executable must be set in `config/analysis_config.yaml`:

```yaml
su2:
  executable: /path/to/SU2_CFD.exe
```

## Usage

```bash
# Full pipeline
python run_analysis.py

# Selected stages only
python run_analysis.py --stages 1 2

# Resume from a given stage
python run_analysis.py --from-stage 4
```

## Tests

```bash
pip install pytest
pytest
```

## Structure

```
su2_vpf/
├── config/          # YAML configuration (fan geometry, flight conditions, engine)
├── data/airfoils/   # NACA .dat profiles
├── src/su2_analysis/
│   ├── adapters/su2/    # gmsh mesh generator, SU2 config writer, runner, parser
│   ├── shared/          # ISA atmosphere, Paul Tol plot style
│   ├── stage1_*/  …  stage7_*/
├── results/         # Generated outputs (gitignored, structure preserved)
├── tests/
└── run_analysis.py
```

## Reference Engine

GE9X-class · 3.40 m fan · 2200 RPM · 16 blades · BPR 10 · baseline SFC 0.50 lb/(lbf·h)
