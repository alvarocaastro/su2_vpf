"""Central registry of physics constants and tunable parameters."""

# ── Rotational corrections (Snel 1994) ───────────────────────────────────────
SNEL_COEFFICIENT: float = 3.0          # ΔCL_rot = a*(c/r)²*CL_2D

# ── Cascade corrections ───────────────────────────────────────────────────────
CARTER_M_COEFFICIENT: float = 0.23     # Carter deviation: δ = m*θ/√σ

# ── 2-D → 3-D efficiency transfer ────────────────────────────────────────────
TAU_TRANSFER: float = 0.50             # damping factor for fan efficiency gain

# ── Korn wave-drag onset (for Stage 3 reference line) ────────────────────────
KORN_KAPPA_AIRFOIL: float = 0.87       # technology factor for supercritical foils

# ── SU2 run quality thresholds ────────────────────────────────────────────────
CL_MAX_MIN_ACCEPTABLE: float = 0.3
CD_MIN_MAX_ACCEPTABLE: float = 0.05
CD_MIN_MIN_ACCEPTABLE: float = 1e-5

# ── Mesh wall-spacing (y+ target) ─────────────────────────────────────────────
# y+ = 1 required for wall-resolved SST (no wall functions); flat-plate estimate
TARGET_YPLUS: float = 1.0

# ── Turbulent inlet conditions (fan face) ─────────────────────────────────────
# Fan rotor inlet: ~5% Tu from upstream OGV wakes and inlet distortion
FREESTREAM_TURBULENCE_INTENSITY: float = 0.05
# mu_t/mu_inf: sets turbulence length scale at far-field boundary
FREESTREAM_TURB_RATIO: float = 10.0

# ── ISA atmosphere ────────────────────────────────────────────────────────────
GAMMA: float = 1.4
R_AIR: float = 287.058              # J/(kg·K)
T0_ISA: float = 288.15              # K
P0_ISA: float = 101325.0            # Pa
LAPSE_RATE: float = 0.0065          # K/m
MU_REF: float = 1.716e-5            # Pa·s
T_MU_REF: float = 273.15            # K
S_SUTHERLAND: float = 110.4         # K

# ── Dixon & Hall design zone (stage loading) ──────────────────────────────────
PHI_MIN: float = 0.35
PHI_MAX: float = 0.55
PSI_MIN: float = 0.25
PSI_MAX: float = 0.50

# ── Figure style ──────────────────────────────────────────────────────────────
DPI: int = 150
FIGURE_FORMAT: str = "png"
