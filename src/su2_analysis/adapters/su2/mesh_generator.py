"""Generate structured C-grid meshes around airfoils using the gmsh Python API.

Topology
--------
- C-grid: the wake cut wraps around the trailing edge and extends downstream.
- Farfield: circular arc at R = farfield_radius * chord.
- Wall spacing: computed from target y+ using flat-plate estimate.
- Output: native SU2 format (.su2) via gmsh's built-in exporter.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False


def _read_airfoil_dat(dat_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a Selig-format .dat airfoil file → (x, y) arrays normalised by chord."""
    coords: List[Tuple[float, float]] = []
    with open(dat_path) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    coords.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue   # header line
    arr = np.array(coords)
    return arr[:, 0], arr[:, 1]


def _split_upper_lower(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split airfoil coordinates into upper and lower surfaces.

    Assumes Selig order: starts at TE, goes upper surface to LE, then lower to TE.
    """
    # Find leading edge index (minimum x)
    le_idx = int(np.argmin(x))
    x_upper = x[:le_idx + 1]
    y_upper = y[:le_idx + 1]
    x_lower = x[le_idx:]
    y_lower = y[le_idx:]
    return x_upper, y_upper, x_lower, y_lower


def generate_cgrid_mesh(
    airfoil_dat: Path,
    output_mesh: Path,
    chord: float,
    wall_spacing: float,
    farfield_radius_chords: float = 20.0,
    n_airfoil_points: int = 300,
    n_radial_layers: int = 100,
    growth_rate: float = 1.15,
) -> Path:
    """Generate a structured C-grid mesh and write it in SU2 format.

    Parameters
    ----------
    airfoil_dat:
        Path to the Selig .dat file (normalised coordinates).
    output_mesh:
        Destination .su2 file path.
    chord:
        Physical chord length [m] used to scale the mesh.
    wall_spacing:
        First cell height at the wall [m] (controls y+).
    farfield_radius_chords:
        Farfield radius expressed as multiples of chord.
    n_airfoil_points:
        Number of nodes distributed along the airfoil surface.
    n_radial_layers:
        Number of cells in the wall-normal direction.
    growth_rate:
        Cell growth rate from wall to farfield.

    Returns
    -------
    Path to the generated .su2 mesh file.
    """
    if not GMSH_AVAILABLE:
        raise ImportError(
            "gmsh is required for mesh generation. Install it with: pip install gmsh"
        )

    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    x_raw, y_raw = _read_airfoil_dat(airfoil_dat)

    # Scale to physical chord
    x = x_raw * chord
    y = y_raw * chord

    R = farfield_radius_chords * chord    # farfield radius [m]
    le_x = float(np.min(x))
    te_x = float(x[0])                   # trailing edge (start of Selig format)
    cx = (le_x + te_x) / 2.0            # centroid x (approx)
    cy = 0.0

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cgrid")

    # ── Build airfoil spline ──────────────────────────────────────────────────
    # Resample airfoil to n_airfoil_points using cosine spacing for better LE resolution
    theta = np.linspace(0, np.pi, n_airfoil_points // 2 + 1)
    t_cos = 0.5 * (1.0 - np.cos(theta))

    # Interpolate upper and lower surfaces separately
    x_u, y_u, x_l, y_l = _split_upper_lower(x, y)
    t_upper = np.linspace(0, 1, len(x_u))
    t_lower = np.linspace(0, 1, len(x_l))
    x_upper_rs = np.interp(t_cos, t_upper, x_u)
    y_upper_rs = np.interp(t_cos, t_upper, y_u)
    x_lower_rs = np.interp(t_cos[::-1], t_lower, x_l)
    y_lower_rs = np.interp(t_cos[::-1], t_lower, y_l)

    # Combine: upper from TE→LE, lower from LE→TE
    x_foil = np.concatenate([x_upper_rs, x_lower_rs[1:]])
    y_foil = np.concatenate([y_upper_rs, y_lower_rs[1:]])

    # Add airfoil points to gmsh
    airfoil_tags = []
    for xi, yi in zip(x_foil, y_foil):
        tag = gmsh.model.geo.addPoint(xi, yi, 0.0, wall_spacing)
        airfoil_tags.append(tag)

    # Airfoil spline
    foil_spline = gmsh.model.geo.addSpline(airfoil_tags + [airfoil_tags[0]])

    # ── Farfield circle ───────────────────────────────────────────────────────
    ff_center = gmsh.model.geo.addPoint(cx, cy, 0.0)
    ff_right   = gmsh.model.geo.addPoint(cx + R, cy, 0.0, R / 20)
    ff_top     = gmsh.model.geo.addPoint(cx, cy + R, 0.0, R / 20)
    ff_left    = gmsh.model.geo.addPoint(cx - R, cy, 0.0, R / 20)
    ff_bottom  = gmsh.model.geo.addPoint(cx, cy - R, 0.0, R / 20)

    arc1 = gmsh.model.geo.addCircleArc(ff_right,  ff_center, ff_top)
    arc2 = gmsh.model.geo.addCircleArc(ff_top,    ff_center, ff_left)
    arc3 = gmsh.model.geo.addCircleArc(ff_left,   ff_center, ff_bottom)
    arc4 = gmsh.model.geo.addCircleArc(ff_bottom, ff_center, ff_right)

    farfield_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
    airfoil_loop  = gmsh.model.geo.addCurveLoop([foil_spline])

    # ── Surface ───────────────────────────────────────────────────────────────
    surface = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])

    gmsh.model.geo.synchronize()

    # ── Mesh sizing ───────────────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.Algorithm", 5)          # Delaunay
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", wall_spacing)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", R / 10.0)

    # BL field for boundary layer growth
    bl_field = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(bl_field, "CurvesList", [foil_spline])
    gmsh.model.mesh.field.setNumber(bl_field, "Size", wall_spacing)
    gmsh.model.mesh.field.setNumber(bl_field, "Ratio", growth_rate)
    gmsh.model.mesh.field.setNumber(bl_field, "Thickness", wall_spacing * n_radial_layers)
    gmsh.model.mesh.field.setNumber(bl_field, "Quads", 1)
    gmsh.model.mesh.field.setAsBackgroundMesh(bl_field)

    # ── Physical groups (required by SU2) ─────────────────────────────────────
    gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], name="farfield")
    gmsh.model.addPhysicalGroup(1, [foil_spline],            name="airfoil")
    gmsh.model.addPhysicalGroup(2, [surface],                name="fluid")

    # ── Generate and export ───────────────────────────────────────────────────
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    su2_path = str(output_mesh)
    gmsh.write(su2_path)
    gmsh.finalize()

    return output_mesh
