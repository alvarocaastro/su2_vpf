"""Preview a C-grid mesh in the gmsh GUI.

Usage
-----
    python tools/preview_mesh.py                        # naca_65-410, mid section
    python tools/preview_mesh.py naca_65-310            # specific airfoil
    python tools/preview_mesh.py naca_65-412 tip        # airfoil + blade section
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from su2_analysis.config import AIRFOIL_DIR, CONFIG_FILE
from su2_analysis.config_loader import load_analysis_config
from su2_analysis.shared.atmosphere import (
    blade_velocity, isa_temperature, relative_velocity,
    reynolds_number, speed_of_sound, wall_spacing_for_yplus,
)
from su2_analysis.settings import TARGET_YPLUS

import gmsh


def main() -> None:
    airfoil = sys.argv[1] if len(sys.argv) > 1 else "naca_65-410"
    section = sys.argv[2] if len(sys.argv) > 2 else "mid"

    cfg = load_analysis_config(CONFIG_FILE)
    dat = AIRFOIL_DIR / f"{airfoil}.dat"
    if not dat.exists():
        print(f"ERROR: {dat} not found")
        sys.exit(1)

    # Use cruise condition for preview
    fc      = cfg.flight_conditions["cruise"]
    sec     = cfg.fan_geometry.sections[section]
    chord   = sec.chord
    radius  = sec.radius
    U       = blade_velocity(radius, cfg.fan_geometry.rpm)
    W       = relative_velocity(fc.axial_velocity, U)
    alt     = fc.altitude
    Re      = reynolds_number(W, chord, alt)
    ws      = wall_spacing_for_yplus(TARGET_YPLUS, W, chord, alt)

    print(f"  Airfoil : {airfoil}")
    print(f"  Section : {section}  (chord={chord} m, radius={radius} m)")
    print(f"  Re={Re:.2e}  W={W:.1f} m/s  wall_spacing={ws:.2e} m")
    print(f"  Mesh params: {cfg.su2.mesh_airfoil_points} pts, "
          f"{cfg.su2.mesh_radial_layers} layers, "
          f"farfield={cfg.su2.mesh_farfield_radius}c")
    print("\nGenerating mesh — gmsh GUI will open when ready...")

    # Import mesh generator internals to reuse geometry, but keep GUI open
    from su2_analysis.adapters.su2.mesh_generator import (
        _read_airfoil_dat, _split_upper_lower,
    )
    import numpy as np

    x_raw, y_raw = _read_airfoil_dat(dat)
    x = x_raw * chord
    y = y_raw * chord
    R  = cfg.su2.mesh_farfield_radius * chord
    cx = float((np.min(x) + x[0]) / 2)
    cy = 0.0

    n_pts = cfg.su2.mesh_airfoil_points
    theta = np.linspace(0, np.pi, n_pts // 2 + 1)
    t_cos = 0.5 * (1.0 - np.cos(theta))
    x_u, y_u, x_l, y_l = _split_upper_lower(x, y)
    t_upper = np.linspace(0, 1, len(x_u))
    t_lower = np.linspace(0, 1, len(x_l))
    x_upper_rs = np.interp(t_cos,       t_upper, x_u)
    y_upper_rs = np.interp(t_cos,       t_upper, y_u)
    x_lower_rs = np.interp(t_cos[::-1], t_lower, x_l)
    y_lower_rs = np.interp(t_cos[::-1], t_lower, y_l)
    x_foil = np.concatenate([x_upper_rs, x_lower_rs[1:]])
    y_foil = np.concatenate([y_upper_rs, y_lower_rs[1:]])

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)   # show gmsh log in terminal
    gmsh.model.add("preview")

    airfoil_tags = []
    for xi, yi in zip(x_foil, y_foil):
        airfoil_tags.append(gmsh.model.geo.addPoint(xi, yi, 0.0, ws))
    foil_spline = gmsh.model.geo.addSpline(airfoil_tags + [airfoil_tags[0]])

    ff_center = gmsh.model.geo.addPoint(cx, cy, 0.0)
    ff_right  = gmsh.model.geo.addPoint(cx + R, cy, 0.0, R / 20)
    ff_top    = gmsh.model.geo.addPoint(cx, cy + R, 0.0, R / 20)
    ff_left   = gmsh.model.geo.addPoint(cx - R, cy, 0.0, R / 20)
    ff_bottom = gmsh.model.geo.addPoint(cx, cy - R, 0.0, R / 20)
    arc1 = gmsh.model.geo.addCircleArc(ff_right,  ff_center, ff_top)
    arc2 = gmsh.model.geo.addCircleArc(ff_top,    ff_center, ff_left)
    arc3 = gmsh.model.geo.addCircleArc(ff_left,   ff_center, ff_bottom)
    arc4 = gmsh.model.geo.addCircleArc(ff_bottom, ff_center, ff_right)

    ff_loop   = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
    foil_loop = gmsh.model.geo.addCurveLoop([foil_spline])
    surface   = gmsh.model.geo.addPlaneSurface([ff_loop, foil_loop])

    gmsh.model.geo.synchronize()

    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ws)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", R / 10.0)

    bl = gmsh.model.mesh.field.add("BoundaryLayer")
    gmsh.model.mesh.field.setNumbers(bl, "CurvesList", [foil_spline])
    gmsh.model.mesh.field.setNumber(bl, "Size",      ws)
    gmsh.model.mesh.field.setNumber(bl, "Ratio",     cfg.su2.mesh_growth_rate)
    gmsh.model.mesh.field.setNumber(bl, "Thickness", ws * cfg.su2.mesh_radial_layers)
    gmsh.model.mesh.field.setNumber(bl, "Quads",     1)
    gmsh.model.mesh.field.setAsBackgroundMesh(bl)

    gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4], name="farfield")
    gmsh.model.addPhysicalGroup(1, [foil_spline],            name="airfoil")
    gmsh.model.addPhysicalGroup(2, [surface],                name="fluid")

    gmsh.model.mesh.generate(2)
    print("\nMesh generated — opening GUI (close window to exit)")

    # Color nodes by element quality for easy inspection
    gmsh.option.setNumber("Mesh.ColorCarousel", 2)
    gmsh.option.setNumber("Mesh.SurfaceEdges",  1)
    gmsh.option.setNumber("Mesh.SurfaceFaces",  1)

    gmsh.fltk.run()   # opens interactive GUI — blocks until window is closed
    gmsh.finalize()


if __name__ == "__main__":
    main()
