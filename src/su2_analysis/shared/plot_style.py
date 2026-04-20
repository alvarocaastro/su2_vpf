"""Paul Tol colorblind-friendly palette and publication figure defaults."""
import matplotlib.pyplot as plt
import matplotlib as mpl

# Paul Tol bright palette
TOLS_BRIGHT = {
    "blue":   "#4477AA",
    "cyan":   "#66CCEE",
    "green":  "#228833",
    "yellow": "#CCBB44",
    "red":    "#EE6677",
    "purple": "#AA3377",
    "grey":   "#BBBBBB",
}

PALETTE = list(TOLS_BRIGHT.values())

CONDITION_COLORS = {
    "takeoff": TOLS_BRIGHT["red"],
    "climb":   TOLS_BRIGHT["yellow"],
    "cruise":  TOLS_BRIGHT["blue"],
    "descent": TOLS_BRIGHT["green"],
}

SECTION_LINESTYLES = {
    "root": "-",
    "mid":  "--",
    "tip":  ":",
}


def apply_style() -> None:
    """Apply global matplotlib style for publication-quality figures."""
    mpl.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       150,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "legend.fontsize":   9,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "lines.linewidth":   1.8,
    })
