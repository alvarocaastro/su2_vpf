"""Run SU2_CFD.exe with live output parsing and restart-chained polar sweeps."""
from __future__ import annotations
import logging
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import List

from su2_analysis.shared.progress import step, ok, warn, info, progress_bar

log = logging.getLogger(__name__)


class SU2ConvergenceError(RuntimeError):
    """Raised when SU2 fails to converge within the allowed iterations."""


class SU2TimeoutError(RuntimeError):
    """Raised when SU2 exceeds the allowed wall-clock time."""


# ── SU2 output parser ─────────────────────────────────────────────────────────

# Matches the SU2 screen history line, e.g.:
#    500  -6.123e+00  -7.456e+00   8.512e-01   1.384e-02
_ITER_RE = re.compile(
    r"^\s*(\d+)"                          # iteration
    r"\s+([-+]?\d+\.\d+[eE][+-]\d+)"     # Rho residual
    r"\s+([-+]?\d+\.\d+[eE][+-]\d+)"     # RhoE residual
    r"\s+([-+]?\d+\.\d+[eE][+-]\d+)"     # CL
    r"\s+([-+]?\d+\.\d+[eE][+-]\d+)"     # CD
)

# How often to print a solver progress line (every N iterations)
_PRINT_EVERY = 100


class _SU2StreamParser:
    """Thread-safe parser that reads SU2 stdout, prints key lines, logs to file."""

    def __init__(self, pipe, log_path: Path, max_iter: int, alpha: float) -> None:
        self._pipe     = pipe
        self._log_path = log_path
        self._max_iter = max_iter
        self._alpha    = alpha
        self._last_iter = 0
        self._last_cl   = float("nan")
        self._last_cd   = float("nan")
        self._last_res  = float("nan")

    def run(self) -> None:
        printed_header = False
        with open(self._log_path, "w") as fh:
            for raw in self._pipe:
                line = raw.rstrip()
                fh.write(line + "\n")

                # Print SU2 startup / version header once
                if not printed_header and ("SU2" in line or "Harrier" in line
                                           or "Version" in line):
                    print(f"    \033[90m{line}\033[0m", flush=True)
                    if "Harrier" in line:
                        printed_header = True
                    continue

                # Parse solver iteration lines
                m = _ITER_RE.match(line)
                if m:
                    it  = int(m.group(1))
                    res = float(m.group(2))
                    cl  = float(m.group(4))
                    cd  = float(m.group(5))
                    self._last_iter = it
                    self._last_cl   = cl
                    self._last_cd   = cd
                    self._last_res  = res

                    if it % _PRINT_EVERY == 0 or it == 1:
                        progress_bar(
                            min(it, self._max_iter), self._max_iter,
                            label=(f"α={self._alpha:+.1f}°  "
                                   f"Res={res:.2e}  CL={cl:.4f}  CD={cd:.5f}"),
                        )
                    continue

                # Print convergence / warning messages from SU2
                low = line.lower()
                if any(k in low for k in ("converged", "warning", "error",
                                          "diverged", "maximum", "cauchy")):
                    print(f"    \033[93m[SU2] {line}\033[0m", flush=True)

    @property
    def last_iter(self) -> int: return self._last_iter

    @property
    def converged_cl(self) -> float: return self._last_cl

    @property
    def converged_cd(self) -> float: return self._last_cd


# ── Low-level single run ──────────────────────────────────────────────────────

def run_su2(
    su2_exe: Path,
    cfg_file: Path,
    work_dir: Path,
    timeout: int = 300,
    max_retries: int = 2,
    max_iter: int = 2000,
    alpha: float = 0.0,
) -> Path:
    """Execute SU2_CFD.exe once and return the path to history.csv.

    Live solver output is parsed: iteration progress bar is shown in the
    terminal and full output is saved to su2_output.log inside work_dir.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    history_file = work_dir / "history.csv"
    su2_log      = work_dir / "su2_output.log"

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            warn(f"Retry {attempt}/{max_retries} — α={alpha:+.1f}°")

        proc = subprocess.Popen(
            [str(su2_exe), str(cfg_file)],
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        parser  = _SU2StreamParser(proc.stdout, su2_log, max_iter, alpha)
        streamer = threading.Thread(target=parser.run, daemon=True)
        streamer.start()

        t0 = time.monotonic()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            streamer.join(timeout=2)
            if attempt == max_retries:
                raise SU2TimeoutError(
                    f"SU2 timed out after {timeout}s for α={alpha:+.1f}°"
                )
            warn(f"Timeout at {timeout}s — retrying")
            continue

        streamer.join(timeout=5)
        elapsed = time.monotonic() - t0

        if proc.returncode != 0:
            warn(f"SU2 exit code {proc.returncode} (α={alpha:+.1f}°, "
                 f"attempt {attempt}/{max_retries})")
            if attempt == max_retries:
                raise SU2ConvergenceError(
                    f"SU2 exited {proc.returncode} after {max_retries} "
                    f"attempts for {cfg_file.name}"
                )
            continue

        if not history_file.exists():
            warn(f"history.csv missing after α={alpha:+.1f}°")
            if attempt == max_retries:
                raise SU2ConvergenceError(
                    f"No history.csv for {cfg_file.name}"
                )
            continue

        ok(f"α={alpha:+.1f}°  converged in {elapsed:.1f}s  "
           f"iter={parser.last_iter}  "
           f"CL={parser.converged_cl:.4f}  CD={parser.converged_cd:.5f}")
        return history_file

    raise SU2ConvergenceError(f"SU2 failed after {max_retries} attempts.")


# ── Restart-chained polar sweep ───────────────────────────────────────────────

_SOLUTION_CSV = "solution_flow.csv"
_RESTART_CSV  = "restart_flow.csv"


def run_polar_sweep(
    su2_exe: Path,
    mesh_file: Path,
    base_cfg_file: Path,
    alpha_list: List[float],
    sweep_dir: Path,
    timeout_per_alpha: int = 300,
    max_retries: int = 2,
) -> dict[float, Path]:
    """Restart-chained AoA polar sweep.

    Each AoA after the first reuses the previous converged solution as the
    initial condition, cutting iterations to convergence by 5-10×.

    Returns
    -------
    Mapping alpha → Path(history.csv) for every successfully converged point.
    """
    sweep_dir.mkdir(parents=True, exist_ok=True)
    alphas = sorted(alpha_list)

    # Read iteration count from base config to pass to the progress bar
    base_text = base_cfg_file.read_text()
    max_iter  = _extract_max_iter(base_text)

    results: dict[float, Path] = {}
    restart_source: Path | None = None

    total = len(alphas)
    info(f"Polar sweep: {total} AoA points  "
         f"[{alphas[0]:+.1f}° → {alphas[-1]:+.1f}°]  "
         f"restart-chained after first cold run")

    for i, alpha in enumerate(alphas):
        aoa_dir = sweep_dir / f"aoa_{alpha:+07.2f}"
        aoa_dir.mkdir(exist_ok=True)

        is_restart = restart_source is not None
        cfg_path   = aoa_dir / "config.cfg"
        _write_aoa_cfg(base_text, cfg_path, alpha, restart=is_restart)

        if is_restart:
            dest = aoa_dir / _RESTART_CSV
            shutil.copy2(restart_source, dest)
            mode_label = "warm-start"
        else:
            mode_label = "cold-start"

        print(f"\n  \033[96m┌─ AoA {i+1}/{total}: α={alpha:+.1f}°  "
              f"[{mode_label}]\033[0m", flush=True)

        history_file = aoa_dir / "history.csv"
        try:
            run_su2(
                su2_exe=su2_exe,
                cfg_file=cfg_path,
                work_dir=aoa_dir,
                timeout=timeout_per_alpha,
                max_retries=max_retries,
                max_iter=max_iter,
                alpha=alpha,
            )
            results[alpha] = history_file

            solution = aoa_dir / _SOLUTION_CSV
            if solution.exists():
                restart_source = solution
            else:
                warn(f"solution_flow.csv missing — next AoA will cold-start")
                restart_source = None

        except (SU2ConvergenceError, SU2TimeoutError) as exc:
            warn(f"α={alpha:+.1f}° failed: {exc}")
            warn("Restart chain broken — next AoA will cold-start")
            restart_source = None

    n_ok = len(results)
    ok(f"Polar sweep complete: {n_ok}/{total} points converged")
    return results


def _write_aoa_cfg(base_text: str, dest: Path, alpha: float, restart: bool) -> None:
    """Write per-AoA config by patching AOA= and RESTART_SOL= lines."""
    lines = []
    for line in base_text.splitlines():
        s = line.strip()
        if s.startswith("AOA="):
            lines.append(f"AOA= {alpha}")
        elif s.startswith("RESTART_SOL="):
            lines.append("RESTART_SOL= YES" if restart else "RESTART_SOL= NO")
        else:
            lines.append(line)
    dest.write_text("\n".join(lines))


def _extract_max_iter(cfg_text: str) -> int:
    """Extract ITER value from config text."""
    for line in cfg_text.splitlines():
        if line.strip().startswith("ITER="):
            try:
                return int(line.split("=")[1].strip())
            except (ValueError, IndexError):
                pass
    return 2000
