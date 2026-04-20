"""Run SU2_CFD.exe for a single configuration file with retry logic."""
from __future__ import annotations
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class SU2ConvergenceError(RuntimeError):
    """Raised when SU2 fails to converge within the allowed iterations."""


class SU2TimeoutError(RuntimeError):
    """Raised when SU2 exceeds the allowed wall-clock time."""


def run_su2(
    su2_exe: Path,
    cfg_file: Path,
    work_dir: Path,
    timeout: int = 600,
    max_retries: int = 3,
) -> Path:
    """Execute SU2_CFD.exe and return the path to history.csv.

    Parameters
    ----------
    su2_exe    : path to SU2_CFD.exe
    cfg_file   : path to the .cfg file (must be in work_dir or use absolute path)
    work_dir   : directory where SU2 will write output files
    timeout    : per-attempt wall-clock limit [seconds]
    max_retries: total attempts before raising

    Returns
    -------
    Path to the history.csv written by SU2.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    history_file = work_dir / "history.csv"

    for attempt in range(1, max_retries + 1):
        log.debug("SU2 attempt %d/%d: %s", attempt, max_retries, cfg_file.name)
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                [str(su2_exe), str(cfg_file)],
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            if attempt == max_retries:
                raise SU2TimeoutError(
                    f"SU2 timed out after {timeout}s on {cfg_file.name} "
                    f"(attempt {attempt}/{max_retries})"
                )
            log.warning("SU2 timeout on attempt %d, retrying...", attempt)
            continue

        elapsed = time.monotonic() - t0
        log.debug("SU2 finished in %.1f s (exit %d)", elapsed, result.returncode)

        if result.returncode != 0:
            log.warning(
                "SU2 non-zero exit %d on attempt %d:\n%s",
                result.returncode, attempt, result.stderr[-2000:],
            )
            if attempt == max_retries:
                raise SU2ConvergenceError(
                    f"SU2 exited with code {result.returncode} after "
                    f"{max_retries} attempts for {cfg_file.name}"
                )
            continue

        if not history_file.exists():
            log.warning("history.csv not found after attempt %d", attempt)
            if attempt == max_retries:
                raise SU2ConvergenceError(
                    f"SU2 produced no history.csv for {cfg_file.name}"
                )
            continue

        return history_file

    raise SU2ConvergenceError(f"SU2 failed after {max_retries} attempts.")
