"""Run SU2_CFD.exe for a single configuration file with retry logic and live output."""
from __future__ import annotations
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)


class SU2ConvergenceError(RuntimeError):
    """Raised when SU2 fails to converge within the allowed iterations."""


class SU2TimeoutError(RuntimeError):
    """Raised when SU2 exceeds the allowed wall-clock time."""


def _stream_output(pipe, label: str, log_file_path: Path) -> None:
    """Read lines from a subprocess pipe, print them live and write to log file."""
    with open(log_file_path, "w") as log_fh:
        for raw in pipe:
            line = raw.rstrip()
            print(f"  [SU2] {line}", flush=True)
            log_fh.write(line + "\n")


def run_su2(
    su2_exe: Path,
    cfg_file: Path,
    work_dir: Path,
    timeout: int = 600,
    max_retries: int = 3,
) -> Path:
    """Execute SU2_CFD.exe with live stdout streaming and return path to history.csv.

    Parameters
    ----------
    su2_exe    : path to SU2_CFD.exe
    cfg_file   : path to the .cfg file
    work_dir   : directory where SU2 writes output files
    timeout    : per-attempt wall-clock limit [seconds]
    max_retries: total attempts before raising

    Returns
    -------
    Path to the history.csv written by SU2.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    history_file = work_dir / "history.csv"
    su2_log      = work_dir / "su2_output.log"

    for attempt in range(1, max_retries + 1):
        log.info(
            "  SU2 attempt %d/%d — %s  (AoA from %s)",
            attempt, max_retries,
            work_dir.name,
            cfg_file.parent.name,
        )
        t0 = time.monotonic()

        proc = subprocess.Popen(
            [str(su2_exe), str(cfg_file)],
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output on a background thread so the main thread can enforce timeout
        stream_thread = threading.Thread(
            target=_stream_output,
            args=(proc.stdout, "SU2", su2_log),
            daemon=True,
        )
        stream_thread.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stream_thread.join(timeout=2)
            elapsed = time.monotonic() - t0
            if attempt == max_retries:
                raise SU2TimeoutError(
                    f"SU2 timed out after {elapsed:.0f}s on {cfg_file.name} "
                    f"(attempt {attempt}/{max_retries})"
                )
            log.warning("  SU2 timeout on attempt %d, retrying...", attempt)
            continue

        stream_thread.join(timeout=5)
        elapsed = time.monotonic() - t0
        log.info("  SU2 finished in %.1f s (exit code %d)", elapsed, proc.returncode)

        if proc.returncode != 0:
            log.warning("  SU2 non-zero exit %d on attempt %d", proc.returncode, attempt)
            if attempt == max_retries:
                raise SU2ConvergenceError(
                    f"SU2 exited with code {proc.returncode} after "
                    f"{max_retries} attempts for {cfg_file.name}"
                )
            continue

        if not history_file.exists():
            log.warning("  history.csv not found after attempt %d", attempt)
            if attempt == max_retries:
                raise SU2ConvergenceError(
                    f"SU2 produced no history.csv for {cfg_file.name}"
                )
            continue

        return history_file

    raise SU2ConvergenceError(f"SU2 failed after {max_retries} attempts.")
