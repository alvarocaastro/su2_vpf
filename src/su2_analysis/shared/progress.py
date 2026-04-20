"""Console progress utilities — banners, step counters, timing."""
from __future__ import annotations
import logging
import time
from contextlib import contextmanager
from typing import Generator

log = logging.getLogger(__name__)

# ANSI colours (safe on Windows 10+ with ENABLE_VIRTUAL_TERMINAL_PROCESSING)
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_GREY   = "\033[90m"
_BLUE   = "\033[94m"
_MAGENTA= "\033[95m"

_W = 70   # banner width


def _enable_ansi_windows() -> None:
    """Enable ANSI escape codes on Windows."""
    import sys, os
    if sys.platform == "win32":
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.SetConsoleMode(kernel.GetStdHandle(-11), 7)
        except Exception:
            pass

_enable_ansi_windows()


# ── Public helpers ─────────────────────────────────────────────────────────────

def banner(title: str, char: str = "═") -> None:
    """Print a full-width banner."""
    line = char * _W
    print(f"\n{_BOLD}{_CYAN}{line}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{line}{_RESET}")


def section(title: str) -> None:
    """Print a section divider."""
    pad = _W - len(title) - 4
    print(f"\n{_BLUE}── {title} {'─' * max(pad, 2)}{_RESET}")


def step(msg: str) -> None:
    """Print an in-progress step."""
    print(f"  {_YELLOW}▶{_RESET}  {msg}")


def ok(msg: str) -> None:
    """Print a success line."""
    print(f"  {_GREEN}✔{_RESET}  {msg}")


def warn(msg: str) -> None:
    """Print a warning line."""
    print(f"  {_RED}⚠{_RESET}  {msg}")


def info(msg: str) -> None:
    """Print a neutral info line."""
    print(f"  {_GREY}·{_RESET}  {msg}")


def stage_banner(num: int, name: str) -> None:
    """Print a stage header."""
    title = f"STAGE {num}  ─  {name.upper()}"
    print(f"\n{_BOLD}{_MAGENTA}{'━' * _W}")
    print(f"  {title}")
    print(f"{'━' * _W}{_RESET}")


def stage_done(num: int, name: str, elapsed: float) -> None:
    """Print a stage completion footer."""
    print(f"{_GREEN}{'─' * _W}")
    print(f"  Stage {num} ({name}) completed in {elapsed:.1f} s")
    print(f"{'─' * _W}{_RESET}\n")


@contextmanager
def timed_step(label: str) -> Generator[None, None, None]:
    """Context manager that prints label, runs block, then prints elapsed time."""
    step(label)
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        ok(f"{label} — done in {elapsed:.1f} s")


def progress_bar(current: int, total: int, label: str = "", width: int = 30) -> None:
    """Print an inline ASCII progress bar (overwrites current line)."""
    filled = int(width * current / total) if total else 0
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / total if total else 0
    suffix = f"  {label}" if label else ""
    print(f"\r  [{_GREEN}{bar}{_RESET}] {pct:5.1f}%  {current}/{total}{suffix}",
          end="", flush=True)
    if current >= total:
        print()   # newline at 100%
