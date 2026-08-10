#!/usr/bin/env python
"""Shared helper for the benchmark scripts in this directory (rne_speed.py,
ik_speed.py) -- factored out once a second script needed the same thing.
"""

import os
import platform
import subprocess


def cpu_info() -> str:
    """Best-effort, portable one-line CPU description (name + core count,
    clock speed when the OS actually exposes one). No new hard dependency:
    psutil is used for clock speed only if already installed, and skipped
    otherwise -- some platforms (e.g. Apple Silicon) don't expose a single
    meaningful clock speed at all, so a missing/nonsensical reading is
    silently omitted rather than shown.
    """
    system = platform.system()
    name = None

    if system == "Darwin":
        try:
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    elif system == "Windows":
        name = platform.processor() or None

    if not name:
        name = platform.processor() or platform.machine() or "unknown CPU"

    cores = os.cpu_count() or "?"
    info = f"{name} ({cores} cores)"

    try:
        import psutil

        freq = psutil.cpu_freq()
        # Sanity floor: real clock speeds are in the hundreds-to-thousands
        # of MHz; some platforms (Apple Silicon via psutil) report bogus
        # single-digit values instead of raising.
        if freq and freq.max and freq.max > 100:
            info += f", {freq.max:.0f} MHz"
    except Exception:
        pass

    return info
