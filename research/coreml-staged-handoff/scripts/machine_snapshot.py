"""A machine snapshot for the environment qualification (qualification.md); public tools only.

    uv run python research/coreml-staged-handoff/scripts/machine_snapshot.py OUT.json

Records: uptime and load averages, CPU utilisation (top -l 2, second sample), memory (top PhysMem,
memory_pressure -Q free percentage), swap (sysctl vm.swapusage, vm_stat swapins / swapouts /
pageouts counters), thermal and performance warnings (pmset -g therm), power source (pmset -g
batt), Time Machine (tmutil status Running), and the top CPU consumers (ps).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # recorded, never fatal
        return f"error: {e!r}"


def main():
    top = sh("top", "-l", "2", "-n", "0", "-s", "1")
    cpu = re.findall(r"CPU usage: ([\d.]+)% user, ([\d.]+)% sys, ([\d.]+)% idle", top)
    phys = re.findall(r"PhysMem: (.*)", top)
    vm = sh("vm_stat")
    counters = {k: int(v) for k, v in re.findall(r'"?(Swapins|Swapouts|Pageouts)"?:\s+(\d+)', vm)}
    mp = re.search(r"free percentage: (\d+)%", sh("memory_pressure", "-Q"))
    ps = sh("ps", "-Ao", "%cpu=,comm=").splitlines()
    busy = sorted((line.split(None, 1) for line in ps if line.strip()), key=lambda x: -float(x[0]))[:8]
    therm = sh("pmset", "-g", "therm")
    snap = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "uptime": sh("uptime").strip(),
        "cpu_user_sys_idle_pct": [float(x) for x in cpu[-1]] if cpu else None,
        "physmem": phys[-1].strip() if phys else None,
        "memory_free_pct": int(mp.group(1)) if mp else None,
        "swapusage": sh("sysctl", "-n", "vm.swapusage").strip(),
        "vm_counters": counters,
        "thermal_warning": "No thermal warning level has been recorded" not in therm,
        "performance_warning": "No performance warning level has been recorded" not in therm,
        "pmset_therm": therm.strip(),
        "power": sh("pmset", "-g", "batt").splitlines()[0] if sh("pmset", "-g", "batt") else None,
        "time_machine_running": bool(re.search(r"Running = 1", sh("tmutil", "status"))),
        "top_cpu": [(float(c), n.rsplit("/", 1)[-1]) for c, n in busy],
    }
    with open(sys.argv[1], "w") as fh:
        json.dump(snap, fh, indent=1)


if __name__ == "__main__":
    main()
