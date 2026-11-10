"""RTL profile selection and final PERF window for reusable analysis snapshots."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re


def profile_directories(data):
    """Match existing batch.py layouts, including CI's six-decimal weights."""
    result = {}
    for workload, entry in data.items():
        for point, weight in entry["points"].items():
            identifier = f"{workload}_{point}"
            result[identifier] = {identifier, f"{identifier}_{weight}",
                                  f"{identifier}_{float(weight)}",
                                  f"{identifier}_{float(weight):.6f}"}
    return result


def filter_profile_paths(paths, root, data):
    directories = profile_directories(data)
    selected = []
    for identifier, path in paths:
        directory = Path(path).relative_to(root).parts[0]
        if identifier not in directories:
            continue
        name = directory if directory.startswith(identifier + "_") else identifier
        if name in directories[identifier]:
            selected.append((identifier, path))
    return selected


def selected_logs(report, profile):
    """Select exact profile folders before reading logs in mixed RTL reports."""
    data = json.loads(Path(profile).read_text())
    result = []
    for identifier, directories in profile_directories(data).items():
        existing = [Path(report) / name / "simulator_err.txt" for name in sorted(directories)
                    if (Path(report) / name / "simulator_err.txt").is_file()]
        if len(existing) > 1:
            raise ValueError(f"同一 RTL 切片存在多个目录，不能自动选择：{identifier}")
        if existing:
            result.append((identifier, existing[0]))
    return sorted(result)


PERF = re.compile(r"^\[PERF \]\[time=\s*(\d+)\]\s+(\S+):\s+([^,]+),\s+([-+\d.eE]+)\s*$")


def rtl_window(path):
    """Return original parser input and raw values from its final PERF epoch.

    Retain the full text for xs_get_stats: existing YAML accumulation patterns
    can consume matches from earlier epochs. Raw keys remain epoch-specific.
    """
    perf, commits = [], []
    text = Path(path).read_text(errors="replace")
    for line in text.splitlines():
        match = PERF.match(line.strip())
        if not match:
            continue
        time, hierarchy, counter, number = match.groups()
        perf.append((time, hierarchy, counter, number, line.rstrip()))
        if ".rob" in hierarchy and counter == "commitInstr":
            commits.append((time, hierarchy, line.rstrip()))
    if len(commits) < 2 or len({row[1] for row in commits}) != 1 or commits[-1][0] == commits[-2][0]:
        return None
    epoch = commits[-1][0]
    final = [row for row in perf if row[0] == epoch]
    counts = {}
    values = {}
    for _, hierarchy, counter, number, _ in final:
        key = f"{hierarchy}:{counter}"
        value = float(number)
        counts[key] = counts.get(key, 0) + 1
        # Ambiguous repeated instances are unavailable; no invented summation.
        values[key] = value if counts[key] == 1 and math.isfinite(value) else None
    return text, values
