"""Prepare reusable counter snapshots and query weighted metrics.

Numerical extraction and weighting reuse batch.py and compute_weighted.
The JSON CLI deliberately has no knowledge of HTTP, GitHub, or cache identities.
"""
from __future__ import annotations

import argparse
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import tempfile
import time
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def frame_rows(frame):
    return json.loads(frame.reset_index(names="name").to_json(orient="records", double_precision=15))


def command(proc: Path, args: list[str]):
    subprocess.run([sys.executable, *args], cwd=proc, check=True)


def weight(proc: Path, raw: Path, profile: Path, out: Path, spec: str, score=False):
    args = ["-m", "simpoint_cpt.compute_weighted", "-r", str(raw), "-j", str(profile),
            "--out-dir", str(out.parent), "-v", spec, "--score" if score else "-o", str(out)]
    command(proc, args)


def prepare(request):
    # batch.py reads local parser input rather than re-reading NFS.
    if request.get("stage_dir"):
        _prepare(request)
    else:
        with tempfile.TemporaryDirectory(prefix="gem5-analysis-stats-") as temporary:
            _prepare({**request, "stage_dir": temporary})


def report_progress(dest, stage, completed=0, total=0):
    payload = {"stage": stage, "completed": completed, "total": total}
    temporary = dest / "progress.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False))
    temporary.replace(dest / "progress.json")


def _prepare(request):
    import pandas as pd
    import utils as u
    from utils import common
    # Keep the existing parser, using larger buffered reads for NFS.
    common.reverse_readline = partial(common.reverse_readline, buf_size=1024 * 1024)
    from utils.targets_yaml import list_groups, load_groups

    proc = Path(__file__).resolve().parent
    source = Path(request["source"])
    dest = Path(request["output"])
    dest.mkdir(parents=True, exist_ok=True)
    backend = request.get("backend", "gem5")
    stats_root = source / "spec_all" if backend == "gem5" else source
    profile = Path(request.get("profile_path", source / "cluster.json"))
    profile_data = json.loads(profile.read_text())
    stage = Path(request["stage_dir"])
    stage.mkdir(exist_ok=True)
    raw_dir = dest / "raw"
    raw_dir.mkdir(exist_ok=True)
    points, excluded, raw_names = [], [], set()
    print("收集已完成切片与原始计数器", flush=True)
    if backend == "xs":
        from analysis_inputs import selected_logs, rtl_window
        paths = selected_logs(source, profile)
    else:
        paths = u.glob_stats(str(stats_root), fname="stats.txt")
    report_progress(dest, "并行读取原始计数器", 0, len(paths))

    def collect(item):
        identifier, stat_path = item
        path = Path(stat_path)
        workload, point = identifier.rsplit("_", 1)
        point_root = path.parent.parent if path.parent.name == "m5out" else path.parent
        reason = None
        if workload not in profile_data or point not in profile_data[workload]["points"]:
            reason = "outside_profile"
        elif backend == "gem5" and (point_root / "abort").exists():
            reason = "aborted"
        elif backend == "gem5" and not (point_root / "completed").exists():
            reason = "not_completed"
        if reason:
            return None, {"name": identifier, "reason": reason}, set()
        if backend == "xs":
            out = path.with_name("simulator_out.txt")
            outcome = out.read_text(errors="replace") if out.is_file() else ""
            if not any(token in outcome for token in ("EXCEEDING CYCLE/INSTR LIMIT", "HIT GOOD TRAP")) or any(token in outcome for token in ("HIT BAD TRAP", "Assertion", "ABORT")):
                return None, {"name": identifier, "reason": "rtl_not_completed"}, set()
            window = rtl_window(path)
            if window is None:
                return None, {"name": identifier, "reason": "rtl_missing_warmup_or_final_window"}, set()
            text, values = window
        else:
            lines, blocks = common.get_last_complete_stats_dump(str(path))
            if blocks < 2 or not any("End Simulation Statistics" in line for line in lines):
                return None, {"name": identifier, "reason": "incomplete_stats"}, set()
            values = {}
            for line in lines:
                parts = line.split()
                if len(parts) < 2 or parts[0].startswith(("#", "-")):
                    continue
                try:
                    value = float(parts[1])
                except ValueError:
                    continue
                values.setdefault(parts[0], value if math.isfinite(value) else None)
        with gzip.open(raw_dir / f"{identifier}.json.gz", "wt", encoding="utf-8", compresslevel=1) as stream:
            stream.write(json.dumps(values, allow_nan=False))
        local = stage / identifier / "m5out" if backend == "gem5" else stage / identifier
        local.mkdir(parents=True, exist_ok=True)
        if backend == "xs":
            # Keep original RTL parser semantics, including multi-match accumulation.
            (local / "simulator_err.txt").write_text(text)
        else:
            # GEM5 consumes the last window; retain its two-dump validation.
            (local / "stats.txt").write_text(
                "---------- Begin Simulation Statistics ----------\n"
                "---------- End Simulation Statistics ----------\n"
                + "\n".join(reversed(lines)) + "\n")
        record = {"name": identifier, "workload": workload, "point": int(point),
                  "benchmark": u.canonical_benchmark_name(workload.split("_")[0]),
                  "weight": float(profile_data[workload]["points"][point])}
        return record, None, set(values)

    last_report = 0
    with ThreadPoolExecutor(max_workers=request.get("extract_jobs", 4)) as pool:
        futures = {pool.submit(collect, item) for item in paths}
        for completed, future in enumerate(as_completed(futures), 1):
            record, exclusion, names = future.result()
            futures.remove(future)  # Release each completed result instead of retaining all raw key sets.
            if record:
                points.append(record)
            if exclusion:
                excluded.append(exclusion)
            raw_names.update(names)
            if time.monotonic() - last_report >= .5 or completed == len(paths):
                report_progress(dest, "并行读取原始计数器", completed, len(paths))
                last_report = time.monotonic()
    points.sort(key=lambda point: point["name"])
    if not points:
        raise ValueError("该归档没有已完成且具有完整测量窗口的切片")
    profile_copy = dest / "cluster.json"
    profile_copy.write_bytes(profile.read_bytes())
    print(f"提取 YAML Groups：{len(points)} 个切片", flush=True)
    report_progress(dest, "提取 YAML 计数器", len(points), len(paths))
    batch_args = [str(proc / "batch.py"), "-s", str(stage), "-g", "all", "--jobs",
                  str(request.get("extract_jobs", 4)), "-o", str(dest / "points.csv")]
    if backend == "xs":
        batch_args += ["-X"]
    command(proc, batch_args)
    frame = pd.read_csv(dest / "points.csv", index_col=0)
    if "cpi" not in frame and "ipc" in frame:
        frame["cpi"] = 1 / frame["ipc"].replace(0, float("nan"))
    required = {"cycles", "committedInsts", "cpi"}
    if not required.issubset(frame.columns):
        raise ValueError(f"缺少算分所需的 cycles / committedInsts / cpi；已有列：{list(frame.columns[:12])}")
    valid = frame[list(required)].apply(lambda col: col.map(lambda x: pd.notna(x) and math.isfinite(x) and x > 0)).all(axis=1)
    for identifier in frame.index[~valid]:
        excluded.append({"name": identifier, "reason": "invalid_required_counters"})
    frame = frame[valid]
    if frame.empty:
        raise ValueError("没有包含正的 cycles / committedInsts / cpi 的有效切片")
    points = [p for p in points if p["name"] in frame.index]
    frame.to_csv(dest / "points.csv")
    spec = request["spec"]
    print("SimPoint 加权与算分", flush=True)
    report_progress(dest, "SimPoint 加权与算分", len(points), len(paths))
    weight(proc, dest / "points.csv", profile_copy, dest / "weighted.csv", spec)
    weight(proc, dest / "points.csv", profile_copy, dest / "score.csv", spec, score=True)
    targets = [str(proc / "targets"), str(proc / "targets/local")]
    groups = load_groups(targets, list_groups(targets)).group_to_columns
    groups["basic"] = list(dict.fromkeys([*groups.get("basic", []), "cpi", "ipc"]))
    frame = pd.read_csv(dest / "points.csv", index_col=0)
    columns = [c for c in frame.columns if c not in {"point", "workload", "bmk"}]
    for name in ["cpi", "ipc"]:
        if name not in columns:
            columns.append(name)
    dump(dest / "catalog.json", {"groups": groups, "yaml_metrics": columns, "raw_metrics": sorted(raw_names)})
    expected = sum(len(v["points"]) for v in profile_data.values())
    manifest = {**request.get("manifest", {}), "points": points, "excluded": excluded,
                "completed_points": len(points), "expected_points": expected,
                "scores": frame_rows(pd.read_csv(dest / "score.csv", index_col=0)),
                "spec": spec, "schema_version": 1}
    dump(dest / "manifest.json", manifest)
    # Only the compact raw snapshots and CSV outputs are needed after preparation.
    for path in dest.glob("*_raw.csv"):
        path.unlink()
    for path in dest.glob("*_decomposed.csv"):
        path.unlink()


def compare(request):
    import pandas as pd

    metrics = request["metrics"]
    result = []
    for snapshot in request["snapshots"]:
        directory = Path(snapshot)
        manifest = json.loads((directory / "manifest.json").read_text())
        frame = pd.read_csv(directory / "points.csv", index_col=0)
        if "cpi" not in frame:
            frame["cpi"] = 1 / frame["ipc"].replace(0, float("nan"))
        raw_metrics = [m for m in metrics if m.startswith("raw:")]
        if raw_metrics:
            values = {}
            for point in manifest["points"]:
                with gzip.open(directory / "raw" / f'{point["name"]}.json.gz', "rt") as stream:
                    raw = json.load(stream)
                values[point["name"]] = {m: raw.get(m[4:]) for m in raw_metrics}
            raw_frame = pd.DataFrame.from_dict(values, orient="index")
            frame = pd.concat([frame, raw_frame.reindex(frame.index)], axis=1).copy()
        chosen = [m for m in metrics if m in frame and m != "ipc"]
        subset = frame[list(dict.fromkeys(["bmk", "workload", "point", "cpi", *chosen]))].copy()
        # Feed arbitrary raw counters through the same weighting implementation.
        out = Path(request["output"]) / str(len(result))
        out.mkdir(parents=True, exist_ok=True)
        if raw_metrics:
            subset.to_csv(out / "points.csv")
            weight(Path(__file__).resolve().parent, out / "points.csv", directory / "cluster.json",
                   out / "weighted.csv", manifest["spec"])
            weighted = pd.read_csv(out / "weighted.csv", index_col=0)
        else:
            weighted = pd.read_csv(directory / "weighted.csv", index_col=0)
        weighted["ipc"] = 1 / weighted["cpi"].replace(0, float("nan"))
        frame["ipc"] = 1 / frame["cpi"].replace(0, float("nan"))
        result.append({"benchmark": frame_rows(weighted), "simpoint": frame_rows(frame[[
            "bmk", "workload", "point", *[m for m in metrics if m in frame]]])})
    dump(Path(request["output"]) / "values.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "compare"])
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for key in ("source", "output", "profile_path", "stage_dir"):
        if key in request:
            request[key] = str(Path(request[key]).resolve())
    if "snapshots" in request:
        request["snapshots"] = [str(Path(path).resolve()) for path in request["snapshots"]]
    if request.get("backend", "gem5") not in {"gem5", "xs"}:
        parser.error("backend must be gem5 or xs")
    if not 1 <= request.get("extract_jobs", 4) <= 16:
        parser.error("extract_jobs must be between 1 and 16")
    proc = Path(__file__).resolve().parent
    os.chdir(proc)
    {"prepare": prepare, "compare": compare}[args.action](request)


if __name__ == "__main__":
    main()
