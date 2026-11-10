#!/usr/bin/env python3

from os.path import join as pjoin
import os
import os.path as osp
import argparse
import pandas as pd
import json
from multiprocessing import get_context

from utils import common as c
from utils.target_stats import *
from utils.targets_yaml import list_groups as yaml_list_groups
from utils.targets_yaml import load_groups as yaml_load_groups
from utils.derived_metrics import apply_derived_metrics
import utils as u

_EXTRACT_WORKER = None
_MAX_AUTO_JOBS = 64


def _run_extract_worker(item):
    return _EXTRACT_WORKER(item)


def _choose_jobs(requested, n_tasks):
    if n_tasks <= 1:
        return 1
    if requested is not None and requested > 0:
        return max(1, min(int(requested), n_tasks))
    cpu = os.cpu_count() or 1
    return max(1, min(n_tasks, cpu, _MAX_AUTO_JOBS))

show_lins = 62
pd.set_option('display.precision', 3)
pd.set_option('display.max_rows', show_lins)
pd.set_option('display.min_rows', show_lins)


def further_proc(pair: str, d: dict, verbose: bool) -> None:
    hpt, lpt = pair.split('_')
    # c.add_st_ipc(hpt, d)
    # c.add_overall_qos(hpt, lpt, d)
    # c.add_ipc_pred(d)
    # c.add_slot_sanity(d)
    # c.add_qos(d)

    if verbose:
        c.print_line()
        print(pair, ':')
        c.print_dict(d)

    return d


def add_eval_targets(opt, targets: dict):
    if opt.eval_stat:
        stat_targets = opt.eval_stat.split('#')
        for stat_target in stat_targets:
            if opt.xiangshan:
                print("Adding eval target: xs_", stat_target)
                targets.update(eval('xs_'+stat_target))
            else:
                print("Adding eval target:", stat_target)
                targets.update(eval(stat_target))
        # print(targets)


def main():
    parser = argparse.ArgumentParser(usage='specify stat directory')
    # Note: keep -s optional so `--list-groups` can work without requiring a dummy path.
    parser.add_argument('-s', '--stat-dir', action='store', required=False,
                        help='gem5 output directory'
                       )
    parser.add_argument('-o', '--output', action='store',
                        help='csv to save results'
                       )
    parser.add_argument('--branch', action='store_true')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='whether output intermediate result'
                       )
    parser.add_argument('-b', '--error-bound', action='store', type=float,
                        default=0.0,
                        help='Threshold to output an entry'
                       )
    parser.add_argument('-i', '--ipc-only', action='store_true',
                        default=0.0,
                        help='Only extract ipc'
                       )
    parser.add_argument('--pair-filter', action='store', default='',
                        help='file that filt pairs'
                       )
    parser.add_argument('-f', '--stat-file', action='store',
                        help='name of stats file',
                       )
    parser.add_argument('-l', '--fanout', action='store_true',
                        help='print fanout'
                       )
    parser.add_argument('--fetch', action='store_true',
                        help='print fetch info'
                        )
    parser.add_argument('-k', '--breakdown', action='store_true',
                        help='print breakdown'
                       )
    parser.add_argument('--op', action='store_true',
                        help='print operand busy state'
                       )
    parser.add_argument('--flow', action='store_true',
                        help='print bandwidth usages'
                       )
    parser.add_argument('-p', '--packet', action='store_true',
                        help='print type and number of different packets'
                       )

    parser.add_argument('-m', '--mem-pred', action='store_true',
                        help='print mem pred stats'
                       )
    parser.add_argument('--fu', action='store_true',
                        help='print fu stats'
                       )
    parser.add_argument('--sched', action='store_true',
                        help='print scheduling related stats'
                       )
    parser.add_argument('--beta', action='store_true',
                        help='print stats demanded by betapoint'
                       )
    parser.add_argument('--cache', action='store_true',
                        help='print cache stats'
                       )
    parser.add_argument('--num-cores', type= int, default= 1,
                        help='set multicore numbers'
                       )
    parser.add_argument('-w', '--warmup', action='store_true',
                        help='print warmup stats'
                       )
    parser.add_argument('-F', '--filter-bmk', action='store',
                        help='Only print select benchmark'
                       )
    parser.add_argument('--json-filter', action='store',
                        help='Only print select benchmark in json file'
                       )

    parser.add_argument('-t', '--topdown', action='store_true',
                        help='handle topdown stats'
                       )
    parser.add_argument('--topdown-raw', action='store_true',
                        help='handle topdown stats but dont post process'
                       )
    parser.add_argument('-t1', '--topdown-intel', action='store_true',
                        help='handle intel topdown stats'   
                        )
    parser.add_argument('-X', '--xiangshan', action='store_true',
                        help='handle XiangShan stats'
                       )
    parser.add_argument('--exclude-l3', action='store_true',
                        help='handle XiangShan stats without L3, for simulation results for CHI (L3 is openLLC or commercial IP)'
                       )
    parser.add_argument('--old-xs', action='store_true',
                        help='handle old xs stats'
                       )
    parser.add_argument('--nix', action='store_true',
                        help='handle nix stats'
                       )
    parser.add_argument('-T','--temp', action='store_true',
                        help='print temp stats'
                       )
    parser.add_argument('--eval-stat', action='store',
            help='evaled stats',
            )
    parser.add_argument('-g', '--groups', action='store', default='all',
                        help="comma-separated YAML groups to enable (default: all)."
                       )
    parser.add_argument('--list-groups', action='store_true',
                        help='list available YAML groups and exit'
                       )
    parser.add_argument('--jobs', type=int, default=0,
                        help='parallel extraction workers (0: auto, capped at 64; 1: sequential)'
                       )

    opt = parser.parse_args()

    target_dirs = ['targets', 'targets/local']
    if opt.list_groups:
        print("\n".join(yaml_list_groups(target_dirs)))
        return

    if not opt.stat_dir:
        # Preserve argparse-style UX when --list-groups is not used.
        parser.error("the following arguments are required: -s/--stat-dir")

    def _dedup_keep_order(xs):
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    selected_groups = [g.strip() for g in (opt.groups or "").split(',') if g.strip()]
    # Convenience: allow `-g all` to mean "enable every YAML group".
    if any(g.lower() == "all" for g in selected_groups):
        all_groups = yaml_list_groups(target_dirs)
        selected_groups = ["basic"] + [g for g in all_groups if g != "basic"]
    elif selected_groups and 'basic' not in selected_groups:
        selected_groups = ['basic'] + selected_groups

    selected_groups = _dedup_keep_order(selected_groups)

    if not selected_groups:
        raise SystemExit("empty --groups is not supported; use at least 'basic'")

    xs_stat_fmt = opt.xiangshan or opt.old_xs
    loaded = yaml_load_groups(target_dirs, selected_groups)
    yaml_gem5_targets = loaded.gem5_targets
    yaml_xs_targets = loaded.xs_targets
    backend_derived = (
        loaded.derived_xs if xs_stat_fmt else loaded.derived_gem5
    )
    # Backend-specific aliases are evaluated first so common expressions can
    # consume a normalized column regardless of the input stat format.
    yaml_derived = {**backend_derived, **loaded.derived}

    add_nanhu_multicore_ipc_targets(opt.num_cores)

    stat_file = opt.stat_file
    if opt.stat_file is None:
        if not opt.xiangshan:
            stat_file = 'stats.txt'
        else:
            stat_file = 'simulator_err.txt'

    paths = u.glob_stats(opt.stat_dir, fname=stat_file)

    # print(paths)
    if len(paths) == 0:
        import sys
        print(f"Error: No '{stat_file}' found in '{opt.stat_dir}'")
        print(f"Please check: 1) directory exists  2) file name is correct")
        sys.exit(1)

    require_flag = False
    if xs_stat_fmt:
        prefix = 'xs_'
    else:
        prefix = ''

    if opt.json_filter is not None:
        from analysis_inputs import filter_profile_paths
        with open(opt.json_filter) as json_file:
            paths = filter_profile_paths(paths, opt.stat_dir, json.load(json_file))
    if opt.filter_bmk:
        paths = [(workload, path) for workload, path in paths
                 if workload.startswith(opt.filter_bmk)]

    if opt.ipc_only:
        targets = yaml_xs_targets if xs_stat_fmt else yaml_gem5_targets
    elif xs_stat_fmt:
        targets = dict(yaml_xs_targets)
        if opt.branch:
            targets = {**xs_branch_targets, **targets}
        if opt.cache:
            if opt.xiangshan:
                if opt.exclude_l3:
                    targets = {**xs_cache_targets_no_l3, **targets}
                else:
                    targets = {**xs_cache_targets, **targets}
            elif opt.old_xs:
                targets = {**xs_cache_targets_22_04_nanhu, **targets}
            else:
                raise Exception('Unknown xs stat format')

        if opt.topdown:
            targets = {**xs_topdown_targets, **targets}

        if opt.topdown_intel:
            targets = {**xs_topdown_intel_targets, **targets}
        if opt.temp:
            targets = {**xs_temp_targets, **targets}

        add_eval_targets(opt, targets)
    else:
        targets = dict(yaml_gem5_targets)
        if opt.branch:
            targets = {**branch_targets, **targets}
        if opt.cache:
            targets = {**cache_targets, **targets}
        if opt.warmup:
            targets = {**warmup_targets, **targets}
        if opt.topdown:
            targets = {**topdown_targets, **targets}
        if opt.topdown_intel:
            targets = {**topdown_intel_targets, **targets}
        if opt.temp:
            targets = {**temp_targets, **targets}

        add_eval_targets(opt, targets)

    def extract_one(item):
        workload, path = item
        diagnostic = None
        if xs_stat_fmt:
            flag_file = osp.join(osp.dirname(path), 'completed')
        else:
            flag_file = osp.join(osp.dirname(osp.dirname(path)), 'completed')
        if require_flag and not osp.isfile(flag_file):
            return workload, None, diagnostic

        local_diag = {}
        if xs_stat_fmt:
            d = c.xs_get_stats(path, targets, re_targets=True,
                               required_keys={'committedInsts', 'cycles'},
                               diagnostics=local_diag)
        else:
            d = c.gem5_get_stats(path, targets, re_targets=True)
        if local_diag:
            diagnostic = next(iter(local_diag.items()))

        if d and len(d):
            if xs_stat_fmt:
                if 'commitInstr' not in d and 'insts' in d:
                    d['commitInstr'] = d['insts']
                if 'total_cycles' not in d and 'cycles' in d:
                    d['total_cycles'] = d['cycles']
            if 'ipc' not in d and 'insts' in d and 'cycles' in d and d.get('cycles', 0) != 0:
                d['ipc'] = d['insts'] / d['cycles']
            if opt.branch:
                eval(f"c.{prefix}add_branch_mispred(d)")
            if opt.cache:
                if opt.xiangshan:
                    c.xs_add_cache_mpki(d, opt.exclude_l3)
                else:
                    c.add_cache_mpki(d)
            if opt.fanout:
                c.add_fanout(d)
            if opt.warmup:
                c.add_warmup_mpki(d)

            if opt.eval_stat is not None:
                if 'mem_targets' in opt.eval_stat:
                    eval(f"c.{prefix}add_mem_bw(d)")
                if 'pf_targets' in opt.eval_stat:
                    eval(f'c.{prefix}add_pf_accuracy(d)')
                if 'rvv_targets' in opt.eval_stat:
                    c.rvv_post_process(d)

            # add bmk and point after topdown processing
            segments = workload.split('_')
            if len(segments):
                d['point'] = segments[-1]
                d['workload'] = '_'.join(segments[:-1])
                if opt.nix: # nix path is <num>_<benchmark>_checkpoint_<point>
                    d['bmk'] = u.canonical_benchmark_name(segments[1])
                else:
                    d['bmk'] = u.canonical_benchmark_name(segments[0])

            # if opt.packet:
            #     c.add_packet(d)
        return workload, d, diagnostic

    jobs = _choose_jobs(opt.jobs, len(paths))
    print(f'Extracting {len(paths)} file(s) using {jobs} worker(s)')

    global _EXTRACT_WORKER
    _EXTRACT_WORKER = extract_one

    if jobs == 1:
        results = [extract_one(item) for item in paths]
    else:
        try:
            ctx = get_context('fork')
            chunksize = 1 if len(paths) < 32 else max(1, len(paths) // (jobs * 4))
            with ctx.Pool(processes=jobs) as pool:
                results = pool.map(_run_extract_worker, paths, chunksize=chunksize)
        except Exception as e:
            print(f'warning: multiprocessing pool unavailable, falling back to sequential mode: {e}')
            results = [extract_one(item) for item in paths]

    all_bmk_dict = {}
    diagnostics = {}
    for workload, d, diagnostic in results:
        all_bmk_dict[workload] = d
        if diagnostic is not None:
            path, message = diagnostic
            diagnostics[path] = message

    warnings_by_message = {}
    for path, message in sorted(diagnostics.items()):
        warnings_by_message.setdefault(message, []).append(path)
    for message, examples in sorted(warnings_by_message.items()):
        print(f"warning: {len(examples)} file(s): {message}")
        print(f"  example: {examples[0]}")

    # Filter out None values
    all_bmk_dict = {k: v for k, v in all_bmk_dict.items() if v is not None}
    if not all_bmk_dict:
        raise SystemExit("Error: no valid stats remain; check profile filtering and required stats")

    df = pd.DataFrame.from_dict(all_bmk_dict, orient='index')

    if opt.topdown and not opt.topdown_raw:
        eval(f"c.{prefix}topdown_post_process(df)")

    df = df.sort_index()
    # df = df.sort_values(['ipc'])
    # for x in df.index:
    #     print(x)

    if not xs_stat_fmt:
        prefetch_columns = set(loaded.group_to_columns.get('l1_prefetch', []))
        prefetch_columns.update(loaded.group_to_columns.get('l2_prefetch', []))
        c.fill_missing_gem5_prefetch_counters(df, prefetch_columns)

    df = apply_derived_metrics(df, yaml_derived)

    # Keep CSV columns in YAML definition order (plus any extra columns at the end),
    # so downstream weighted CSV + compare UI show a stable, meaningful order.
    preferred_front = [c for c in ('bmk', 'workload', 'point') if c in df.columns]
    ordered = preferred_front + [
        c for c in loaded.column_order if c in df.columns and c not in preferred_front
    ]
    other = [c for c in df.columns if c not in ordered]
    df = df.reindex(columns=ordered + other)

    if opt.output:
        df.to_csv(opt.output, index=True)

    # print('filted QoS')
    # print(df['QoS_0'][df['QoS_0'] < 0.9])

if __name__ == '__main__':
    main()
