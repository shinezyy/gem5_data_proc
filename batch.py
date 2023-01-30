#!/usr/bin/env python3

from os.path import join as pjoin
import os.path as osp
import argparse
import pandas as pd

from utils import common as c
from utils.target_stats import *
import utils as u

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


def main():
    parser = argparse.ArgumentParser(usage='specify stat directory')
    parser.add_argument('-s', '--stat-dir', action='store', required=True,
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
    parser.add_argument('--smt', action='store_true',
                        help='processing SMT stats'
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
    parser.add_argument('-w', '--warmup', action='store_true',
                        help='print warmup stats'
                       )
    parser.add_argument('-F', '--filter-bmk', action='store',
                        help='Only print select benchmark'
                       )

    parser.add_argument('--xiangshan', action='store_true',
                        help='handle XiangShan stats'
                       )
    parser.add_argument('--old-xs', action='store_true',
                        help='handle old xs stats'
                       )

    parser.add_argument('--eval-stat', action='store',
            help='evaled stats',
            )

    opt = parser.parse_args()

    stat_file = opt.stat_file
    if opt.stat_file is None:
        if not opt.xiangshan:
            stat_file = 'stats.txt'
        else:
            stat_file = 'err.txt'

    paths = u.glob_stats(opt.stat_dir, fname=stat_file)

    print(paths)
    assert len(paths) > 0

    matrix = {}

    require_flag = False
    xs_stat_fmt = opt.xiangshan or opt.old_xs
    for workload, path in paths:
        if opt.filter_bmk and not workload.startswith(opt.filter_bmk):
            continue
        if xs_stat_fmt:
            flag_file = osp.join(osp.dirname(path), 'completed')
        else:
            flag_file = osp.join(osp.dirname(osp.dirname(path)), 'completed')
        if require_flag and not osp.isfile(flag_file):
            print('Skip unfinished job:', workload, path, flag_file)
            continue
        
        print('Process finished job:', workload)
        # print(workload, path)
        # print(workload)
        if opt.ipc_only:
            if xs_stat_fmt:
                d = c.xs_get_stats(path, xs_ipc_target, re_targets=True)
            else:
                d = c.gem5_get_stats(path, ipc_target, re_targets=True)
        else:
            if xs_stat_fmt:
                targets = xs_ipc_target
                if opt.branch:
                    targets = {**xs_branch_targets, **targets}
                if opt.cache:
                    if opt.xiangshan:
                        targets = {**xs_cache_targets_nanhu, **targets}
                    elif opt.old_xs:
                        targets = {**xs_cache_targets_22_04_nanhu, **targets}
                    else:
                        raise Exception('Unknown xs stat format')

                d = c.xs_get_stats(path, targets, re_targets=True)
            else:
                targets = brief_targets
                if opt.branch:
                    targets += branch_targets
                if opt.fanout:
                    targets += fanout_targets

                if opt.fetch:
                    targets += fetch_targets

                if opt.breakdown:
                    targets += breakdown_targets
                if opt.op:
                    targets += operand_targets
                if opt.packet:
                    targets += packet_targets
                if opt.flow:
                    targets += flow_target
                if opt.fu:
                    targets += fu_targets
                if opt.beta:
                    targets += beta_targets
                if opt.cache:
                    targets += cache_targets
                if opt.warmup:
                    targets += warmup_targets

                d = c.gem5_get_stats(path, targets, re_targets=True)

            if opt.eval_stat:
                stat_targets = opt.eval_stat.split('#')
                for stat_target in stat_targets:
                    if opt.xiangshan:
                        targets += eval('xs_'+stat_target)
                    else:
                        targets += eval(stat_target)

        if xs_stat_fmt:
            prefix = 'xs_'
        else:
            prefix = ''
        if len(d):
            if opt.smt:
                matrix[workload] = further_proc(workload, d, opt.verbose)
            else:
                matrix[workload] = d
            if opt.branch:
                eval(f"c.{prefix}add_branch_mispred(d)")
            if opt.cache:
                eval(f"c.{prefix}add_cache_mpki(d)")
            if opt.fanout:
                c.add_fanout(d)
            if opt.warmup:
                c.add_warmup_mpki(d)
            # if opt.packet:
            #     c.add_packet(d)

    df = pd.DataFrame.from_dict(matrix, orient='index')
    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)
    # df = df.sort_values(['ipc'])
    # for x in df.index:
    #     print(x)
    if len(df):
        df.loc['mean'] = df.mean()

    df = df.fillna(0)

    print(df)

    if opt.output:
        df.to_csv(opt.output, index=True)

    # print('filted QoS')
    # print(df['QoS_0'][df['QoS_0'] < 0.9])

if __name__ == '__main__':
    main()
