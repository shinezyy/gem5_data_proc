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
                        help='name of stats file', default='m5out/stats.txt'
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

    parser.add_argument('--xiangshan', action='store_true',
                        help='handle XiangShan stats'
                       )

    parser.add_argument('--eval-stat', action='store',
            help='evaled stats',
            )

    opt = parser.parse_args()

    paths = u.glob_stats_l2(opt.stat_dir, opt.stat_file)
    if len(paths) == 0:
        paths = u.glob_stats(opt.stat_dir, opt.stat_file)

    print(paths)
    assert len(paths) > 0

    matrix = {}

    for workload, path in paths:
        # print(workload, path)
        # print(workload)
        if opt.ipc_only:
            if opt.xiangshan:
                d = c.xs_get_stats(path, xs_ipc_target, re_targets=True)
            else:
                d = c.gem5_get_stats(path, ipc_target, re_targets=True)
        else:
            if opt.xiangshan:
                targets = xs_ipc_target
                if opt.branch:
                    targets += xs_branch_targets

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

                d = c.gem5_get_stats(path, targets, re_targets=True)

                if opt.eval_stat:
                    stat_targets = opt.eval_stat.split('#')
                    for stat_target in stat_targets:
                        targets += eval(stat_target)

        if len(d):
            if opt.smt:
                matrix[workload] = further_proc(workload, d, opt.verbose)
            else:
                matrix[workload] = d
            if opt.branch:
                if not opt.xiangshan:
                    c.add_branch_mispred(d)
            if opt.fanout:
                c.add_fanout(d)
            if opt.cache:
                c.add_cache_mpki(d)
            # if opt.packet:
            #     c.add_packet(d)

    df = pd.DataFrame.from_dict(matrix, orient='index')
    df = df.sort_index()
    df = df.sort_index(1)
    df = df.sort_values(['ipc'])
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
