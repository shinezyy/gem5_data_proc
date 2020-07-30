#!/usr/bin/env python3

import os.path as osp
import sys
sys.path.append('..')

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import common as c
import graphs
import target_stats as t

strange_const = 3

full = '_f'
bp = '_o'
# bp = '_perceptron'
lbuf = '_lbuf'

suffix = f'{lbuf}{bp}{full}'

n_cols = 1
n_rows = 3

gm = graphs.GraphMaker(
        fig_size=(7,5.5),
        multi_ax=True,
        legend_loc='best',
        with_xtick_label=False,
        nrows=n_rows, ncols=n_cols,
        sharex=False)

with open('./bench_order.txt') as f:
    index_order = [l.strip() for l in f]

colors = plt.get_cmap('Dark2').colors
color_map = {
    'F1': colors[0],
    'F1-2': colors[0],
    'O1*': colors[1],
    'O1* w/ Xbar16': colors[2],
}


def draw_queueing_throughput():
    global gm
    gm.set_cur_ax(gm.ax[0])
    plt.sca(gm.cur_ax)
    baseline_stat_dirs = {
            'F1': c.env.data(f'f1_rand_ltu1_b2b_x5{suffix}'),
            'F1-2': c.env.data(f'f1_rand_ltu1_b2b_x5{suffix}'),
            }
    stat_dirs = {
            # 'Xbar4*2-SpecSB': c.env.data('dedi-xbar4-rand-hint'),
            'O1*': c.env.data(f'o1_rand_ltu1_b2b_x5{suffix}'),
            'O1* w/ Xbar16': c.env.data(f'o1_rand_xbar_ltu1_b2b_x5{suffix}'),
            # 'Xbar16-OPR': c.env.data('xbar-rand'),
            }
    # for k in baseline_stat_dirs:
    #     baseline_stat_dirs[k] += suffix
    # for k in stat_dirs:
    #     stat_dirs[k] += suffix

    baselines_ordered = [x for x in baseline_stat_dirs]
    configs_ordered = [x for x in stat_dirs]
    benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

    points = []
    for b in benchmarks:
        for i in range(0, 3):
            points.append(f'{b}_{i}')

    num_configs = len(stat_dirs)

    data_all = []

    def process_data(configs: [str], dirs: dict):
        for baseline in configs:
            stat_dir = dirs[baseline]
            stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]
            matrix = {}
            for point, stat_file in zip(points, stat_files):
                d = c.get_stats(stat_file, t.brief_targets + t.breakdown_targets, re_targets=True)
                matrix[point] = d
            df = pd.DataFrame.from_dict(matrix, orient='index')
            df = df.reindex(index_order)

            num_points = len(df)

            df.loc['mean'] = df.iloc[-1]
            df.loc['mean']['queueingDelay'] = np.mean(df['queueingDelay'])
            df.loc['mean']['Insts'] = np.mean(df['Insts'])

            df['queueingDelayPerInst'] = df['queueingDelay'] / df['Insts']

            indices = df.index

            data = np.concatenate([
                df['queueingDelayPerInst'].values[:-1],
                [np.NaN],
                df['queueingDelayPerInst'].values[-1:]
            ])
            # print(data.shape)
            data_all.append(data)
        return num_points, indices

    process_data(baselines_ordered, baseline_stat_dirs)
    num_points, indices = process_data(configs_ordered, stat_dirs)

    print(f'Queueing cycle reduced by {configs_ordered[0]}:',
          data_all[2][-1]/data_all[0][-1] - 1)

    print(f'Queueing cycle reduced by {configs_ordered[1]}:',
          data_all[3][-1]/data_all[1][-1] - 1)

    num_points += 2
    data_all = np.array(data_all)
    # print(data_all.shape)

    benchmarks_ordered = []
    for point in indices:
        if point.endswith('_0'):
            benchmarks_ordered.append(point.split('_')[0])

    xticklabels = [''] * num_points
    # print(len(xticklabels))
    for i, benchmark in enumerate(benchmarks_ordered + ["mean"]):
        xticklabels[i*strange_const + 1] = benchmark

    # print(num_points, num_configs)
    # print(xticklabels)
    legends = baselines_ordered + configs_ordered
    color_config = [[color_map[x] for x in configs_ordered], [color_map[x] for x in baselines_ordered]]
    # print(color_config)
    fig, ax = gm.reduction_bar_graph(data_all[:2], data_all[2:], xticklabels, legends, 
            xlabel='Simulation points from SPECCPU 2017 (sorted by TPI)',
            ylabel='Token queueing cycles\nper instruction',
            xlim=(-0.5, num_points-0.5),
            # xtick_scale=1.5,
            title = "(a) Impact of PDQB and interconnect networks on token queueing cycles",
            colors=color_config,
            # colors=[['red', 'gray'], ['green', 'green']],
            markers=[['x', '+'], ['.', '.']],
            redundant_baseline=True,
            )


def draw_ipc_throughput():
    global gm
    # print(gm.ax)
    gm.set_cur_ax(gm.ax[1])
    plt.sca(gm.cur_ax)

    show_reduction = False

    stat_dirs = {
        'F1': c.env.data(f'f1_rand_ltu1_b2b_x5{suffix}'),
        'O1*': c.env.data(f'o1_rand_ltu1_b2b_x5{suffix}'),
        'O1* w/ Xbar16': c.env.data(f'o1_rand_xbar_ltu1_b2b_x5{suffix}'),
        'baseline': c.env.data(f'trad_4w{suffix}'),
    }

    configs_ordered = ['F1', 'O1*', 'O1* w/ Xbar16']
    baseline = 'baseline'

    benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

    points = []
    for b in benchmarks:
        for i in range(0, 3):
            points.append(f'{b}_{i}')

    data_all = []
    num_points, num_configs = 0, len(stat_dirs)
    dfs = dict()
    for config in configs_ordered + [baseline]:
        # print(config)
        stat_dir = stat_dirs[config]
        stat_dir = osp.expanduser(stat_dir)
        stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

        matrix = {}
        for point, stat_file in zip(points, stat_files):
            d = c.get_stats(stat_file, t.ipc_target, re_targets=True)
            matrix[point] = d

        df = pd.DataFrame.from_dict(matrix, orient='index')
        df = df.reindex(index_order)
        dfs[config] = df

        if num_points == 0:
            num_points = len(df)

    # dfs[baseline].loc['rel_geo_mean'] = [1.0]
    # print(baseline)
    # print(dfs[baseline])
    for config in configs_ordered:
        if config != baseline:
            # print(config)
            rel = dfs[config]['ipc'] / dfs[baseline]['ipc']
            geo_mean = rel.prod() ** (1/len(rel))
            print(f'rel geomean of {config}: {geo_mean}')
            data = np.concatenate([rel, [np.NaN], [geo_mean]])
            data_all.append(data)
            if config.startswith('O1'):
                print(f'geomean improvement of {config}: {geo_mean - data_all[0][-1]}')

    num_points += 2
    data_all = np.array(data_all)

    benchmarks_ordered = []
    for point in df.index:
        if point.endswith('_0'):
            benchmarks_ordered.append(point.split('_')[0])

    xticklabels = [*index_order, '', 'mean']

    # print(data_all.shape)
    # print(xticklabels)
    if show_reduction:
        num_configs -= 1
        data_high = np.array([data_all[2], data_all[2]])
        data_low = data_all[:2]
        legends = [configs_ordered[2], configs_ordered[0], configs_ordered[1]]
        fig, ax = gm.reduction_bar_graph(data_high, data_low, xticklabels, legends,
                # xlabel='Simulation points from SPEC 2017',
                # ylabel='IPC with different configurations',
                ylabel='IPC',
                xlim=(-0.5, num_points),
                ylim=(0, 3),
                title = "(b) IPC improved by PDQB\n" +\
                        "and wider interconnect networks",
                colors=[['red', 'gray'], ['green', 'green']],
                markers=[['x', '+'], ['.', '.']],
                redundant_baseline=True,
                )
    else:
        fig, ax = gm.simple_bar_graph(
            data_all, xticklabels, configs_ordered,
            xlabel='Simulation points from SPECCPU 2017 (sorted by TPI)',
            ylabel='Normalized IPC',
            # ylabel='IPC with different configurations',
            xlim=(-0.5, num_points-0.5),
            ylim=(0, 1.1),
            title = "(b) Impact of PDQB and interconnect networks on IPC",
            # colors=['green', 'red', 'gray'],
            colors=[color_map[x] for x in configs_ordered],
            markers=['.', 'x', '+'],
        )


def draw_correlation():
    # extract stats
    stat_dirs = {
        'F1': c.env.data(f'f1_rand_ltu1_b2b_x5{suffix}'),
        'O1*': c.env.data(f'o1_rand_ltu1_b2b_x5{suffix}'),
        'O1* w/ Xbar16': c.env.data(f'o1_rand_xbar_ltu1_b2b_x5{suffix}'),
        'ideal': c.env.data(f'trad_4w{suffix}'),
    }

    dfs = {}
    for config in stat_dirs:
        matrix = {}
        for point in index_order:
            d = c.get_stats(
                osp.join(stat_dirs[config], point, 'stats.txt'),
                t.brief_targets + t.breakdown_targets,
                re_targets=True)
            matrix[point] = d
        df = pd.DataFrame.from_dict(matrix, orient='index')
        dfs[config] = df

    ideal_ipc = dfs['ideal']['ipc']
    for config in stat_dirs:
        if config != 'ideal':
            df = dfs[config]
            df['rel_cpi'] = 1/(df['ipc'] / ideal_ipc)
            df['rel_ipc'] = df['ipc'] / ideal_ipc
            df['Delayed cycles per instruction'] = \
                (
                    df['queueingDelay']
                ) / df['Insts']

    def correlation(conf, ax):
        # fetch_rate_improved = get_improvement('fetch.rate', 'loop buf')
        # ipc_improved = get_improvement('ipc', 'loop buf')
        sns.set(style="whitegrid")
        # sns.scatterplot(
        sns.regplot(
            ax=ax,
            x=dfs[conf]['Delayed cycles per instruction'],
            y=dfs[conf]['rel_cpi'],
            color=color_map[conf],
            label=conf,
            marker='.'
        )
        ax.legend()
        ax.set(
            title='(C) Correlation between CPI and token queue cycles',
        )
        ax.set_xlabel('Token queueing cycles per instruction', fontsize=13)
        ax.set_ylabel('Normalized CPI', fontsize=13)

    gm.set_cur_ax(gm.ax[2])
    plt.sca(gm.cur_ax)
    correlation('F1', gm.ax[2])
    correlation('O1*', gm.ax[2])


draw_queueing_throughput()
draw_ipc_throughput()
draw_correlation()

plt.tight_layout()
gm.save_to_file('merged_throughput')
# plt.show(block=True)

