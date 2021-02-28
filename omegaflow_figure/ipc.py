#!/usr/bin/env python3

import os.path as osp
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import common as c, target_stats as t
import graphs

strange_const = 3
show_lins = 62
pd.set_option('precision', 3)
pd.set_option('display.max_rows', show_lins)
pd.set_option('display.min_rows', show_lins)

do_normalization = True
# do_normalization = False

with open('./bench_order.txt') as f:
    tpi_order = [l.strip() for l in f]

stat_dirs = {
    # 'Xbar4': 'xbar4',
    # 'F1': 'xbar4-rand',
    'F1': 'f1_rand_ltu1_b2b_x5',
    # 'Xbar4-SpecSB': 'xbar4-rand-hint',
    # 'Xbar4*2-SpecSB': 'dedi-xbar4-rand-hint',
    #'Omega16': 'omega',
    #'Omega16-OPR': 'omega-rand',
    # 'O1': 'omega-rand-hint',
    'O1': 'o1_rand_hint_ltu1_b2b_x5',
    #'Xbar16': 'xbar',
    #'Xbar16-OPR': 'xbar-rand',
    #'Xbar16-OPR-SpecSB': 'xbar-rand-hint',
    'Ideal-OOO': 'trad_4w',
    # 'o2': 'o2_rand_hint',
    # 'o4': 'o4_rand_hint',
}

# full = '_full'
full = '_f'
# bp = '_obp'
bp = '_o'
# bp = '_perceptron'
lbuf = '_lbuf'

for k in stat_dirs:
    stat_dirs[k] = c.env.data(f'{stat_dirs[k]}{lbuf}{bp}{full}')
configs_ordered = ['F1', 'O1', 'Ideal-OOO']

benchmarks = [*c.get_spec2017_int(), *c.get_spec2017_fp()]

points = []
for b in benchmarks:
    for i in range(0, 3):
        points.append(f'{b}_{i}')

num_points = 0
num_configs = len(stat_dirs)
dfs = dict()
for config in configs_ordered:
    # print(config)
    stat_dir = stat_dirs[config]
    stat_dir = osp.expanduser(stat_dir)
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]

    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file,
                # t.ipc_target + t.packet_targets,
                t.ipc_target + t.cache_targets + t.packet_targets,
                re_targets=True)
        matrix[point] = d
    df = pd.DataFrame.from_dict(matrix, orient='index')

    dfs[config] = df
    if num_points == 0:
        num_points = len(df)
# print(dfs['Ideal-OOO'].columns)

dfs['Ideal-OOO']['access_latency'] = \
    dfs['Ideal-OOO']['dcache.demand_miss_rate'] * dfs['Ideal-OOO']['dcache.demand_avg_miss_latency'] / 500 + \
    (1 - dfs['Ideal-OOO']['dcache.demand_miss_rate']) * 2.0
dfs['Ideal-OOO']['mem_time'] = \
    dfs['Ideal-OOO']['access_latency'] * dfs['Ideal-OOO']['ExecLoadInsts']
dfs['Ideal-OOO']['factor'] = \
    dfs['Ideal-OOO']['mem_time'] / dfs['O1']['TotalP']

dfs['Ideal-OOO'].sort_values(inplace=True, by=['mem_time'])
cache_miss_order = list(dfs['Ideal-OOO'].index)

# selected = index_order[:len(index_order)//2]
# selected = index_order[len(index_order)//2:]
# print(selected)

# drop useless
for config in configs_ordered:
    df = dfs[config]
    df = df[['ipc']]
    dfs[config] = df

# print(dfs['Ideal-OOO'])
dfs['Ideal-OOO'].loc['geo_mean'] = {'ipc': 1.0}
print('Ideal-OOO')
# print(dfs['Ideal-OOO'])

num_points += 1

gm = graphs.GraphMaker(
    fig_size=(14, 5),
    multi_ax=True,
    nrows=2,
    ncols=1,
)


num_points += 1
if do_normalization:
    num_configs -= 1
    configs_ordered = configs_ordered[:-1]

ax_id = 0
# compute relative performance
for config in configs_ordered:
    if config != 'Ideal-OOO':
        # print(config)
        rel = dfs[config]['ipc'] / dfs['Ideal-OOO']['ipc'][:-1]
        dfs[config]['rel'] = rel
        # print(dfs[config].columns)
        dfs[config].loc['geo_mean'] = [rel.prod() ** (1 / len(rel))] * 2
        if config == 'O1':
            dfs[config]['boost'] = dfs[config]['rel'] / dfs['F1']['rel'] - 1
            print('O1 boost the performance of F1 by', dfs[config]['boost'][-1])
        print(f'Relative performance of {config} is {dfs[config].loc["geo_mean"]}')


def draw(order: [], xlabel, title):
    global num_configs
    global configs_ordered
    global dfs
    global ax_id

    gm.set_cur_ax(gm.ax[ax_id])
    ax_id += 1
    plt.sca(gm.cur_ax)
    data_all = []

    for config in configs_ordered:
        df = dfs[config]
        df = df[['ipc', 'rel']]
        dfs[config] = df

    for i, config in enumerate(configs_ordered):
        df = dfs[config]
        # whitespace before geomean
        # insert_val = np.ones(1) if i == 2 and do_normalization else np.zeros(1)
        df = df.reindex(order)
        data = df['rel']
        # data = np.concatenate((df['rel'].values[:-1], [np.nan], df['rel'].values[-1:]))
        data_all.append(data)

    data_all = np.array(data_all)
    # print(data_all)

    # if do_normalization:
    #     data_all = np.array([data_all[0] / data_all[2], data_all[1] / data_all[2]])
        # print(data_all, data_all.shape)

    xticklabels = list(order) + ['', 'mean']

    # print(xticklabels)
    fig, ax = gm.simple_bar_graph(
        data_all, xticklabels, configs_ordered,
        # xlabel='Simulation points from SPEC 2017',

        xlabel=xlabel,
        ylabel='Normalized IPC',
        title=title,

        xlim=(-0.5, num_points-0.5),
        ylim=(0.0, 1.05 if do_normalization else 3),
        with_borders=False,
        # markers=['x', '+'],
        markers=['x', '+', 'v'],
        show_tick=True,
    )
    ax.legend(
        configs_ordered,
        loc='best',
        ncol=3,
        fancybox=True,
        framealpha=0.5,
        fontsize=13,
    )


print(dfs)
draw(tpi_order + ['', 'geo_mean'],
     xlabel='(a) Simulation points from SPECCPU 2017 sorted by Token per Instruction (TPI)',
     # title="Performance of F1 and O1 (normalized to ideal OoO core)")
     title=None)
draw(cache_miss_order + ['', 'geo_mean'],
     xlabel='(b) Simulation points from SPECCPU 2017 sorted by Average Memory Access Time (AMAT)',
     title=None)

plt.tight_layout()
gm.save_to_file("ipc")
# plt.show(block=True)
