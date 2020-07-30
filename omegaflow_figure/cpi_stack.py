import common as c
import os.path as osp
import target_stats as t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

stat_dirs = {
    'o1': 'o1_rand_hint',
    # 'o1x': 'o1_rand_hint_x',
    # 'o1x2': 'o1_rand_hint_x2',
    # 'o1x3': 'o1_rand_hint_x3',
    'o1x4': 'o1_rand_hint_x4',
    'final': 'o1_rand_hint_ltu1_b2b_x5',
    'trad': 'trad_4w',
}

# ful = ''
# full = '_full'
full = '_f'
# obp = ''
# bp = '_obp'
bp = '_o'
# bp = '_perceptron'
lbuf = '_lbuf'

for k in stat_dirs:
    stat_dirs[k] = c.env.data(f'{stat_dirs[k]}{lbuf}{bp}{full}')

points = c.get_all_spec2017_simpoints()
print(points)
data = []
data_dict = {}
for config in stat_dirs:
    stat_dir = stat_dirs[config]
    stat_files = [osp.join(stat_dir, point, 'stats.txt') for point in points]
    matrix = {}
    for point, stat_file in zip(points, stat_files):
        d = c.get_stats(stat_file, t.breakdown_targets + t.brief_targets, re_targets=True)
        matrix[point] = d
    df = pd.DataFrame.from_dict(matrix, orient='index')
    df['point'] = df.index
    df.reset_index(drop=True, inplace=True)
    df['config'] = [config] * len(df)
    df = c.scale_tick(df)
    data_dict[config] = df
    data.append(df)
    print(df)

delays = ['queueingDelay', 'ssrDelay', 'pendingDelay', 'readyExecDelayTicks',]


def draw_stacked_bar(conf, baseline, title):
    print(data_dict)
    df = data_dict[conf]
    baseline_df = data_dict[baseline]
    ind = np.arange(len(df))
    width = 0.5
    rects = []
    omega_cpi = 1.0/df['ipc']
    baseline_cpi = 1.0/baseline_df['ipc']
    # rects.append(plt.bar(ind, omega_cpi, width))
    rects.append(plt.bar(ind, -(omega_cpi - baseline_cpi) * 4.0, width + 0.2))
    bots = [0] * len(df)
    insts = df['Insts']
    for delay in delays:
        y = df[delay]/insts
        rects.append(plt.bar(ind, y, bottom=bots, width=width))
        bots = y+bots

    plt.ylim((-1, 4.5))
    plt.xticks(ind, df.point, rotation=90)
    plt.title(title)
    rects_l = [x[0] for x in rects]
    plt.legend(rects_l, ['CPI diff'] + delays)


def draw_correlation(conf, baseline):
    df = data_dict[conf]
    baseline_df = data_dict[baseline]
    omega_cpi = 1.0/df['ipc']
    baseline_cpi = 1.0/baseline_df['ipc']
    Ys = omega_cpi - baseline_cpi
    insts = float(200 * 10**6)
    Xs = [0] * len(df)
    for delay in delays:
        Xs += df[delay]/insts
        print(Xs)
    sns.scatterplot(x=Xs, y=Ys)


fig = plt.gcf()
fig.set_size_inches(10, 5)

# draw_stacked_bar('o1', 'trad', 'Original')
# draw_stacked_bar('o1x', 'trad', 'Improvement 1')
# draw_stacked_bar('o1x2', 'trad', 'Improvement 1 + 2')
# draw_stacked_bar('o1x3', 'trad', 'Improvement 1 + 2 + 3')
draw_stacked_bar('o1x4', 'trad', 'Improvement 1 + 2 + 4')
# draw_stacked_bar('final', 'trad', 'Improvement x5')
# draw_correlation('o1x2', 'trad')

plt.show()

