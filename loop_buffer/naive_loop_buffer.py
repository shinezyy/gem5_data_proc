import common as c
import os.path as osp
import target_stats as t
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

stat_dirs = {
    # 'w/o lbuf': 'loop_buffer_off',
    # 'loop buf': 'conservative_lbuf_on',
    # 'loop buf': 'conservative_lbuf_128_on',
    # 'backward branch buf': 'loop_buffer_on',
    # 'forward branch': 'lbuf_fjump_128_on',
    # '128x16': 'lbuf_fjump_128x16_on',
    # '128x32': 'lbuf_fjump_128x32_on',
    # '128x64': 'lbuf_fjump_128x64_on',
    # 'o1': 'o1_rand_hint',
    # 'o1x': 'o1_rand_hint_x',
    # 'o1x2': 'o1_rand_hint_x2',
    # 'o1x4': 'o1_ran_ hint_x4',
    # 'o1_b2b_x4': 'o1_rand_hint_ltu1_b2b_x4',
    # 'o4_b2b_x5': 'o4_rand_hint_ltu1_b2b_x5',
    # 'o2_b2b_x5': 'o2_rand_hint_ltu1_b2b_x5',
    'o1_b2b_x5': 'o1_rand_hint_ltu1_b2b_x5',
    'f1': 'f1_rand_ltu1_b2b_x5',
    # 'o2': 'o2_rand_hint',
    # 'o4': 'o4_rand_hint',
    'trad': 'trad_4w',
}

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
        d = c.get_stats(stat_file, t.fetch_targets + t.brief_targets, re_targets=True)
        matrix[point] = d
    df = pd.DataFrame.from_dict(matrix, orient='index')
    df['point'] = df.index
    df.reset_index(drop=True, inplace=True)
    df['config'] = [config] * len(df)
    data_dict[config] = df
    data.append(df)
    # print(df)

df = pd.concat(data)


def get_improvement(target: str, conf: str, baseline: str):
    return data_dict[conf][target] / data_dict[baseline][target]


def draw_bar(target: str):
    for conf in stat_dirs:
        fetch_rate_improved = get_improvement(target, conf, 'trad')
        print(f'{conf}: {target} improved = {fetch_rate_improved.mean() * 100 - 100: .2f}%')
        print(f'{conf}: {target} improved max = {fetch_rate_improved.max() * 100 - 100: .2f}%')

    sns.set(rc={'figure.figsize': (5, 8)})

    sns.set(style="whitegrid")
    ax = sns.barplot(
        x=target, y="point", data=df, hue='config',
    )
    ax.set(xlim=(0, 3.5),
           # xticklabels=[0, 2.0, 4.0, 6.0, 8.0]
           )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'Impact on {target}')


def draw_correlation():
    fetch_rate_improved = get_improvement('fetch.rate', 'loop buf')
    ipc_improved = get_improvement('ipc', 'loop buf')
    sns.set(rc={'figure.figsize': (6, 6)})
    sns.set(style="whitegrid")
    ax = sns.scatterplot(
        x=fetch_rate_improved, y=ipc_improved,
    )


def draw_loop_ratio(conf: str):
    print(data_dict[conf])
    data_dict[conf]['loop ratio'] = \
        data_dict[conf]['fetchFromLoopBuffer'] / data_dict[conf]['Insts']
    sns.set(rc={'figure.figsize': (5, 8)})

    sns.set(style="whitegrid")
    ax = sns.barplot(
        x='loop ratio', y="point", data=data_dict[conf],
    )
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(f'Ratio of instructions in loop')
    ax.set(xlim=(0, 1.0),
           # xticks=[0, 0.5, 1.0]
           )

    print('Mean loop ratio:', data_dict[conf]['loop ratio'].mean())



# draw_correlation()
draw_bar('ipc')
# draw_bar('fetch.rate')
# draw_loop_ratio('loop buf 128')
# draw_loop_ratio('forward branch')
plt.show()

