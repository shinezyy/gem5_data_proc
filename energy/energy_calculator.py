import common as c
import target_stats as t
import os
import sys
import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import energy.counters as ct
from matplotlib import cm

show_lins = 62
pd.set_option('precision', 3)
pd.set_option('display.max_rows', show_lins)
pd.set_option('display.min_rows', show_lins)
pd.set_option('display.max_columns', 15)
# pd.set_option('display.min_cols', show_lins)


data = {
    'F1': osp.join(c.env.get_stat_dir(), 'f1_rand-full'),
    'O1': osp.join(c.env.get_stat_dir(), 'o1_rand_hint-full'),
    'OoO': osp.join(c.env.get_stat_dir(), 'trad_4w-full'),
}
power_sheet_file = {
    'F1': osp.join('ff_power.csv'),
    'O1': osp.join('of_power.csv'),
    'OoO': osp.join('ooo_power.csv'),
}
irregular_stat = {
    'F1': False,
    'O1': False,
    'OoO': True,
}
with open('../omegaflow_figure/bench_order.txt') as f:
    index_order = [l.strip() for l in f]


def get_power_df(arch: str):
    df_raw = c.get_df(data, arch, t.ff_power_targets)
    cols = list(df_raw.columns)
    d = {}
    power_sheet = pd.read_csv(power_sheet_file[arch], header=0, index_col=0)
    # print(power_sheet)

    if not irregular_stat[arch]:
        for counter in ct.counters:
            # print(counter)
            for col in cols:
                if col.endswith(counter):
                    if counter in d:
                        d[counter] = df_raw[col] + d[counter]
                    else:
                        d[counter] = df_raw[col]

        # Data Frame merged
        df_raw_2 = pd.DataFrame.from_records(d)
    else:
        df_raw_2 = df_raw.fillna(0)

    df = pd.DataFrame()
    r = 'Read'
    w = 'Write'
    for module in power_sheet.index:
        if '#' in module:
            raw = module.replace('#', '')
            if not irregular_stat[arch]:
                r_str = module.replace('#', 'Read')
                w_str = module.replace('#', 'Write')
                df[raw] = df_raw_2[r_str] * power_sheet[r][module] + \
                          df_raw_2[w_str] * power_sheet[w][module]
            else:
                read_energy = df_raw_2[power_sheet['ReadStat'][module]] * power_sheet[r][module]
                if power_sheet['WriteStat'][module] != 'not_exist':
                    write_energy = df_raw_2[power_sheet['WriteStat'][module]] * power_sheet[w][module]
                    df[raw] = read_energy + write_energy
                else:
                    df[raw] = read_energy

        # else:
        #     df[unit] = df[unit] * power_sheet[r][unit]
    # print(df.sum(axis=1))

    static = 'StaticPower'
    dyn = 'DynamicPower'

    df[dyn] = df.sum(axis=1) / (10**6)
    sim_time = 'sim_seconds'
    df[sim_time] = df_raw[sim_time]
    df[static] = df[sim_time] * (
            power_sheet['Static'] * power_sheet['Copies']).sum()
    df['total'] = df[static] + df[dyn]
    df['ED'] = df['total'] * df_raw[sim_time]
    df['ED^2'] = df['total'] * df_raw[sim_time] * df_raw[sim_time]
    # df.drop([sim_time], axis=1, inplace=True)
    df = df.reindex(index_order)
    df.to_csv(f'{arch}-Energy.csv')
    return df


def main():
    # fig
    fig, ax = plt.subplots(1, 1, sharex=True)
    axs = [ax]
    fig.set_size_inches(7, 2.5)
    colors = plt.get_cmap('Dark2').colors
    rects = []

    # stats

    # width = 0.25 + 1e-9
    width = 0.20
    delta = width + 0.05
    iter = 0
    archs = ['OoO', 'F1', 'O1', ]
    one_bmk_width = len(archs) * delta + 0.25

    means = {}
    for arch in archs:
        df = get_power_df(arch)
        df.loc['mean'] = df.mean()
        df.loc['empty'] = np.nan
        index = list(df.index)
        df = df.reindex(index[:-2] + [index[-1], index[-2]])
        print(df.index)

        shift = delta * iter

        l = len(df)
        xticks = np.arange(0, l * one_bmk_width, one_bmk_width)
        # Energy:
        rects.append(axs[0].bar(xticks + shift, df['total'], 0.3, color=colors[iter]))
        # E*D:
        # rects.append(axs[0].bar(xticks + shift, df['ED'], 0.3, color=colors[iter]))

        # l = len(df.columns)
        # xticks = np.arange(0, l * one_bmk_width, one_bmk_width)
        # rects.append(ax.bar(xticks + shift, df.loc['leela_0'], 0.3, color=colors[iter]))
        means[arch] = df['total']['mean']
        iter += 1
    print('omf / ff:', means['O1']/means['F1'])
    print('1-omf / OoO:', 1.0-means['O1']/means['OoO'])

    axs[0].legend(rects, archs, ncol=3)

    # xtick_locations = np.arange(delta, l * one_bmk_width + delta, one_bmk_width)
    # plt.xticks(ticks=xtick_locations, labels=list(df.index), rotation=90, fontsize=7)

    # axs[-1].set_xlabel('SPECCPU 2017 benchmarks', fontsize=12)
    axs[0].set_ylabel('Energy consumption (mJ)', fontsize=12)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    def save_to_file(name):
        for f in ['eps', 'png', 'pdf']:
            d = '/home/zyy/projects/gem5_data_proc/figures'
            filename = f'{d}/{name}.{f}'
            print("save to", filename)
            plt.savefig(filename, format=f'{f}')

    # plt.show()
    plt.tight_layout()
    save_to_file('energy')


if __name__ == '__main__':
    main()
