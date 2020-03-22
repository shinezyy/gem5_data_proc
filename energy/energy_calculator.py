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
    'OoO': osp.join(c.env.get_stat_dir(), 'ooo_4w-full'),
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
    print(df.sum(axis=1))
    df['DynamicPower'] = df.sum(axis=1) / (10**6)
    sim_time = 'sim_seconds'
    df[sim_time] = df_raw[sim_time]
    df['StaticPower'] = df[sim_time] * (
            power_sheet['Static'] * power_sheet['Copies']).sum()
    df.drop([sim_time], axis=1, inplace=True)
    df.to_csv(f'{arch}-Energy.csv')
    return df


def main():
    # fig
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 5)
    colors = plt.get_cmap('Dark2').colors
    rects = []

    # stats

    width = 0.3
    delta = width + 0.1
    iter = 0
    archs = ['OoO', 'F1', 'O1', ]
    # power = 'StaticPower'
    power = 'DynamicPower'
    one_bmk_width = len(archs) * delta
    for arch in archs:
        df = get_power_df(arch)
        l = len(df)
        xticks = np.arange(0, l * one_bmk_width, one_bmk_width)
        shift = delta * iter
        rects.append(ax.bar(xticks + shift, df[power], 0.4, color=colors[iter]))
        iter += 1
    ax.legend(rects, archs)
    ax.set_xlabel('SPECCPU 2017 benchmarks', fontsize=14)
    ax.set_ylabel('Energy consumption for 200M instructions (mJ)', fontsize=14)
    plt.show()


if __name__ == '__main__':
    main()
