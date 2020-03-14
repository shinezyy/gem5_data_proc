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

data = {
    'F1': osp.join(c.env.get_stat_dir(), 'o2_rand_hint_power'),
    'O1': osp.join(c.env.get_stat_dir(), 'o2_rand_hint_power'),
    'OoO': osp.join(c.env.get_stat_dir(), 'o2_rand_hint_power'),
}


def main():
    df = c.get_df(data, 'O1', t.ff_power_targets)
    cols = list(df.columns)
    d = {}
    power_table = pd.read_csv('power_table.csv', header=0, index_col=0)
    print(power_table)

    for counter in ct.counters:
        # print(counter)
        for col in cols:
            if col.endswith(counter):
                if counter in d:
                    d[counter] = df[col] + d[counter]
                else:
                    d[counter] = df[col]

    # Data Frame merged
    df = pd.DataFrame.from_records(d)
    print(df)

    r = 'Read'
    w = 'Write'
    for unit in power_table.index:
        if '#' in unit:
            r_str = unit.replace('#', 'Read')
            w_str = unit.replace('#', 'Write')
            raw = unit.replace('#', '')
            df[raw] = df[r_str] * power_table[r][unit] + \
                      df[w_str] * power_table[w][unit]
        else:
            df[unit] = df[unit] * power_table[r][unit]

    print(df)





if __name__ == '__main__':
    main()
