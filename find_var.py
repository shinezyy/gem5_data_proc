#!/usr/bin/env python3.6

import pandas as pd
import numpy as np


def read_raw():
    with open("./stat/big_core_whole_cache_dyn_core.csv") as f:
        data_frame_raw = pd.read_csv(
            filepath_or_buffer=f, header=0, sep=',', index_col=0,
            engine='python'
        )
        return data_frame_raw


def add_qos_to_dict(benchmark, d, qos):
    if benchmark in d:
        d[benchmark].append(qos)
    else:
        d[benchmark] = [qos]


df = read_raw()
# print(df.index.values)
# print(df.loc['GemsFDTD_bwaves'])

d = dict()

for index, row in df.iterrows():
    b1, b2 = index.split('_')
    qos1 = row.loc['QoS_0']
    qos2 = row.loc['QoS_1']
    add_qos_to_dict(b1, d, qos1)
    add_qos_to_dict(b2, d, qos2)

df2 = pd.DataFrame.from_dict(d, orient='index')
df2['std'] = df2.std(axis=1)
df2.sort_values(['std'], ascending=False, inplace=True)
print(df2)

