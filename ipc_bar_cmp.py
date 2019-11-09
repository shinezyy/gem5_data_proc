#!/usr/bin/env python3

import target_stats as t
import common as c
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(usage='experiment_dir baseline_dir -f filter')

parser.add_argument('-f', '--filter', action='store', default=None,
        help='benchmark filter file')

parser.add_argument('dirs', action='store', nargs=2,
        help='experiment_dir baseline_dir')

args = parser.parse_args()
experiment = args.dirs[0]
baseline = args.dirs[1]

selected_benchmarks = []
if args.filter is not None:
    with open(args.filter) as f:
        for line in f:
            if not line.startswith("#"):
                selected_benchmarks.append(line.strip())
else:
    selected_benchmarks = None


experiment = c.get_stats_from_parent_dir(experiment, selected_benchmarks,
        t.ipc_target, re_targets=True)
baseline = c.get_stats_from_parent_dir(baseline, selected_benchmarks,
        t.ipc_target, re_targets=True)

def extract(d: dict):
    ret = {}
    for k, v in d.items():
        ret[k] = v.get('ipc', 1)
    return ret

experiment = extract(experiment)
baseline = extract(baseline)

df = pd.DataFrame([experiment, baseline])

relative = df.iloc[0]/df.iloc[1]
relative = relative.fillna(1)

ideal = []
for n in relative:
    if n > 1.0:
        ideal.append(n)
    else:
        ideal.append(1.0)

# relative = ideal
print(relative)

mean = [np.mean(relative), 1]
geomean = [np.array(relative).prod() ** (1/len(relative)), 1]
df['mean'] = mean
df['geomean'] = geomean
df['inv_mean'] = np.mean(1/relative)
print(df)

# plt.bar(np.arange(len(df.columns)) - 0.3, df.iloc[0].values, width=0.3 )
# plt.bar(np.arange(len(df.columns)), df.iloc[1].values, width=0.3)
# plt.xticks(np.arange(len(df.columns)), df.columns, rotation=90)
# 
# plt.show()
