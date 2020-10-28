import pandas as pd
import numpy as np
import os
from os.path import join as pjoin
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, action='store', required=True)
parser.add_argument('-b', '--baseline', type=str, action='store', required=True)
parser.add_argument('-s', '--suffix', type=str, action='store')

args = parser.parse_args()

baseline = pd.read_csv(args.baseline, index_col=0)
baseline.sort_index(inplace=True)
result = None

csvs = []

if os.path.isdir(args.input):
    baseline_name = args.baseline.split('/')[-1]
    for f in os.listdir(args.input):
        if f.endswith(args.suffix) and f != baseline_name:
            csvs.append(pjoin(args.input, f))
else:
    csvs = [args.input]

for f in csvs:
    ipcs = pd.read_csv(f, index_col=0)
    ipcs.sort_index(inplace=True)
    print(f)
    if result is None:
        result = pd.DataFrame(ipcs.values / baseline.values,
                columns=[f], index=baseline.index)
    else:
        result.loc[:, f] = pd.Series((ipcs.values / baseline.values)[:, 0],
                index=baseline.index, )

result.loc['mean'] = result.values.mean(axis=0)
result.to_csv('./results/compare.csv', float_format='%.3f')
print(result)
