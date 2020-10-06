import pandas as pd
import numpy as np
import os
from os.path import join as pjoin

baseline_name = 'ooo-128.csv'
baseline = pd.read_csv(f'./data/{baseline_name}', index_col=0)

result = None
for f in os.listdir('./data'):
    if f.endswith('.csv') and f != baseline_name:
        path = pjoin('./data', f)
        ipcs = pd.read_csv(path, index_col=0)
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
