#!/usr/bin/env python3

import target_stats as t
import common as c
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

assert len(sys.argv) == 3

experiment = sys.argv[1]
baseline = sys.argv[2]

experiment = c.get_stats_from_parent_dir(experiment, t.ipc_target, re_targets=True)
baseline = c.get_stats_from_parent_dir(baseline, t.ipc_target, re_targets=True)

def extract(d: dict):
    ret = {}
    for k, v in d.items():
        ret[k] = v['ipc']
    return ret

experiment = extract(experiment)
baseline = extract(baseline)

df = pd.DataFrame([experiment, baseline])

plt.bar(np.arange(len(df.columns)) - 0.3, df.iloc[0].values, width=0.3 )
plt.bar(np.arange(len(df.columns)), df.iloc[1].values, width=0.3)
plt.xticks(np.arange(len(df.columns)), df.columns, rotation=90)

plt.show()
