import pandas as pd
import numpy as np

speed = {
        1: 2000,
        2: 3000,
        4: 4890,
        8: 7072,
        16:11077,
        }
df = pd.read_csv('./outputs/xs/xs.csv', index_col=0)
speed = pd.DataFrame.from_dict(speed, orient='index')
speed = speed / 4
print(speed)
time_limit = 24 * 3600
insts = 50*10**6 + 50*10**6
df['cycles'] = insts / df['ipc']
df['demand_speed'] = df['cycles'] / time_limit
df = df.sort_values('demand_speed')

cores = 0
for demand in df['demand_speed']:
    core_demand = 0
    for core, s in speed.iterrows():
        if s[0] > demand:
            core_demand = core
            break
    if core_demand == 0:
        core_demand = 16
    # print(core_demand)
    cores += core_demand
print(cores)
