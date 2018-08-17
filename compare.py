import pandas as pd
import numpy as np

origin = pd.read_csv('./origin.csv', index_col=0).values
tpi = pd.read_csv('./tpi.csv', index_col=0).values

rel_ipc = tpi/origin
print(rel_ipc)

print('mean =', np.mean(rel_ipc))
