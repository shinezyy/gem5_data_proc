import common as c
import os.path as osp
import target_stats as t
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


with open('./bench_order.txt') as f:
    index_order = [l.strip() for l in f]

full = '_f'
bp = '_o'
# bp = '_perceptron'
lbuf = '_lbuf'
suffix = f'{lbuf}{bp}{full}'
