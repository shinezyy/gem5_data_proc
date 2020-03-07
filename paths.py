import os
import os.path as osp

script_dir = os.path.dirname(os.path.realpath(__file__))

st_stat_dir = osp.join(script_dir, 'data')

st_cache = os.path.join(script_dir, 'cache_st_stat.txt')

