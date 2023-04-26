import os
import os.path as osp
import pandas as pd
import numpy as np
import utils.common as c
import utils.target_stats as t
import json
import re

def get_ipc(stat_path: str):
    targets = t.ipc_target
    stats = c.get_stats(stat_path, targets, insts=100*10**6, re_targets=True)
    return stats['ipc']

def single_stat_factory(targets, key, prefix=''):
    def get_single_stat(stat_path: str):
        # print(stat_path)
        if prefix == '':
            stats = c.get_stats(stat_path, targets, re_targets=True)
        else:
            assert prefix == 'xs_'
            stats = c.xs_get_stats(stat_path, targets, re_targets=True)
        if stats is not None and key in stats:
            return stats[key]
        else:
            return None
    return get_single_stat

#input may be: work_load_point/ ; workload_point/ ; work_load/point/ ; workload/point/ 
def workload_point_frompath(path):
    print(path)
    split_path = path.split('/')[0].split('_')
    print(split_path)
    second_layer = path.split('/')[1]
    level = 1
    if second_layer.isdigit() and len(second_layer) > 1:# workload/point/ ; work_load/point/ 
        workload = path.split('/')[0]
        point = second_layer
        level = 2
    elif len(split_path) > 1 and split_path[1].isdigit() and len(split_path[1]) > 5:# workload_point_xxx/
        print(split_path)
        workload = split_path[0]
        point = split_path[1]
    elif len(split_path) > 2 and split_path[2].isdigit():#work_load_point_xxx/
        workload = split_path[0] + '_' + split_path[1]
        point = split_path[2]
    else:
        workload = path.split('/')[0]
        point = '0'
        level = 1
        
    return workload,point,level


def strip_path(file_path: str, prefix_path: str):
    x = prefix_path.join(file_path.split(prefix_path)[1:])
    if prefix_path.startswith('.') and x.startswith('/'):
        x = x[1:]
    if prefix_path.startswith('/') and x.startswith('/'):
        x = x[1:]
    return x

def glob_stats(path: str, fname = 'x'):
    files = []
    #check for checkpoints conflict
    files_map = {}
    flatten_pat = re.compile(r'.*/(?P<workload_point>.*)/')
    two_layer_pat = re.compile(r'.*/(?P<workload>.*)/(?P<point>\d+)/')

    probe_stat_path = find_file_in_maze(path, fname)  # use it to probe the directory layout
    workload, point, segments = workload_point_frompath(strip_path(probe_stat_path, path))
    for l2_dir in os.listdir(path):
        l2_path = osp.join(path, l2_dir)
        #workload/point
        if segments == 2:
            # two layer directory
            for l3_dir in os.listdir(l2_path):
                l3_path = osp.join(l2_path, l3_dir)
                if not osp.isdir(l3_path):
                    continue
                stat_path = find_file_in_maze(l3_path, fname)
                if stat_path is not None:
                    workload,point,_ = workload_point_frompath(strip_path(stat_path, path))
                    point_identifier = workload + '_' + point
                    files_map.update({point_identifier:stat_path})
                    files.append((point_identifier, stat_path))
        else:
            #workload_point_xx/
            stat_path = find_file_in_maze(l2_path, fname)
            if stat_path is not None:
                workload, point, _ = workload_point_frompath(strip_path(stat_path, path))
                point_identifier = workload + '_' + point
                files_map.update({point_identifier:stat_path})
                files.append((point_identifier, stat_path))
    return files

def find_file_in_maze(path: str, stat_file='stats.txt'):
    file_path = osp.join(path, stat_file)
    if osp.isfile(file_path) or osp.islink(file_path):
        return file_path
    else:
        #print(f'No stat file found in {file_path}')
        if not osp.isdir(path):
            return None
    for l2_dir in os.listdir(path):
        l2_path = osp.join(path, l2_dir)
        if not osp.isdir(l2_path):
            continue
        ret = find_file_in_maze(l2_path, stat_file)
        if ret is not None:
            return ret
    return None


def glob_weighted_stats(path: str, get_func, filtered=True,
        simpoints='/home51/zyy/expri_results/simpoints17.json',
        stat_file='m5out/stats.txt', dir_layout='flatten'):

    assert dir_layout == 'flatten' or dir_layout == 'maze'
    stat_path_tree = {}
    stat_tree = {}
    with open(simpoints) as jf:
        points = json.load(jf)
    print(points.keys())

    for point_path in os.listdir(path):
        point_dir = osp.join(path, point_path)
        if not osp.isdir(point_dir):
            continue
        if '-0' in point_path:
            point = point_path.split('-0')[0]
        elif '_0.' in point_path:
            point = point_path.split('_0.')[0]
        else:
            point = point_path
        workload = '_'.join(point.split('_')[:-1])
        print('Extracting', point_path, point, workload)
        point = int(point.split('_')[-1])
        bmk = workload.split('_')[0]
        if dir_layout == 'flatten':
            point_stat_file = osp.join(point_dir, stat_file)
            print(point_dir)
            assert osp.isfile(point_stat_file)
            stat_path_tree[(bmk, workload, point)] = point_stat_file
        elif dir_layout == 'maze': # has to search until find stat_file
            stat_path = find_file_in_maze(point_dir, stat_file)
            assert stat_path is not None
            if stat_path is not None:
                stat_path_tree[(bmk, workload, point)] = stat_path
            else:
                print(f'No stat file found in {point_dir}')
        else:
            raise NotImplementedError
    
    weight_dict = json.load(open(simpoints))

    for (bmk, workload, point), stats_file in stat_path_tree.items():
        weight = float(weight_dict[workload][str(point)])
        stat = get_func(stats_file)
        if bmk not in stat_tree:
            stat_tree[bmk] = {}
        if workload not in stat_tree[bmk]:
            stat_tree[bmk][workload] = {}
        if stat is not None:
            print(bmk, workload, point)
            print(weight, stat)
            assert int(point) not in stat_tree[bmk][workload]
            stat_tree[bmk][workload][int(point)] = (weight, stat)

    for bmk in stat_tree:
        for workload in stat_tree[bmk]:
            if len(stat_tree[bmk][workload]):
                df = pd.DataFrame.from_dict(stat_tree[bmk][workload], orient='index')
                df.columns = ['weight', 'ipc']
                stat_tree[bmk][workload] = df
            else:
                stat_tree[bmk].pop(workload)
                if not len(stat_tree[bmk]):
                    stat_tree.pop(bmk)

    return stat_tree


def glob_weighted_cpts(path: str):
    stat_tree = {}
    for dir_name in os.listdir(path):
        weight_pattern = re.compile(f'(\w+)_(\d+)_(\d+\.\d+([eE][-+]?\d+)?)')
        workload_dir = osp.join(path, dir_name)
        print(dir_name)
        m = weight_pattern.match(dir_name)
        if m is not None:
            workload = m.group(1)
            point = m.group(2)
            weight = float(m.group(3))
            bmk = workload.split('_')[0]
            if bmk not in stat_tree:
                stat_tree[bmk] = {}
            if workload not in stat_tree[bmk]:
                stat_tree[bmk][workload] = {}
            stat_tree[bmk][workload][int(point)] = weight
    for bmk in stat_tree:
        for workload in stat_tree[bmk]:
            item = stat_tree[bmk][workload]
            df = pd.DataFrame.from_dict(item, orient='index')
            df.columns = ['weight']
            stat_tree[bmk][workload] = df
    return stat_tree



def coveraged(coverage: float, df: pd.DataFrame):
    df = df.sort_values('weight', ascending=False)
    i = 0
    cummulated = 0.0
    for row in df.iterrows():
        cummulated += row[1]['weight']
        i += 1
        if cummulated > coverage:
            break
    df = df.iloc[:i]
    return(cummulated, df)


def weighted_cpi(df: pd.DataFrame):
    if 'cpi' in df.columns:
        return np.dot(df['weight'], df['cpi']) / np.sum(df['weight']), np.sum(df['weight'])
    else:
        assert 'ipc' in df.columns
        print(df['weight'])
        print(df['ipc'])
        return np.dot(df['weight'], 1.0/df['ipc']) / np.sum(df['weight']), np.sum(df['weight'])


def gen_json(path, output):
    tree = glob_weighted_cpts(path)
    d = {}
    count = 0
    expected_coverage = 0.5
    for bmk in tree:
        for workload in tree[bmk]:
            d[workload] = {}
            coverage, selected = coveraged(expected_coverage, tree[bmk][workload])
            weights = selected['weight']
            for row in weights.index:
                count += 1
                d[workload][int(row)] = weights[row]
            print(d[workload])
    print(f'{count} points in total')
    with open(f'{output}_cover{expected_coverage}.json', 'w') as f:
        json.dump(d, f, indent=4)



if __name__ == '__main__':
    tree = glob_weighted_stats(
            '/home51/zyy/expri_results/omegaflow_spec17/of_g1_perf/',
            single_stat_factory(t.ipc_target, 'cpi')
            )
    # gen_json('/home51/zyy/expri_results/nemu_take_simpoint_cpt_06',
    #         '/home51/zyy/expri_results/simpoints06')
