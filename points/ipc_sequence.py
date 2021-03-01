import os
import os.path as osp
import pandas as pd
import numpy as np
import re
import json
import utils as u
import utils.target_stats as t


stat_file = '/home51/zyy/expri_results/gem5_ooo_spec06_shotgun_dump_10000/gcc_200/140/m5out/stats.txt'
csv_file = '/home51/zyy/expri_results/gem5_ooo_spec06_shotgun_dump_10000/gcc_200/140/m5out/cpi.csv'

cpi_pattern = re.compile(r'system\.cpu\.cpi_total\s+(\d+\.\d+)')
inst_count_pattern = re.compile(r'/_(\d+)_\.gz')


def get_inst_start(config_json):
    with open(config_json) as f:
        js = json.load(f)
        cpt_name = js['system']['gcpt_file']
        m = inst_count_pattern.search(cpt_name)
        assert m is not None
        inst_count = m.group(1)
    return inst_count


def small_point_ipc_extract(stat_file, config_json):
    sim_insts = re.compile(r'sim_insts\s+(\d+)')

    inst_start = int(get_inst_start(config_json))
    cur_insts = 0
    d = {}
    with open(stat_file) as f:
        for line in f:
            m = sim_insts.match(line)
            if m is not None:
                cur_insts = round(int(m.group(1)), -4)

            m = cpi_pattern.match(line)
            if m is not None:
                cpi = float(m.group(1))
                assert cur_insts not in d
                d[inst_start + cur_insts] = cpi

    df = pd.DataFrame.from_dict(d, orient='index')
    return df


def large_point_ipc_extract(stat_file, config_json, workload=None):
    if workload is None:
        workload = ''
    else:
        workload = workload + '_'

    inst_count = get_inst_start(config_json)

    with open(stat_file) as f:
        for line in f:
            m = cpi_pattern.match(line)
            if m is not None:
                cpi = float(m.group(1))
                # label = f'{workload}{inst_count}'
                label = f'{inst_count}'
                return label, cpi
    return None

def multi_phase_labeled_ipc_extract(workload, workload_dir, large_point=True):
    d = {}
    dfs = []
    for fname in os.listdir(workload_dir):
        phase_path = osp.join(workload_dir, fname)
        if osp.isdir(phase_path):
            if large_point:
                pair = large_point_ipc_extract(
                        osp.join(phase_path, 'm5out', 'stats.txt'),
                        osp.join(phase_path, 'm5out', 'config.json'),
                        workload=workload,
                        )
                if pair is not None:
                    label, cpi = pair
                    assert label not in d
                    d[label] = cpi
                else:
                    print(f'{workload} {fname} yields None')
            else:
                d = u.c.get_stats(
                        osp.join(phase_path, 'm5out', 'stats.txt'),
                        t.ipc_target,
                        re_targets=True,
                        all_chunks=True,
                        config_file=osp.join(phase_path, 'm5out', 'config.json'),
                        )
                df = pd.DataFrame.from_dict(d, orient='index')
                df.sort_index(inplace=True)
                df = df.iloc[1:]
                dfs.append(df)

    if large_point:
        if len(d):
            df = pd.DataFrame.from_dict(d, orient='index')
            return df
        else:
            return None
    else:
        df = pd.concat(dfs)
        df.sort_index(inplace=True)
        return df

def multi_workload_labeled_ipc_extract(task_dir, large_point=True):
    for workload in os.listdir(task_dir):
        workload_path = osp.join(task_dir, workload)
        if osp.isdir(workload_path):
            if large_point:
                df = multi_phase_labeled_ipc_extract(workload, workload_path, True)
                df.to_csv(f'outputs/shotgun/{workload}.csv', header=None)
            else:
                df = multi_phase_labeled_ipc_extract(workload, workload_path, False)
                df.to_csv(f'outputs/shotgun_continuous_point/{workload}.csv')
    # df = pd.concat(dfs)
    # df.to_csv(f'outputs/shotgun/mixed.csv', header=None)


if __name__ == '__main__':
    # multi_workload_labeled_ipc_extract(stat_dir)
    # stat_dir = '/home51/zyy/expri_results/gem5_ooo_spec06_shotgun/'

    # stat_dir = '/home51/zyy/expri_results/shotgun/gem5_ooo_spec06_large_chunk'
    # multi_workload_labeled_ipc_extract(stat_dir, large_point=True)

    stat_dir = '/home51/zyy/expri_results/shotgun/gem5_shotgun_cont_06/FullWindowO3Config'
    multi_workload_labeled_ipc_extract(stat_dir, large_point=False)

