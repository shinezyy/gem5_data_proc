import re
import os
import os.path as osp
from datetime import datetime
import time
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import stats
from sklearn import metrics

def get_emu_run_time(emu_log, pattern):
    """Get the run time of the emulator from the log file.

    Args:
        emu_log: The emulator log file.

    Returns:
        The run time of the emulator in seconds.
    """
    time = 0
    with open(emu_log, 'r') as f:
        for line in f:
            m = re.search(pattern, line)
            if m:
                time = int(m.group(1).replace(',', ''))
                break
    return time


def get_func_warmup_time(log, pattern):
    mod_time = time.ctime(osp.getmtime(log))
    mod_time = datetime.strptime(mod_time, "%a %b %d %H:%M:%S %Y")
    print(mod_time)
    with open(log, 'r') as f:
        for line in f:
            m = re.match(pattern, line)
            if m:
                start_func_warmup_time = datetime.strptime(line.strip(), '%Y-%m-%d--%H:%M:%S')
                print(start_func_warmup_time)
                return (mod_time - start_func_warmup_time).total_seconds() * 1000
    return None

def simulate_rand(times: np.array, n_slots = 16):
    slots = []
    slot_end_time = []
    for i in range(n_slots):
        slots.append([])
        slot_end_time.append(0)
    np.random.shuffle(times)
    for i in range(len(times)):
        # pick the slot with the earliest end time
        slot_idx = np.argmin(slot_end_time)
        slots[slot_idx].append(times[i])
        slot_end_time[slot_idx] += times[i]
    # for row in slots:
    #     print(row)
    # print('Slot end time:', slot_end_time)
    max_finish_time = max(slot_end_time)
    # print('Max finish time of rand:', max(slot_end_time))
    return max_finish_time, np.mean(slot_end_time)


def simulate_ljf(times: pd.DataFrame, n_slots: int, pred_ipc_df: pd.DataFrame):
    slots = []
    slot_end_time = []
    assert len(times) == len(pred_ipc_df)
    for i in range(n_slots):
        slots.append([])
        slot_end_time.append(0)
    # times = sorted(times, reverse=True)
    # print(pred_ipc_df)
    pred_longest_indices = pred_ipc_df['time'].argsort()[::-1]
    true_longest_indices = times['time'].argsort()[::-1]
    # print(pred_longest_indices)
    # print(true_longest_indices)
    print("NDCG:", metrics.ndcg_score([true_longest_indices], [pred_longest_indices]))
    # raise

    time_descending = times.iloc[pred_longest_indices]
    time_descending = time_descending['time'].values
    # print(time_descending)

    for i in range(len(times)):
        # pick the slot with the earliest end time
        slot_idx = np.argmin(slot_end_time)
        # select predicted longest job here:
        slots[slot_idx].append(time_descending[i])
        slot_end_time[slot_idx] += time_descending[i]
    # for row in slots:
    #     print(row)
    print('Slot end time:', slot_end_time)
    max_finish_time = max(slot_end_time)
    # print('Max finish time of LJF:', max(slot_end_time))
    return max_finish_time, np.mean(slot_end_time)


def main():
    log_name = 'log.txt'
    emu_log_name = 'emu_out.txt'

    # low: oc114
    # dw=10
    # header = f'0+{dw}'
    # top_dir = f'/data/zhouyaoyang/exec-storage/compare-warmup-xs-8t-vs-gem5'
    # warmup_date_pattern = re.compile(r'2022-12-')
    # tag = f'0-{dw}-5'

    # # 25M: oc113
    # header = '0+25'
    # top_dir = f'/data/zhouyaoyang/exec-storage/compare-warmup-xs-8t-vs-gem5-25M'
    # warmup_date_pattern = re.compile(r'2022-12-')
    # tag = '0-25-5'

    # # oc115
    # header = '95+5'
    # top_dir = f'/data/zhouyaoyang/exec-storage/func-warmup-used-90M'
    # warmup_date_pattern = re.compile(r'2023-01-')
    # tag = '90-5-5'

    # # oc112
    header = 'ada'
    top_dir = f'/data/zhouyaoyang/exec-storage/ada-warmup-est-cycles'
    warmup_date_pattern = re.compile(r'2023-01-')
    tag = '-'  # all dirs are ada related

    # all
    # top_dir = '/nfs/home/share/EmuTasks/SPEC06_EmuTasks_10_18_2021'
    # emu_time_pattern = re.compile(r'Host time spent: ([\d,]+)ms')
    # warmup_date_pattern = None
    # tag = '_'  # all dirs are ada related
    # emu_log_name = 'simulator_out.txt'

    emu_time_pattern = re.compile(r'Host time spent: (\d+)ms')
    # emu_time_pattern = None

    emu_ms = 0
    emu_dict = {}
    warmup_dict = {}
    warmup_dict_short_name = {}
    warmup_ms = 0
    max_ms = 0
    warmup_times = []
    emu_times = []
    for sub_dir in os.listdir(top_dir):
        print(sub_dir)
        task = sub_dir.split('-')[0]
        # todo: guide ljf with gem5 ipc and true ipc
        if tag not in sub_dir:
            continue
        if osp.isfile(osp.join(top_dir, sub_dir)):
            continue
        if emu_time_pattern is not None:
            emu_log = osp.join(top_dir, sub_dir, emu_log_name)
            ms = get_emu_run_time(emu_log=emu_log, pattern=emu_time_pattern)
            emu_ms += ms
            emu_dict[sub_dir] = ms
            # emu_times.append(ms)

        if warmup_date_pattern is not None:
            warmup_log = osp.join(top_dir, sub_dir, log_name)
            ms = get_func_warmup_time(warmup_log, warmup_date_pattern)
            warmup_ms += ms
            warmup_dict[sub_dir] = ms
            warmup_dict_short_name[task] = ms
            warmup_times.append(ms)

        max_ms = max(max_ms, ms)
        if max_ms == ms:
            print(sub_dir)
    # times = np.array(warmup_times) + np.array(emu_times)

    
    sim_schedule = True
    output_to_csv = False
    time_predicted = {
        'simualtor': "/nfs-nvme/home/zhouyaoyang/gem5-results/gem5-ipc-for-ljf.csv",
        'xs': "/nfs-nvme/home/zhouyaoyang/gem5-results/xs-ipc-for-ljf.csv",
        # 'simualtor': "/nfs-nvme/home/zhouyaoyang/gem5-results/gem5-ipc-for-ljf-25M.csv",
        # 'xs': "/nfs-nvme/home/zhouyaoyang/gem5-results/xs-ipc-for-ljf-25M.csv",
    }
    if sim_schedule:
        times = pd.DataFrame.from_dict(warmup_dict_short_name, orient='index', columns=['time'])
        times_list = warmup_times
        for n_slots in (4, 8, 16):
            for source, prediction in time_predicted.items():
                pred_ipc_df = pd.read_csv(prediction, index_col=0)
                icount_df = pd.read_csv('results/ada-warmup.lst', index_col=0, sep=' ', header=None)
                icount_df.columns = ['bmk', 'skip', 'fw', 'dw', 'sample']
                icount_df['icount'] = icount_df['dw'] + icount_df['sample']
                icount_df = icount_df.sort_index()
                pred_ipc_df = pred_ipc_df.sort_index()
                times = times.sort_index()
                # print(pred_ipc_df['ipc'])
                # print(icount_df['icount'])
                pred_ipc_df['time'] = (icount_df['icount']).values / pred_ipc_df['ipc']
                # print(pred_time)
                rand_max_times = []
                rand_balance = []
                for i in range(30):
                    rand_max_time, rand_mean_finish_time = simulate_rand(times_list, n_slots)
                    rand_max_times.append(rand_max_time)
                    rand_balance.append(rand_mean_finish_time/rand_max_time)

                ljf_sched, ljf_mean_finish_time = simulate_ljf(times, n_slots, pred_ipc_df)
                ljf_balance = ljf_mean_finish_time / ljf_sched

                print('-' * 80)
                print(f'Prediction from {source} with #{n_slots} slots')
                print(f'Average max finish time of rand scheduling: {np.mean(rand_max_times)/1000/3600:0.2f} hour, balance:{np.mean(rand_balance):0.2f}')
                print(f'Max finish time of LJF scheduling: {ljf_sched/1000/3600:0.2f} hour, balance:{ljf_balance:0.2f}')
                print(f'Host time spent: {(emu_ms + warmup_ms)/1000/3600:0.2f} hour')
                print(f'max single task: {max_ms/1000/3600:0.2f} hour')
                print(f'LJF improvements over rand:{(1-ljf_sched/np.mean(rand_max_times))*100:0.2f}%')

    # df2 = pd.DataFrame.from_dict(emu_dict, orient='index')
    df = pd.DataFrame.from_dict(warmup_dict, orient='index')
    # df = pd.concat([df, df2], axis=1)
    # df.columns = [f'{header}_emu_time', f'{header}_warmup_time']
    df.columns = [f'{header}_total_time']
    # df.columns = [f'{header}_time']
    df.loc['sum', df.columns] = np.sum(df, axis=0)
    df.sort_index(inplace=True)
    df = df/1000/3600
    if output_to_csv:
        df.to_csv(osp.join('results', f'host_time_{header}.csv'))


if __name__ == '__main__':
    main()