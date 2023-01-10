import re
import os
import os.path as osp
from datetime import datetime
import time
import pandas as pd

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


def main():
    log_name = 'log.txt'
    emu_log_name = 'emu_out.txt'
    # # low: oc114
    # # 25M: oc113
    # top_dir = f'/data/zhouyaoyang/exec-storage/compare-warmup-xs-8t-vs-gem5'
    # warmup_date_pattern = re.compile(r'2022-12-')
    # tag = '0-5-5'

    # # oc115
    # top_dir = f'/data/zhouyaoyang/exec-storage/func-warmup-used-90M'
    # warmup_date_pattern = re.compile(r'2023-01-')
    # tag = '90-5-5'

    # # oc112
    # top_dir = f'/data/zhouyaoyang/exec-storage/ada-warmup-est-cycles'
    # warmup_date_pattern = re.compile(r'2023-01-')
    # tag = '-'  # all dirs are ada related

    # all
    top_dir = '/nfs/home/share/EmuTasks/SPEC06_EmuTasks_10_18_2021'
    emu_time_pattern = re.compile(r'Host time spent: ([\d,]+)ms')
    warmup_date_pattern = None
    tag = '_'  # all dirs are ada related
    emu_log_name = 'simulator_out.txt'

    # emu_time_pattern = re.compile(r'Host time spent: (\d+)ms')
    # emu_time_pattern = None

    emu_ms = 0
    ms_dict = {}
    warmup_ms = 0
    max_ms = 0
    for sub_dir in os.listdir(top_dir):
        if tag not in sub_dir:
            continue
        if osp.isfile(osp.join(top_dir, sub_dir)):
            continue
        if emu_time_pattern is not None:
            emu_log = osp.join(top_dir, sub_dir, emu_log_name)
            ms = get_emu_run_time(emu_log=emu_log, pattern=emu_time_pattern)
            emu_ms += ms
            ms_dict[sub_dir] = ms

        if warmup_date_pattern is not None:
            warmup_log = osp.join(top_dir, sub_dir, log_name)
            ms = get_func_warmup_time(warmup_log, warmup_date_pattern)
            warmup_ms += ms
        max_ms = max(max_ms, ms)
        if max_ms == ms:
            print(sub_dir)
    print(f'Host time spent: {(emu_ms + warmup_ms)/1000/3600:0.2f} hour')
    print(f'max single task: {max_ms/1000/3600:0.2f} hour')

    pd.DataFrame.from_dict(ms_dict, orient='index').to_csv(osp.join('results', 'host_time_full.csv'))


if __name__ == '__main__':
    main()