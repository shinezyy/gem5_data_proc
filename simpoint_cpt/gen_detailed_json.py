import json
import os.path as osp
import os
import utils as u
import pprint
import argparse
from glob import glob
import re


def gen_from_cpt_name_and_profiling_log(workload_list, log_pattern, cpt_pattern, cpt_path_pattern, config_name):
    with open(workload_list) as f:
        workloads = f.read().splitlines()
    print(workloads)
    pat = re.compile(cpt_pattern)
    desc = {}
    for workload in workloads:
        wl_desc = {
            'points': {},
        }
        for f in glob(cpt_path_pattern.format(workload)):
            m = pat.search(f)
            if m is None:
                continue
            point = m.group(1)
            weight = m.group(2)
            wl_desc['points'][point] = weight

        wl_desc['insts'] = u.get_insts(log_pattern.format(workload))
        desc[workload] = wl_desc
    new_file_name = f'simpoint_cpt/resources/{config_name}.json'
    print(desc)
    with open(new_file_name, 'w') as f:
        json.dump(desc, f, indent=4)


def gen_from_dir(top, config_name=None):
    json_path = osp.join(top, 'json/simpoint_summary.json')
    if not osp.isfile(json_path):
        json_path = osp.join(top, 'simpoint_summary.json')
    print(json_path)
    assert osp.isfile(json_path)

    with open(json_path) as f:
        js = dict(json.load(f))
    new_js = {}
    for workload in js:
        log_path = osp.join(top, f'logs/profiling/{workload}.log')
        if not osp.isfile(log_path):
            log_path = osp.join(top, f'profiling/{workload}/nemu_out.txt')
        print(log_path)
        assert osp.isfile(log_path)

        new_js[workload] = {}
        new_js[workload]['insts'] = u.get_insts(log_path)
        new_js[workload]['points'] = js[workload]
    if config_name is None:
        config = top.split('/')[-1]
    else:
        config = config_name
    new_file_name = f'simpoint_cpt/resources/{config}.json'
    with open(new_file_name, 'w') as f:
        json.dump(new_js, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='specify cpt top directory')
    parser.add_argument('-t', '--top', action='store')
    parser.add_argument('--from-cpt-name', action='store_true')
    parser.add_argument('-w', '--workload-list')
    parser.add_argument('-c', '--config')
    parser.add_argument('--cpt-pattern', default=r'_(\d+)_(.+)_\.gz')
    parser.add_argument('--cpt-path-pattern',
                        default='/nfs-nvme/home/zhouyaoyang/projects/rv-linux/NEMU-23/cpt_dir/sp_profile/{}/*/*.gz')
    parser.add_argument('-l', '--log-pattern',
                        default='/nfs-nvme/home/zhouyaoyang/projects/rv-linux/NEMU-23/profiling/{}-prof-out.log')
    args = parser.parse_args()

    if not args.from_cpt_name:
        assert args.top is not None
        gen_from_dir(osp.realpath(args.top), args.config)
    else:
        assert args.workload_list is not None
        assert args.config is not None
        gen_from_cpt_name_and_profiling_log(args.workload_list, args.log_pattern,
                                            args.cpt_pattern, args.cpt_path_pattern, args.config)
