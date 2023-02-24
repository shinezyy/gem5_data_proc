import json
import os.path as osp
import os
import utils as u
import pprint
import argparse


def gen_from_dir(top):
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
    config = top.split('/')[-1]
    new_file_name = f'simpoint_cpt/resources/{config}.json'
    with open(new_file_name, 'w') as f:
        json.dump(new_js, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='specify cpt top directory')
    parser.add_argument('-t', '--top', action='store', required=True)
    args = parser.parse_args()

    gen_from_dir(osp.realpath(args.top))
