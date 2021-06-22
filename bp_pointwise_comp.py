import utils as u
import utils.common as c
import utils.target_stats as t
import json
import sys
import os
from os.path import join as opj
import pandas as pd
import matplotlib.pyplot as plt

res_base = '/home51/glr/expri_results'
gem5_res_path = opj(res_base, 'SPEC06/FullWindowO3Config_2021-04-09_22:21:46')
xs_res_path = opj(res_base, 'xs_simpoint_batch/SPEC06_EmuTasksConfig-04-06-2021')

ver = '06'
def get_tree(prefix, simpoints, file_path, stat_file):
    target = eval(f't.{prefix}branch_misp')
    if prefix == '':
        branch_misp = ['cpus?\.(?:diewc|commit|iew)\.(branchMispredicts)']
    else:
        assert prefix == 'xs_'
        branch_misp = ['BpBWrong']
    tree = u.glob_weighted_stats(
        file_path,
        u.stats_factory(target, target, prefix),
        simpoints=simpoints,
        stat_file=stat_file
    )
    # print(tree)
    return tree

tree_gem5 = get_tree(
    prefix='',
    simpoints=f'/home51/zyy/expri_results/simpoints{ver}.json',
    file_path=gem5_res_path,
    stat_file='m5out/stats.txt')

tree_xs = get_tree(
    prefix='xs_',
    simpoints=f'/home51/zyy/expri_results/simpoints{ver}.json',
    file_path=xs_res_path,
    stat_file='simulator_err.txt')

tar_path = opj(res_base, 'bp_diff')
plot_path = opj(tar_path, 'png')
csv_path = opj(tar_path, 'csv')

def check_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

res_tree = {}
abnormal_points = {}
ratio_thres = 0.5
misp_abs_thres = 50000 # mpki 1 for 50M insts
dump_to_json = True

save_to_csv = False
save_to_png = False


        
for bench in tree_xs:
    for subbench in tree_xs[bench]:
        xs_subbench_stat = tree_xs[bench][subbench]
        if bench in tree_gem5.keys() and subbench in tree_gem5[bench].keys():
            gem5_subbench_stat = tree_gem5[bench][subbench]
        else:
            print(f'bench {bench} in stats: {bench in tree_gem5.keys()}')
            print(f'subbench {subbench} in stats: {subbench in tree_gem5[bench].keys()}')
            print('skip!')
            continue
        # print(gem5_subbench_stat.columns)
        # print(gem5_subbench_stat.shape)
        concatenated = pd.concat([xs_subbench_stat, gem5_subbench_stat], axis=1)
        concatenated.drop(['weight'], axis=1, inplace=True)
        res = concatenated.dropna(axis=0, how='any')
        res.columns=['xs', 'gem5']
        short_inds = [int(str(i)[:-7]) for i in res.index]
        res.index = short_inds
        print(res)

        # save to csv file
        if (save_to_csv):
            check_dir(csv_path)
            res.to_csv(opj(csv_path, subbench+'.csv'))

        # plot and save to png file
        if (save_to_png):
            res.plot(kind='bar', title=subbench)
            check_dir(plot_path)
            plt.savefig(opj(plot_path, subbench+'.png'))
        
        # check abnormal points and save to dict
        for i, r in res.iterrows():
            rl = list(r)
            ratio = rl[0] / rl[1]
            ratio_beyond_thres = ratio < ratio_thres or ratio > (1/ratio_thres)
            misp_abs_beyond_thres = rl[0] > misp_abs_thres or rl[1] > misp_abs_thres
            if ratio_beyond_thres and misp_abs_beyond_thres:
                point_name = subbench + '/' + str(i) + '0'*7
                abnormal_points[point_name] = {}
                abnormal_points[point_name]['xs_gem5_misp_ratio'] = ratio
                abnormal_points[point_name]['xs_mpki']   = rl[0] / 50000
                abnormal_points[point_name]['gem5_mpki'] = rl[1] / 50000


        if bench not in res_tree:
            res_tree[bench] = {}
        if subbench not in res_tree[bench]:
            res_tree[bench][subbench] = res

print(abnormal_points)
if (dump_to_json):
    json_path = opj(tar_path, 'abnormal_points.json')
    with open(json_path, 'w') as j:
        json.dump(abnormal_points, j, indent=4)