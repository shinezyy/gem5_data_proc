import json
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--weight-json', help='json file to sort')
parser.add_argument('--csv1', help='csv file 1 to sort')
parser.add_argument('--csv2', help='csv file 2 to sort')
parser.add_argument('-b', '--bmk', help='csv file 2 to sort')

args = parser.parse_args()

js = json.load(open(args.weight_json))
df1 = pd.read_csv(args.csv1, index_col=0)
df2 = pd.read_csv(args.csv2, index_col=0)
intersection = set(list(df1.index)).intersection(set(list(df2.index)))
points = [
    df1.loc[list(intersection)],
    df2.loc[list(intersection)],
]

def get_related_points_on_wl(workload, points):
    selected = [x for x in points.index if x.startswith(workload)]
    return points.loc[selected]

def add_cpi(points):
    points['cpi'] = 1.0 / points['ipc']

def add_weighted_cycles(points_df, js, wokload):
    print(js[wokload])
    for point in list(points_df['point']):
        wl_point = f'{wokload}_{point}'
        points_df.loc[wl_point, 'weighted_cycles'] = \
            float(js[wokload]['points'][str(point)]) * \
                points_df.loc[wl_point, 'cpi']

weighted_cycles = []
for i in range(2):
    add_cpi(points[i])
    print(points[i])
    add_weighted_cycles(points[i], js, args.bmk)
    df = get_related_points_on_wl(args.bmk, points[i])
    weighted_cycles.append(df['weighted_cycles'])

diff = pd.DataFrame((weighted_cycles[0] - weighted_cycles[1])*100)
diff['csv1 ipc'] = points[0]['ipc']
diff['csv2 ipc'] = points[1]['ipc']

diff = diff.sort_values(by=['weighted_cycles'], ascending=False)
print(diff)

