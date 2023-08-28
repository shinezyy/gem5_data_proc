import json
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--weight-json', help='json file to sort')
parser.add_argument('--csv1', help='csv file 1 to sort')
parser.add_argument('--csv2', help='csv file 2 to sort')

args = parser.parse_args()

js = json.load(open(args.weight_json))
points = [
    pd.read_csv(args.csv1, index_col=0),
    pd.read_csv(args.csv2, index_col=0),
]

def get_related_points_on_wl(workload, points):
    selected = [x for x in points.index if x.startswith(workload)]
    return points.loc[selected]

def add_cpi(points):
    points['cpi'] = 1.0 / points['ipc']

def add_weighted_cycles(points_df, js, wokload):
    for point in js[wokload]['points']:
        wl_point = f'{wokload}_{point}'
        points_df.loc[wl_point, 'weighted_cycles'] = \
            float(js[wokload]['points'][point]) * points_df.loc[wl_point, 'cpi']
    
weighted_cycles = []
for i in range(2):
    add_cpi(points[i])
    add_weighted_cycles(points[i], js, 'mcf')
    df = get_related_points_on_wl('mcf', points[i])
    print(df)
    weighted_cycles.append(df['weighted_cycles'])
diff = weighted_cycles[0] - weighted_cycles[1]
print(diff)
