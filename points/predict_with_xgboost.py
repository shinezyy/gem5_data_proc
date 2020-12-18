import os.path as osp
import xgboost as xgb
import scipy.sparse as sparse
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer


def predict_npz_with_model(bbv_npz, xgb_model, id_limit, output_file):
    ids = []
    with open(id_limit) as f:
        for line in f:
            ids.append(int(line.strip()))
    class_map = {}
    centroids = []

    csc = sparse.load_npz(bbv_npz)
    csr = csc.tocsr()

    scaler = StandardScaler(with_mean=False)
    scaler.fit(csr)
    csr = scaler.transform(csr)

    m = xgb.Booster({'nthread': 56})
    m.load_model(xgb_model)

    for x in range(0, csr.shape[1]):
        if x not in ids or x < 20:
            continue

        if not len(centroids):
            centroids.append(x)
            class_map[x] = x
            print(f'New class: {x}')

        found_similar = False
        X = []
        for c in reversed(centroids):
            X.append(np.abs(csr[x] - csr[c]))

        X = sparse.vstack(X, format='csr')
        dpredict = xgb.DMatrix(X)
        y_predict = m.predict(dpredict)

        min_val = 1000
        min_index = -1
        for pred, c in zip(y_predict, reversed(centroids)):
            if pred < 0.05:
                min_val = pred
                min_index = c

        if min_index >= 0:
            found_similar = True
            print(f'{x} -> {min_index}')
            class_map[x] = class_map[min_index]

        if not found_similar:
            assert x not in centroids
            centroids.append(x)
            class_map[x] = x
            print(f'New class: {x}, num classes: {len(centroids)}')

    with open(output_file, 'w') as f:
        for k, v in class_map.items():
            print(k, v, file=f)


if __name__ == '__main__':
    predict_npz_with_model(
        osp.expanduser('~/projects/NEMU/outputs3/simpoint_profile_06/gcc_200/csc.dump.npz'),
        xgb_model='/home51/zyy/projects/betapoint/betapoint_trained_with_100points.xgb',
        id_limit="/home51/zyy/projects/gem5_data_proc/points/outputs/gcc_200/shotgun_idmap.txt",
        output_file='class_map_min.txt',
    )
