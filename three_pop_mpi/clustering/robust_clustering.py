import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
from tqdm import tqdm

import utils
import hhclustering as hc
import argparse


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fout", required=True, help="output file name")
    parser.add_argument("--fdata", required=True, help="input data file name")
    parser.add_argument("-K", required=True, type=int, help="size of clusters")
    return parser


def load_saved(fname):
    import os
    import pickle as pkl
    
    fdir = "../simulation_data"
    with open(os.path.join(fdir, fname), "rb") as fp:
        return pkl.load(fp)


def kmeans_multi(K, data, seed):
    np.random.seed(seed)
    
    km_obj = KMeans(n_clusters=K, init="k-means++", n_init="auto")
    km_obj.fit(data.T)
    id_cluster = km_obj.predict(data.T)
    sval, scoeff = hc.get_silhouette_scores(data, id_cluster)
    
    return km_obj, sval, scoeff


def main(fout=None, fdata=None, K=None):
    nitr = 1000

    pdata = load_saved(fdata)

    np.random.seed(100) # gen random seeds
    seeds = np.random.randint(low=1, high=10000, size=nitr)

    km_objs = []
    svals = []
    scoeffs = []

    N = pdata["data"].shape[1]
    same_prob = np.zeros([N, N])
    for n in tqdm(range(nitr)):
        km_obj, sval, sc = kmeans_multi(K, pdata["data"], seeds[n])
        cluster_id = km_obj.predict(pdata["data"].T)

        for nk in range(K):
            id_c = np.where(cluster_id == nk)[0]
            for i, id0 in enumerate(id_c):
                for id1 in id_c[i+1:]:
                    same_prob[id0, id1] += 1
                    same_prob[id1, id0] += 1
        
        km_objs.append(km_obj)
        svals.append(sval)
        scoeffs.append(sc)
    same_prob /= nitr

    with open(fout, "wb") as fp:
        pkl.dump({"km_objs": km_objs,
                "same_prob": same_prob,
                "svals": svals,
                "scoeff": scoeffs,
                "seeds": seeds}, fp)


if __name__ == "__main__":
    main(**vars(build_args().parse_args()))