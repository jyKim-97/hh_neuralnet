import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from sklearn.cluster import KMeans
from tqdm import tqdm
import os

import utils
import hhclustering as hc


def load_data():    
    fname = "../simulation_data/purified_data.pkl"
    with open(fname, "rb") as fp:
        return pkl.load(fp)


def main(seed=100, nitr=100):
    # get seed
    np.random.seed(seed)
    seeds = np.random.randint(low=1, high=10000, size=nitr).astype(int)

    # get cluster 
    pdata = load_data()
    num_clusters = np.arange(2, 20)

    def get_empty_list():
        return [[] for _ in num_clusters]

    # run
    km_obj_sets = get_empty_list()
    sval_sets = get_empty_list()
    kcoeff_sets = get_empty_list()
    scoeff_sets = get_empty_list()

    for n in tqdm(range(len(num_clusters))):
        nc = num_clusters[n]
        for seed in seeds:
            kobj, sval, scoeff = hc.kmeans_specific_seed(nc, pdata["data"].copy(), seed)

            km_obj_sets[n].append(kobj)
            sval_sets[n].append(sval)
            kcoeff_sets[n].append(kobj.inertia_)
            scoeff_sets[n].append(scoeff)
    kcoeff_sets = np.array(kcoeff_sets)
    scoeff_sets = np.array(scoeff_sets)

    # save
    fname = "./data/kmeans_explore.pkl"
    with open(fname, "wb") as fp:
        pkl.dump({"km_obj_sets": km_obj_sets,
                  "sval_sets": sval_sets,
                  "kcoeff_sets": kcoeff_sets,
                  "scoeff_sets": scoeff_sets}, fp)

    print(fname)


if __name__ == "__main__":
    main(seed=100, nitr=100)
