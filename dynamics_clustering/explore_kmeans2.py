import numpy as np
import pickle as pkl
from sklearn import cluster
from numba import njit
from tqdm import tqdm
import argparse
from sklearn.metrics import silhouette_score

"""
Run K-Means clustering iteratively

"""


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdata", required=True, help="clustering feature file (.pkl)")
    parser.add_argument("--fout", required=True, help="output file name (.pkl)")
    parser.add_argument("--kmin", default=3,  help="minimum number of clusters", type=int)
    parser.add_argument("--kmax", default=15, help="maximum number of clusters", type=int)
    parser.add_argument("--nitr", default=100, help="The number of iterations", type=int)
    return parser


@njit
def accumulate_evidence(pred_set):
    npoint = pred_set.shape[1]
    N = pred_set.shape[0]
    
    cmat = np.zeros((npoint, npoint))
    for i in range(npoint):
        cmat[i, i] = 1
        for j in range(i+1, npoint):
            for n in range(N):
                if pred_set[n, i] == pred_set[n, j]:
                    cmat[i, j] += 1/N
            cmat[j, i] = cmat[i, j]
    
    return cmat


def main(fdata=None, fout=None, kmin=3, kmax=15, nitr=100):
    # load dataset
    with open(fdata, 'rb') as fp:
        align_data_sub = pkl.load(fp)
    data = align_data_sub["data"].copy()
    data = data[:-2]
    
    # run clustering
    # np.random.seed(5000)
    np.random.seed(42)
    N = data.shape[1]
    
    ksets = np.arange(kmin, kmax)
    
    meta_info = []
    pred_labels = []
    kscores = [] # k-means inertia value
    sscores = [] # silhouette scores
    
    for n in tqdm(range(len(ksets) * nitr), desc="kmeans clustering"):
        k = ksets[n//nitr]
        seed = np.random.randint(10000)
        
        meta_info.append({
            "k": k, "seed": seed
        })

        kobj = cluster.KMeans(n_clusters=k, n_init=1, copy_x=True, random_state=seed, init="k-means++")
        pred_id = kobj.fit_predict(data.T)
        ks = kobj.score(data.T)
        ss = silhouette_score(data.T.copy(), pred_id, sample_size=1000, metric="euclidean")
        
        pred_labels.append(pred_id)
        kscores.append(ks)
        sscores.append(ss)
    
    with open(fout, "wb") as fp:
        pkl.dump({"labels": np.array(pred_labels),
                  "inertia": -np.array(kscores), # convert sign from - to +
                  "silhouette": np.array(sscores),
                  "meta_info": meta_info,
                  "nitr": nitr, "k": ksets,
                  "fdata": fdata}, fp)
        

if __name__ == "__main__":
    # main()
    main(**vars(build_parser().parse_args()))
