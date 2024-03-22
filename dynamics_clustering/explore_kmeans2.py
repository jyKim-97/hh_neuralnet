import numpy as np
import pickle as pkl
from sklearn import cluster
from numba import njit
from tqdm import tqdm


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


def main():
    # load dataset
    # with open("./data/align_data_sub.pkl", 'rb') as fp:
    with open("./data/align_data_sub.pkl", 'rb') as fp:
        align_data_sub = pkl.load(fp)
    data = align_data_sub["data"].copy()
    data = data[:-2]
    
    # run clustering
    np.random.seed(5000)
    N = data.shape[1]
    
    ksets = np.arange(3, 30)
    nmax = 100
    
    meta_info = []
    pred_labels = []
    
    for n in tqdm(range(len(ksets) * nmax), desc="kmeans clustering"):
        k = ksets[n//nmax]
        seed = np.random.randint(10000)
        
        meta_info.append({
            "k": k, "seed": seed
        })

        kobj = cluster.KMeans(n_clusters=k, n_init=1, copy_x=True, random_state=seed, init="k-means++")
        pred_labels.append(kobj.fit_predict(data.T))
            
    pred_labels = np.array(pred_labels)
    
    with open("./data/kmeans_pred_partial.pkl", "wb") as fp:
        pkl.dump({"labels": pred_labels, "meta_info": meta_info,
                  "nitr": nmax, "k": ksets}, fp)
        

if __name__ == "__main__":
    main()
