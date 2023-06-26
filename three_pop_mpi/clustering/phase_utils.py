import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# ================================================================================================
# Dimension reduction
# ================================================================================================
def pca(data):
    # data (features X data)
    # normalize to 0 mean and 1 std
    pca_data = data - np.average(data, axis=1)[:, np.newaxis]
    pca_data = pca_data / np.std(pca_data, axis=1)[:, np.newaxis]
    cov_mat = np.matmul(pca_data, pca_data.T)/pca_data.shape[1]

    # get eigval
    eigval, eigvec = np.linalg.eig(cov_mat)
    ind = np.argsort(eigval)[::-1]
    eigval = eigval[ind].real
    eigvec = eigvec[:, ind].real
    
    return eigval, eigvec, pca_data, cov_mat


# Test, X = WH
# X: D x N / W: D x K / H: K x N, where D is the # of features, N is the # of datum, and K is the reducted dimension
# use one of the simplest method

def nnmf_simple(data, n_features=2, nre=5, nitr=50, n_repeat=10, tol=1e-4):

    def get_mat_df(mat1, mat2):
        return np.sqrt(np.sum((mat1 - mat2)**2))
    
    def select_w(data):
        nq = 5
        norm = np.sqrt(np.sum(data**2, axis=0))
        p = norm / np.sum(norm)
        W = np.zeros([data.shape[0], n_features])
        for n in range(n_features):
            ind = np.random.choice(len(p), size=nq, replace=False, p=p)
            W[:, n] = np.average(data[:, ind], axis=1)
        return W

    # (data) = WH
    def optimize_nnmf(data):
        # W = np.random.uniform(size=(data.shape[0], n_features))
        W = select_w(data)
        H = np.random.uniform(size=(n_features, data.shape[1]))

        df_min = np.inf
        df_stack = 0
        df_hist = np.ones(nitr) * (-1)
        Wopt, Hopt = None, None
        for n in range(nitr):
            grad_h = np.dot(W.T, data) / np.dot(np.dot(W.T, W), H)
            grad_w = np.dot(data, H.T) / np.dot(np.dot(W, H), H.T)
            H = H * grad_h
            W = W * grad_w
            
            # check
            data_r = np.dot(W, H)
            df_hist[n] = get_mat_df(data, data_r)
            if df_hist[n] < df_min-tol:
                df_min = df_hist[n]
                Wopt = W.copy()
                Hopt = H.copy()
                df_stack = 0
            else:
                df_stack += 1
            
            if df_stack == n_repeat:
                break

        return Wopt, Hopt, df_hist, df_min
    
    df_min = np.inf
    for _ in range(nre):
        w_tmp, h_tmp, df_tmp, df_min_tmp = optimize_nnmf(data)
        if df_min_tmp < df_min:
            df_min = df_min_tmp
            Wopt = w_tmp.copy()
            Hopt = h_tmp.copy()
            df_hist = df_tmp.copy()
    
    # normalize
    wc = np.sqrt(np.sum(Wopt**2, axis=0))
    Wopt = Wopt / wc[np.newaxis, :]
    Hopt = Hopt * wc[:, np.newaxis]

    return Wopt, Hopt, df_hist


# ================================================================================================
# Clustering
# ================================================================================================

def get_silhouette_scores(data, cluster_labels):
    # get distance between abitrary points
    # row (features), col (sample points)

    def get_distance(X):
        # X (n_features, n_points)
        x0 = np.transpose(X[:,:,np.newaxis], [1, 2, 0])
        x1 = np.transpose(X[:,:,np.newaxis], [2, 1, 0])
        return np.sqrt(np.sum((x1 - x0)**2, axis=2))

    cluster_labels = np.squeeze(cluster_labels)
    d = get_distance(data)

    # align clusters
    cluster_types = [n for n in np.unique(cluster_labels) if -1 != n]
    num_cluster_types = len(cluster_types)

    d_clusters = dict()
    num_clusters = dict()
    for nt in cluster_types:
        nid = cluster_labels == nt
        d_clusters[nt] = d[:, nid]
        num_clusters[nt] = np.sum(nid)
        # d_clusters.append(d[:, nid])
        # num_clusters.append(np.sum(nid))

    # get silhouette scores
    num_points = data.shape[1]
    silhouette_scores = np.zeros(num_points)
    for n in range(num_points):
        nt0 = cluster_labels[n]
        dsub = {n:0 for n in cluster_types}
        for nt1 in cluster_types:
            if nt0 == nt1:
                norm = num_clusters[nt1] - 1
            else:
                norm = num_clusters[nt1]

            dsub[nt1] = np.sum(d_clusters[nt1][n, :]) / norm

        ai = dsub[nt0] # calculate ai
        bi = np.inf
        for k, v in dsub.items():
            if k == nt0:
                continue
            else:
                if bi > v:
                    bi = v

        silhouette_scores[n] = (bi - ai) / max(ai, bi)

    # get silhouette coef
    silhouette_avg = [np.average(silhouette_scores[cluster_labels == nt]) for nt in cluster_types]
    silhouette_coef = np.max(silhouette_avg)

    return silhouette_scores, silhouette_coef
    

# ================================================================================================
# Utility functions
# ================================================================================================



def draw_silhouette(sval, cluster_labels, scoeff=None, cmap="jet"):
    palette = get_palette(cmap)
    cluster_labels = np.squeeze(cluster_labels)
    cluster_types = [n for n in np.unique(cluster_labels) if -1 != n]
    num_cluster_types = len(cluster_types)

    n0 = 0
    for n, nt in enumerate(cluster_types):
        nid = cluster_labels == nt
        sval_sub = np.sort(sval[nid])
        id_bar = np.arange(np.sum(nid)) + n0

        plt.barh(id_bar, sval_sub, color=palette(n/num_cluster_types), label="cid%d"%(nt))
        n0 += np.sum(nid)

    yl = plt.ylim()
    plt.plot([0]*2, yl, 'k--', lw=1, alpha=0.5)
    if scoeff is not None:
        plt.plot([scoeff]*2, yl, 'r--', lw=1., alpha=0.8, label=r"$s_{coef}$")
    plt.legend(fontsize=10, edgecolor="none")
    
    xl = plt.xlim()
    plt.xlim([xl[0], 1])
    plt.ylim([0, len(sval)])
    plt.yticks([])
    plt.xlabel("Silhouette scores", fontsize=14)


def hsmooth(arr, wsize=10, fo=2):
    """ smooth array through hortizontal direction (for each row)"""
    
    from scipy.signal import savgol_filter
    
    s_arr = np.zeros_like(arr)
    for n in range(s_arr.shape[0]):
        s_arr[n] = savgol_filter(arr[n], wsize, fo)
    return s_arr


# clustering
