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
def concat_data(post_data, key_to_rm=["cv"], include_std=False, show_mm_scale=False):
    """ Load data & concat """
    # concat data into one large matrix, row: features, col: data points
    row_names = []
    col_names = []

    # Convince that summary_data and summary_data_var have exactly same structure (not the value of them)
    summary_data = post_data["summary_data"]
    summary_data_std = post_data["summary_data_var"]

    # get size
    key_test = "nr0np0"
    Nr, Nc = np.shape(summary_data[key_test]["chi"])[:2]
    ndim = 0
    is_passed = {k: False for k in key_to_rm}
    for key in summary_data[key_test].keys():
        if key in key_to_rm:
            is_passed[key] = True
            continue

        shape = np.shape(summary_data[key_test][key])
        if len(shape) == 3:
            ndim += shape[2]
        else:
            ndim += 1
    
    # check key passing
    flag_pass = True
    for k, v in is_passed.items():
        if not v:
            print("%s not skipped, check the input"%(k))
            flag_pass = False
    
    if not flag_pass:
        print("Existing key in data")
        print(post_data.keys())
        return None

    # align data
    flag_row = True
    row_names = []
    align_data = []
    for k1 in summary_data.keys():
        keys = summary_data[k1].keys()

        if include_std:
            points = np.zeros([2*ndim, Nr*Nc])
        else:
            points = np.zeros([ndim, Nr*Nc])

        n = 0
        for k2 in keys:
            if k2 in key_to_rm:
                continue

            data = summary_data[k1][k2]
            if include_std:
                data_var = summary_data_std[k1][k2]

            shape = np.shape(data)
            if len(shape) == 3:
                ndim_sub = shape[2]
                for i in range(ndim_sub):
                    points[n, :] = np.array(data[:,:,i]).flatten(order="C")
                    if include_std:
                        points[n+ndim, :] = np.array(data_var[:,:,i]).flatten(order="C")

                    n += 1
                    if flag_row:
                        row_names.append(k2+"(%d)"%(i))
                        if include_std:
                            row_names.append(k2+"_std"+"(%d)"%(i))
            else:
                points[n, :] = np.array(data).flatten(order="C")
                if include_std:
                    points[n+ndim, :] = np.array(data_var).flatten(order="C")

                n += 1
                if flag_row:
                    # row_names.append(k2)
                    row_names.append(k2)
                    if include_std:
                        row_names.append(k2+"_std")

        flag_row = False
        for nr in range(Nr):
            for nc in range(Nc):
                col_names.append([k1, nr, nc])
        
        if len(align_data) == 0:
            align_data = points
        else:
            align_data = np.concatenate([align_data, points], axis=1)

    if include_std:
        # realign row_names if include_var=True
        row_names = row_names[0::2] + row_names[1::2]
        # remove overlap data: leading_ratio, leading_ratio(abs), dphi
        overlap_keys = ("leading_ratio", "leading_ratio(abs)", "dphi")
        for k in overlap_keys:
            if k+"_std" in row_names:
                nid = row_names.index(k+"_std")
                align_data = np.delete(align_data, nid, 0)
                row_names.pop(nid)
                print("skipped overlap key:", k+"_std")
    
    # check dimension
    print("datadim:", align_data.shape, end=", ")
    print("nrow: %d, ncol: %d"%(len(row_names), len(col_names)))

#     # Post processing: inverse tau
    key_inds = [n for n, s in enumerate(row_names) if "tau" in s and "std" in s]
    print("correcting std names target keys", [row_names[n] for n in key_inds])
    for n in key_inds:
        row_names[n] = row_names[n].replace("1/", "")
    
    # print("inversing target keys:", [row_names[n] for n in key_inds])
    

#     for n in key_inds:
#         align_data[n, :] = 1/align_data[n, :]
#         row_names[n] = "1/%s"%(row_names[n])

#     # Post processing: min-max scaling
    xmin = np.min(align_data, axis=1)
    xmax = np.max(align_data, axis=1)

    if show_mm_scale:
        xavg = np.average(align_data, axis=1)
        plt.figure(dpi=100, figsize=(4, 4))
        plt.plot((xmax - xmin) / xavg, 'k.--', lw=1)
        plt.ylabel("(max(X) - min(X)) / mean(X)", fontsize=14)
        plt.xlabel("features", fontsize=14)
        plt.show()

    align_data = (align_data - xmin[:, np.newaxis]) / (xmax - xmin)[:, np.newaxis]

    return align_data, row_names, col_names


def default_colors(col_names):
    cs = []
    for tag in col_names:
        idr, idp = int(tag[0][2]), int(tag[0][5])
        cr = 1 if idr == 0 else 0
        cb = idp/6
        cs.append([cr, cb, 0])
    return cs


def reorder_data(data, id_cluster, sval=None):
    # sval is the silhouette scores
    cluster_types = np.unique(id_cluster)
    data2 = np.zeros_like(data)
    # for n in range(cluster_types):
    stack = 0
    bds = []
    id_sort = np.zeros(data.shape[1])
    seq = np.arange(data.shape[1])
    for nc in cluster_types:
        ind = np.squeeze(id_cluster == nc)
        num = sum(ind)

        data_sub = data[:, ind]
        seq_sub = seq[ind]
        if sval is not None:
            sval_sub = sval[ind]
            id_order = np.argsort(sval_sub)[::-1]
            data_sub = data_sub[:, id_order]
            seq_sub = seq_sub[id_order]

        data2[:, stack:stack+num] = data_sub
        id_sort[stack:stack+num] = seq_sub
        stack += num
        bds.append(stack)

    return data2, id_sort, bds


def get_palette(cmap="jet"):
    from matplotlib.cm import get_cmap
    return get_cmap(cmap)


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