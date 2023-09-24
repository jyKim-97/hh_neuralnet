import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import utils

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
def kmeans_specific_seed(K, data, seed):
    # data (features, sample_points)
    np.random.seed(seed)
    
    km_obj = KMeans(n_clusters=K, init="k-means++", n_init="auto")
    km_obj.fit(data.T)
    id_cluster = km_obj.predict(data.T)
    sval, scoeff = get_silhouette_scores(data, id_cluster)
    
    return km_obj, sval, scoeff


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
    palette = utils.get_palette(cmap)
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


# denoise image
def denoise_square_cluster(sq_cluster_id):
    sq_cluster_id = np.array(sq_cluster_id).copy()
    nr = sq_cluster_id.shape[0]
    nc = sq_cluster_id.shape[1]
    
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    cmax = int(np.max(sq_cluster_id))
    
    for i in range(nr):
        for j in range(nc):
            cid = sq_cluster_id[i, j]
            cid_exist = np.zeros(cmax+1)
            flag_r = True
            
            for d in dirs:
                i1 = i + d[0]
                j1 = j + d[1]
                if (i1<0) or (i1>nr-1) or (j1<0) or (j1>nc-1):
                    continue
                    
                cid_nn = sq_cluster_id[i1, j1]
                if cid == cid_nn:
                    flag_r = False
                    break
                
                cid_exist[int(cid_nn)] += 1
            
            # alter the cluster_id
            if flag_r:
                sq_cluster_id[i, j] = np.argmax(cid_exist)
            
    return sq_cluster_id


def gather_clusters(sq_cluster_id):
    from collections import deque
    nrow, ncol = sq_cluster_id.shape
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def is_in(nr, nc):
        return (nr1 >= 0) and (nr1 < nrow) and (nc1 >= 0) and (nc1 < ncol)

    points = deque([(0, 0)])
    id_island = np.zeros_like(sq_cluster_id) - 1
    iid = 0

    while len(points) > 0:
        nr, nc = points.popleft()

        if id_island[nr, nc] != -1:
            continue

        id_island[nr, nc] = iid
        cid = sq_cluster_id[nr, nc]
        pt_clust = deque([(nr, nc)])
        visited = np.zeros_like(sq_cluster_id, dtype=bool)

        while len(pt_clust) > 0:
            nr, nc = pt_clust.popleft()

            for d in dirs:
                nr1 = nr + d[0]
                nc1 = nc + d[1]

                if not is_in(nr1, nc1):
                    continue
                
                if visited[nr1, nc1]:
                    continue

                visited[nr1, nc1] = True
                if id_island[nr1, nc1] == -1:
                    if (sq_cluster_id[nr1, nc1] == cid):
                        id_island[nr1, nc1] = iid
                        pt_clust.append((nr1, nc1))
                    else:
                        points.append((nr1, nc1))
            
        iid += 1

    id_island = id_island.astype(int)

    max_id = np.max(id_island)
    cluster_island = [[] for _ in range(max_id+1)]
    for nr in range(nrow):
        for nc in range(ncol):
            cid = id_island[nr, nc]
            cluster_island[cid].append((nr, nc))
            
    return cluster_island


def construct_square_image(prefix, data, col_names, ld=15):
    # data (-1, N)
    if ld is None:
        pass

    im = np.zeros([ld, ld])
    id_col = []
    
    is_valid = False
    for n, cn in enumerate(col_names):
        if prefix in cn[0]:
            nr, nc = cn[1], cn[2]
            im[nr, nc] = data[n]
            is_valid = True
            id_col.append(n)
    
    if not is_valid:
        raise ValueError("%s does not exist in col_names"%(prefix))

    return im, np.array(id_col).astype(int)


def flat_square_image(sq_stacks, col_names):
    # col_names: [prefix, nr, nc] -> sq_stacks should have [nr, nc, ~prefix order] shape
    lc = len(col_names)
    sq_flat = np.zeros(lc)
    for n in range(lc):
        prefix, nr, nc = col_names[n]
        n0 = int(prefix[2])
        n1 = int(prefix[5])
        sq_flat[n] = sq_stacks[n0, n1, nr, nc]
    return sq_flat


def remove_cluster_island(sq_cluster_id_, nth=3):
    from collections import defaultdict

    nrow = sq_cluster_id_.shape[0]
    ncol = sq_cluster_id_.shape[1]

    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    def is_in(nr, nc):
        return (nr1 >= 0) and (nr1 < nrow) and (nc1 >= 0) and (nc1 < ncol)

    sq_cluster_id = sq_cluster_id_.copy()
    cluster_island = gather_clusters(sq_cluster_id_)

    for cid in cluster_island:
        if len(cid) < nth:
            
            nn_id = defaultdict(int)
            for nr, nc in cid:
                for d in dirs:
                    nr1 = nr + d[0]
                    nc1 = nc + d[1]

                    if not is_in(nr1, nc1):
                        continue
                    
                    if sq_cluster_id_[nr1, nc1] == sq_cluster_id_[nr, nc]:
                        continue

                    nn_id[sq_cluster_id_[nr1, nc1]] += 1
            
            # flip
            if len(list(nn_id.values())) == 0:
                return sq_cluster_id, cid
            
            nid = np.argmax(list(nn_id.values()))
            nn_max = list(nn_id.keys())[nid]

            for nr, nc in cid:
                sq_cluster_id[nr, nc] = nn_max
        
    return sq_cluster_id


def reorder_sq_cluster_id(sq_cluster_id, start_id=1):
    # reorder cluster id (row first)
    
    nrow, ncol = sq_cluster_id.shape[:2]
    ntail_org = sq_cluster_id.shape[2:]
    
    sq_cid_flat = sq_cluster_id.reshape(nrow, ncol, -1)
    ntail = 1 if len(sq_cluster_id.shape) == 2 else sq_cid_flat.shape[2]
    changed_id = {n: -1 for n in np.unique(sq_cluster_id)}
    
    cid = start_id
    old_id = 0
    
    reorder_id = np.zeros_like(sq_cid_flat) * np.nan
    for nt in range(ntail):
        for i in range(nrow):
            for j in range(ncol):
                old_id = sq_cid_flat[i, j, nt]
                if changed_id[old_id] != -1:
                    continue
                
                reorder_id[sq_cid_flat == old_id] = cid
                changed_id[int(old_id)] = cid
                cid += 1
    
    reorder_id = np.reshape(reorder_id, (nrow, ncol, *ntail_org))
    return reorder_id, changed_id


# def reorder_sq_cluster_id(sq_cluster_id, start_id=1):
#     # row first
#     reorder_id = np.ones_like(sq_cluster_id) * np.inf
#     nrow, ncol = sq_cluster_id.shape

#     changed_id = dict()
#     cid = start_id
#     old_id = -1
#     for i in range(nrow):
#         for j in range(ncol):
#             if (old_id == sq_cluster_id[i, j]) or (cid > reorder_id[i, j]):
#                 continue
            
#             old_id = sq_cluster_id[i, j]
#             reorder_id[sq_cluster_id == old_id] = cid
#             changed_id[int(old_id)] = cid
#             cid += 1
    
#     return reorder_id, changed_id


def extract_mean_val(data, cluster_id):
    # data: (n_features, n_points)
    max_c = np.max(cluster_id)+1
    prods_avg = np.zeros([data.shape[0], max_c])
    prods_std = np.zeros([data.shape[0], max_c])
    cids = np.unique(cluster_id)
    for nc in cids:
        v = data[:, cluster_id == nc]
        prods_avg[:, nc] = np.average(v, axis=1)
        prods_std[:, nc] = np.std(v, axis=1)
    return prods_avg, prods_std


def realign_cluster(cluster_id, col_names, num_r=2, num_w=7, ld=15, denoise=True, nth_remain=3):
    num_k = np.max(cluster_id) + 1
    rcluster_id = np.zeros_like(cluster_id)
    nid2r = np.zeros(num_k).astype(int)

    for nr in range(num_r):
        for nw in range(num_w):
            prefix = "nr%dnp%d"%(nr, nw)
            sq_cid, id_col = construct_square_image(prefix, cluster_id, col_names, ld=ld)

            # denoise
            if denoise:
                sq_cid = denoise_square_cluster(sq_cid)
                sq_cid = remove_cluster_island(sq_cid, nth=nth_remain)

            # reorder
            re_cid, changed_id = reorder_sq_cluster_id(sq_cid, start_id=1)
            
            rmap = re_cid.copy()
            for old_id, new_id in changed_id.items():
                if nid2r[old_id] == 0:
                    nid2r[old_id] = np.max(nid2r) + 1
                
                rmap[re_cid == new_id] = nid2r[old_id]
            
            rcluster_id[id_col] = rmap.flatten()

    return rcluster_id


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


def show_sq_cluster(sq_cluster, x=None, y=None, cmap="jet", cth=None, vmin=None, vmax=None, fontsize=14, aspect=None):

    x = np.arange(sq_cluster.shape[1]) if x is None else x
    y = np.arange(sq_cluster.shape[0]) if y is None else y
    vmin = np.min(sq_cluster) if vmin is None else vmin
    vmax = np.max(sq_cluster) if vmax is None else vmax

    dx = x[1] - x[0]; dy = y[1] - y[0]
    cax = plt.imshow(sq_cluster, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=[x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2],
                    aspect=aspect)
    
    cids = [c for c in np.unique(sq_cluster).astype(int) if c > 0]
    cth = (max(cids)+min(cids))/2 if cth is None else cth
    for cid in cids:
        nr, nc = np.where(sq_cluster == cid)
        xc = np.average(x[nc])
        yc = np.average(y[nr])
        c = "k" if cid > cth else "w"
        plt.text(xc, yc, "%d"%(cid), fontsize=fontsize, c=c)

    return cax


def print_largest_differ(cid0, cid1, prods, row_names, nprint=2):
    dp = prods[:, cid0] - prods[:, cid1]
    id_sort = np.argsort(np.abs(dp))[::-1]
    dp_sort = dp[id_sort]
    row_names_sort = [row_names[n] for n in id_sort]

    print("cluster %d-%d difference"%(cid0, cid1))
    for n in range(nprint):
        print("%d. %15s, diff = %6.3f"%(n+1, row_names_sort[n], dp_sort[n]))
    
    return dp_sort, row_names_sort


def regress_differ(cids, prods, row_names, nshow=2):
    N = len(cids)
    Nr = prods.shape[0]
    prods_sub = prods[:, np.array(cids).astype(int)]

    # regress
    x = np.arange(N)
    models = np.zeros([Nr, 2])
    for n in range(Nr):
        models[n, :] = np.polyfit(x, prods_sub[n, :], 1)
    
    # sort
    id_sort = np.argsort(np.abs(models[:, 0]))[::-1]
    colors = []
    for n in id_sort[:nshow]:
        lb = row_names[n] + "\np=%.3fx+%.3f"%(models[n, 0], models[n, 1])
        p_obj = plt.plot(x, prods_sub[n, :], '.', ms=10, label=lb)
        colors.append(p_obj[0].get_color())
        plt.plot(x, models[n, 0] * x + models[n, 1], '--', c=colors[-1], lw=1)

    # for i, nid in enumerate(id_sort[:nshow]):
    #     nc = 0 if models[nid, 0] > 0 else -1
    #     xsub = x[nc]+0.01 if models[i, 0] > 0 else x[nc]-0.4
    #     ysub = prods_sub[nid, nc]+0.04
        # plt.text(xsub, ysub, "a=%.3fx+%.3f"%(models[nid, 0], models[nid, 1]), 
        #         color=colors[i], va="center")

    plt.xticks(x, labels=["%s"%(c) for c in cids])
    plt.xlabel("Cluster id", fontsize=14)
    
    return models

# ----------------------------- Temporal function, move to another lib

def draw_quadratic_summary(data, fname=None, xl_raster=(1500, 2500), nsamples=20, wbin_t=1, fs=2000, shuffle=False):
    
    from tqdm.notebook import tqdm
    from scipy.ndimage import gaussian_filter1d

    import sys
    sys.path.append("../../include/")
    import hhtools
    import hhsignal
    
    teq = 0.5
    
    fig = plt.figure(dpi=100, figsize=(9, 12))
    plt.axes([0.1, 0.8, 0.8, 0.15])
    
    seq = np.arange(len(data["step_spk"]))
    cr = [800, 1000, 1800, 2000]
    cs = ["r", "b", "deeppink", "navy"]
    
    if shuffle:
        np.random.shuffle(seq)
        cr = None
        cs = None
        
    
    hhtools.draw_spk(data["step_spk"], color_ranges=cr, colors=cs, xl=xl_raster, sequence=seq)
    plt.ylabel("# neuron", fontsize=14)
    plt.xlabel("Time (ms)", fontsize=14)
    
    title = "nid: %d"%(data["nid"][0])
    for n in data["nid"][1:]:
        title += ",%d"%(n)
    plt.title(title, fontsize=14)

    plt.twinx()
    t = data["ts"] * 1e3
    plt.plot(t, data["vlfp"][0], c='k', zorder=10, label=r"$V_T$")
    plt.plot(t, data["vlfp"][1], c='b', lw=1, label=r"$V_F$")
    plt.plot(t, data["vlfp"][2], c='r', lw=1, label=r"$V_S$")
    plt.legend(fontsize=14, loc="upper left", ncol=3, edgecolor="none")
    plt.ylabel("V", fontsize=14)
    
    # ----------------- Generate AC for x -----------------#
    t0_set = np.random.uniform(low=teq, high=data["ts"][-1]-wbin_t, size=nsamples)
    cc_set = [[] for _ in range(4)]
    for t0 in tqdm(t0_set):
        n0 = int(t0 * fs)
        n1 = n0 + wbin_t * fs

        for i in range(4):
            if i < 3:
                x = data["vlfp"][i][n0:n1]
                y = x.copy()
            else:
                x = data["vlfp"][1][n0:n1]
                y = data["vlfp"][2][n0:n1]

            cc, tlag = hhsignal.get_correlation(x, y, fs, max_lag=0.1)
            cc_set[i].append(cc)

    cc_set_avg = np.average(cc_set, axis=1)
    cc_set_std = np.std(cc_set, axis=1)
    
    # ----------------- Draw AC for x -----------------#
    labels = ["AC(T)", "AC(F)", "AC(S)", "CC(F, S)"]
    yl = [-0.8, 1.1]

    for n in range(4):
        # plt.subplot(1, 4, n+1)
        plt.axes([0.1+0.22*n, 0.6, 0.15, 0.15])
        plt.plot([0, 0], yl, 'g--', lw=1)
        plt.plot([-0.1, 0.1], [0, 0], 'g--', lw=1)
        plt.plot(tlag, cc_set_avg[n], c='k')
        plt.fill_between(tlag, cc_set_avg[n]-cc_set_std[n]/2, cc_set_avg[n]+cc_set_std[n]/2, alpha=0.5, color='k', edgecolor="none")
        plt.xlim([-0.1, 0.1])
        plt.xlabel(r"$\Delta t$ (s)", fontsize=14)
        plt.title(labels[n], fontsize=14)
        plt.ylim(yl)
        if n == 3:
            plt.text(-0.06, -0.7, r"$V_F$ lead", horizontalalignment="center")
            plt.text(0.06, -0.7, r"$V_S$ lead", horizontalalignment="center")
        
    # ----------------- Draw figure -----------------#
    tags = ["T", "F", "S"]
    xt = np.arange(0.5, 4.1, 0.5)
    fl = [20, 80]
    
    for nid in range(3):

        psd, ff, tf = hhsignal.get_stfft(data["vlfp"][nid], data["ts"], 2000, frange=[2, 100])
        yf, f = hhsignal.get_fft(data["vlfp"][nid], 2000, frange=[2, 100])

        plt.axes([0.1, 0.4-0.17*nid, 0.7, 0.15])
        hhtools.imshow_xy(psd, x=tf, y=ff, cmap="jet", interpolation="spline16")
        plt.ylabel("frequency (%s) (Hz)"%(tags[nid]), fontsize=14)
        plt.ylim(fl)
        plt.colorbar()

        if nid < 2:
            plt.xticks(xt, labels=["" for _ in xt])
        else:
            plt.xticks(xt)
            plt.xlabel("Time (s)", fontsize=14)

        plt.axes([0.8, 0.4-0.17*nid, 0.11, 0.15])
        yf_s = gaussian_filter1d(yf, 3)
        plt.plot(yf, f, c='k')
        plt.plot(yf_s, f, c='r', lw=1.5)
        plt.xlabel(r"FFT($V_{%s}$)"%(tags[nid]), fontsize=14)
        plt.ylim(fl)
        
    if fname is not None:
        plt.savefig(fname, dpi=150)
    
    return fig



def show_sample_cases(obj, target_cluster_id, cluster_id, silhouette_vals, col_names, case="best", nshow=2, save=False, fdir="./sample_figs"):
    import os

    if case not in ("best", "intermediate", "worst"):
        raise ValueError("input 'case' must be in ('best', 'intermediate', 'worst')")
    
    in_target = target_cluster_id == cluster_id
    id_sort = np.argsort(silhouette_vals[in_target]) # ascending way
    id_target = np.where(in_target)[0][id_sort].astype(int)
    
    if case == "best":
        nid = id_target[-nshow:]
    elif case == "intermediate":
        nhalf = len(id_target)//2
        n0 = nhalf-nshow//2
        n1 = nhalf+nshow//2
        if n1-n0+1 > nshow:
            n0 += 1
        nid = id_target[n0:n1+1]
    else: # "worst"
        nid = id_target[:nshow]
        
    tags = [col_names[n] for n in nid]
    
    for i, tag in enumerate(tags):
        nrow, ncol = tag[1], tag[2]
        nr = int(tag[0][2])
        nw = int(tag[0][5])
        
        fname = None
        if save:
            fname = os.path.join(fdir, "cid%d_%s(%d).png"%(target_cluster_id, case, i))

        data_sub = obj.load_detail(nrow, ncol, nr, nw, 0)
        fig = draw_quadratic_summary(data_sub, fname=fname)
        fig.close()
    
    return tags