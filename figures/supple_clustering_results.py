import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from numba import njit

fdir = "/home/jungyoung/Project_win/hh_neuralnet/"

import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/pytools')
sys.setrecursionlimit(10000)

import utils_fig as uf
import figure_manager as fm
import hhclustering as hc

uf.set_plt()

# path_pred_data = "../clustering/data/kmeans_sub.pkl"
xl = (2, 15)
nopt = 4
fm.track_global("nopt", nopt)

reset = True

@fm.figure_renderer("kmeans_clustering", reset=reset, exts=[".png", ".svg"])
def show_kmeans_result(figsize=(12, 4), path_pred_data="../dynamics_clustering/data/kmeans_sub.pkl"):
    with open(path_pred_data, "rb") as f:
        pred_data = pkl.load(f)
    
    
    fig = uf.get_figure(figsize)
    
    # K-means result
    inertia = pred_data["inertia"].reshape([-1, pred_data["nitr"]]) / pred_data["labels"].shape[1]
    x = pred_data["k"]
    m = inertia.mean(axis=1)
    s = inertia.std(axis=1)
    
    # plt.subplot(121)
    plt.axes(position=(0.05, 0.05, 0.35, 0.9))
    plt.plot(x, m, 'k.-')
    plt.fill_between(x, m+s, m-s, color='k', edgecolor="none", alpha=0.2)
    plt.plot(x[nopt], m[nopt], "rp")
    plt.ylabel("inertia / (# of samples)")
    plt.xlabel("Number of clusters (K)")
    plt.xlim(xl)
    plt.xticks(np.arange(x[0], x[-1], 2))
    plt.ylim((4, 11))
    
    plt.axes(position=(0.55, 0.05, 0.35, 0.9))
    m2 = (m[2:] + m[:-2] - 2*m[1:-1]) / 2
    plt.plot(x[1:-1], m2, 'k.-')
    plt.plot(x[nopt], m2[nopt-1], "rp")
    plt.xticks(np.arange(x[0], x[-1], 2))
    plt.ylabel("2nd derivative")
    plt.xlabel("Number of clusters (K)")
    
    plt.xlim(xl)
    plt.ylim([-0.02, 0.3])
    
    return fig
    
@fm.figure_renderer("consensus_clustering", reset=reset, exts=[".png", ".svg"])
def show_pac_result(figsize=(12, 4), path_pac_data="../dynamics_clustering/data/consensus_clustering_hists_sub.pkl",
                    nboundary_lines=(2, 18), cmap="jet"):
    with open(path_pac_data, "rb") as f:
        pac_hist = pkl.load(f)
        
    palette = plt.get_cmap(cmap)
    
    hists = pac_hist["hists"]
    ksets = pac_hist["k"]
    edges = pac_hist["edges"]
    
    x = (edges[1:] + edges[:-1])/2
    pacs = []
    for i in range(len(hists)):
        c = np.cumsum(hists[i])
        pacs.append(c[nboundary_lines[1]] - c[nboundary_lines[0]])
    
    fig = uf.get_figure(figsize)
    plt.axes(position=(0.05, 0.05, 0.35, 0.9))
    for i in range(len(hists)):
        plt.plot(x, np.cumsum(hists[i]), c=palette(i/(len(hists)-1)), alpha=0.5)
    plt.text(0.12, 0.52, "K=%d"%(ksets[0]), color="b")
    plt.text(0.12, 0.88,  "K=%d"%(ksets[-1]), color='r')

    for i in range(2):
        plt.plot([x[nboundary_lines[i]]]*2, [0, 1.2], 'k--')
    plt.plot(x, np.cumsum(hists[nopt]), lw=2, c='k')
    plt.xlabel("Consensus index value")
    plt.ylabel("CDF")
    plt.ylim([0.4, 1.05])
    plt.xlim([0, 1])

    plt.axes(position=(0.55, 0.05, 0.35, 0.9))
    plt.plot(ksets, pacs, 'ko-', markersize=4)
    plt.plot(ksets[nopt], pacs[nopt], 'rp')
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("PAC")

    plt.ylim([-0.02, 0.32])
    # plt.xticks(np.arange(x[0], x[-1], 2))
    plt.xticks(np.arange(3, 15, 2))
    plt.xlim((2, 15))

    return fig

@fm.figure_renderer("silhouette_scores", reset=reset, exts=[".png", ".svg"])
def show_silhouette_scores(figsize=(6, 4), path_pred_data="../dynamics_clustering/data/kmeans_sub.pkl"):
    with open(path_pred_data, "rb") as f:
        pred_data = pkl.load(f)
    
    
    x = pred_data["k"]
    svals = pred_data["silhouette"].reshape((-1, pred_data["nitr"]))
    m = svals.mean(axis=1)
    s = svals.std(axis=1)
    
    fig = uf.get_figure(figsize)
    plt.plot(x, m, 'k.-', markersize=4)
    plt.fill_between(x, m+s, m-s, color='k', edgecolor="none", alpha=0.2)
    plt.plot(x[nopt], m[nopt], "rp")
    plt.ylabel("Silhouette score")
    plt.xlabel("Number of clusters (K)")
    plt.xlim(xl)
    plt.xticks(np.arange(x[0], x[-1], 2))
    plt.ylim((0.225, 0.3))
    
    return fig


@njit 
def build_consensus_matrix(pred_set, sval_set=None):
    
    npoint = pred_set.shape[1]
    N = pred_set.shape[0]
    sval_set = np.ones(N) if sval_set is None else sval_set
    cmat = np.zeros((npoint, npoint))
    
    for n in range(N):
        s = sval_set[n]
        pred = pred_set[n]
        for i in range(npoint):
            cmat[i, i] += s
            for j in range(i+1, npoint):
                if pred[i] == pred[j]:
                    cmat[i, j] += s
                # cmat[j, i] = cmat[i, j]
    
    cmat = cmat / cmat[0, 0]
    for i in range(npoint):
        for j in range(i):
            cmat[i, j] = cmat[j, i]

    return cmat


def compute_dend(path_pred_data):
       
    
    with open(path_pred_data, "rb") as f:
        pred_data = pkl.load(f)
        
    np.random.seed(42)
    nitr = pred_data["nitr"]
    cmat_opt = build_consensus_matrix(pred_data["labels"][nopt*nitr:(nopt+1)*nitr])
    dmat = 1 - cmat_opt
    dmat[dmat <= 0] = 0
    
    model_tree = hc.SLHC(metric="precomputed", method="average") # complete
    model_tree.fit(dmat)
    
    return model_tree, cmat_opt


@fm.figure_renderer("dendrogram", reset=reset, exts=[".png", ".svg"])
def show_dendrogram(figsize=(10, 10), path_pred_data="../dynamics_clustering/data/kmeans_sub.pkl"):
    
    model_tree, cmat_opt = compute_dend(path_pred_data)
    
    sorted_mat, sort_id = model_tree.sort_dmat(cmat_opt.copy())
    cid_tmp = model_tree.cut_dend(dth=0.7)
    
    fig = uf.get_figure(figsize)
    hc.draw_with_dendrogram(model_tree.linkmat, sorted_mat, cid_tmp[sort_id], label="permuted ID (sub)", color_threshold=0.7,
                            fig=fig)
    
    
    return fig

@fm.figure_renderer("cluster_position", reset=reset, exts=[".png", ".svg"])
def show_cluster_position(figsize=(18, 10), path_pred_data="../dynamics_clustering/data/kmeans_sub.pkl"):
    
    feature_dims = (15, 15, 3, 16)
        
    model_tree, cmat_opt = compute_dend(path_pred_data)
    cid_tmp = model_tree.cut_dend(dth=0.7)
    
    sq_cid_tmp = np.reshape(cid_tmp, feature_dims)
    sq_cid = sq_cid_tmp.copy()

    for nr in range(feature_dims[-2]):
        for nw in range(feature_dims[-1]):
            tmp = hc.remove_cluster_island(sq_cid_tmp[:, :, nr, nw], nth=6)
            # tmp2 = hc.remove_cluster_island(tmp, nth=3)
            sq_cid[:, :, nr, nw] = hc.remove_cluster_island(tmp, nth=3)
            
    sq_cid, id_old2new = hc.reorder_sq_cluster_id(sq_cid, start_id=1)

    wsets = [-1, -0.9, -0.7, -0.5, -0.3, -0.1, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.85, 0.9, 0.95, 1]
    
    fig = uf.get_figure(figsize)
    use_row = 2
    
    num_w = len(wsets) // use_row
    num_r = sq_cid.shape[-2] * use_row
    
    len_p = 0.95 / num_w
    len_r = 0.9 / num_r
    xy = np.arange(15)
    
    lb_rank = ("Echelon 0", "Echelon 0.5", "Echelon 1")
    lb_w = [r"$\omega$=%.2f"%(x) for x in wsets]
    
    lines_cluster = []
    for idp in range(len(wsets)):
        for idr in range(sq_cid.shape[-2]):
            lines_cluster.append(hc.get_im_boundary(sq_cid[:, :, idr, idp]))
    
    # ax_sets =[]
    for row in range(use_row):
        for idp in range(num_w):
            for idr in range(sq_cid.shape[-2]):
            # ax_sets.append([])
                idx_w = idp + row * num_w
                idx_r = idr
                
                y_pos = 0.025 + (2-idx_r) * len_r + (3*len_r+0.05)*(1-row) 
                x_pos = 0.025 + idp * len_p
                
                ax = plt.axes(position=[x_pos, y_pos, len_p, len_r])
                im = sq_cid[:, :, idx_r, idx_w].astype(float)
                im[0, 0] = np.nan

                hc.show_sq_cluster(im, x=xy, y=xy, cmap="turbo", cth=2, vmin=1, vmax=7, fontsize=7, aspect="auto")
                for l in lines_cluster[idx_w * sq_cid.shape[-2] + idx_r]:
                    plt.plot(l[0], l[1], 'w', lw=1)

                plt.xticks([0, 7, 14], labels=["", "", ""])
                plt.yticks([0, 7, 14], labels=["", "", ""])
                plt.xlim([-0, 14]); plt.ylim([0, 14])

                if idp == 0:
                    plt.ylabel(lb_rank[idr])
                if idr == 0:
                    plt.title(lb_w[idx_w])

                for n, k in enumerate(("left", "right", "bottom", "top")):
                    ax.spines[k].set_color("k")
                    ax.spines[k].set_linewidth(1.5)
    
    return fig


if __name__ == "__main__":
    show_cluster_position()
    show_kmeans_result()
    # show_pac_result()
    show_silhouette_scores()
    show_dendrogram()
    