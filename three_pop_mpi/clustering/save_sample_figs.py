import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import sys

curr_dir = "~/jungyoung/Project/hh_neuralnet/three_pop_mpi"

sys.path.append("../../include/")
import hhtools
import hhsignal
import hhclustering as hc

def load_obj():
    obj = hhtools.SummaryLoader("../asym_link_full/")
    # correction
    obj.summary["chi"][:,:,:,:,24:48,0] = np.nan
    return obj


def draw_quadratic_summary(data, fname=None, xl_raster=(1500, 2500), nsamples=20, wbin_t=1, fs=2000, shuffle=False):
    
    from tqdm.notebook import tqdm
    from scipy.ndimage import gaussian_filter1d
    
    teq = 0.5
    
    plt.figure(dpi=100, figsize=(9, 12))
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
    
    plt.show()


def export_sample_figs(obj, cluster_id, silhouette_vals, col_names, nshow=3):
    k0 = min(cluster_id)
    k1 = max(cluster_id)
    targets = []
    for target_cid in range(k0, k1+1):
        targets.append([])
        for case in ("best", "intermediate", "worst"):
            tags = hc.show_sample_cases(obj, target_cid, cluster_id, silhouette_vals, col_names, case=case, nshow=nshow, save=True)
            targets[-1].append(tags)

    import pickle as pkl
    with open("./sample_figs/tags.pkl", "wb") as fp:
        pkl.dump({"targets": targets}, fp)


def main():
    obj = load_obj()

    # load data
    with open("./data/rcluster.pkl", "rb") as fp:
        tmp = pkl.load(fp)

    rcluster_id = tmp["rcluster_id"]
    rsval = tmp["rsval"]

    # load data
    with open("./data/purified_data.pkl", "rb") as fp:
        buf = pkl.load(fp)
    col_names = buf["col_names"]

    export_sample_figs(obj, rcluster_id, rsval, col_names, nshow=3)


if __name__ == "__main__":
    main()
