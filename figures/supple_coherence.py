import numpy as np
import matplotlib.pyplot as plt
import xarray as xa

import pickle as pkl
import figure_manager as fm
import os
import sys
sys.path.append("../include/pytools")
import utils_fig as uf
import oscdetector as od
uf.set_plt()

"""
Run "compute_bicoh.py" first to generate bicoh data.
"""

fdir_coh = "../extract_osc_motif/data/bicoh"
cw_pairs = [
            [(2, 2), (1, 2), (3, 2)],
            [(2, 10), (1, 5), (3, 15)],
            [(6, 10), [], (4, 2)],
            [(5, 4), [], (4, 5)],
            [(5, 10), [], (4, 10)],
            [(5, 14), [], (4, 15)],
            [(7, 2), (7, 5), (7, 15)]
        ]

fm.track_global("fdir_coh", fdir_coh)
fm.track_global("cw_pairs", cw_pairs)


def load_pickle(fname):
    with open(fname, "rb") as f:
        return pkl.load(f)
    
    
@fm.figure_renderer("coherence", reset=True, exts=[".png", ".svg"])
def draw_coherence(figsize=(4, 3), cid_target=0, yl=(-0.1, 0.15)):
    
    fig = uf.get_figure(figsize)
    cmap = plt.get_cmap("tab10")
    num = 0
    for _sub in cw_pairs:
        if len(_sub) == 0:
            continue
        for cw in _sub:
            if len(cw) == 0:
                continue
            
            cid, wid = cw
            if cid != cid_target:
                continue
            
            bicoh_data = load_pickle(os.path.join(fdir_coh, "bicoh_%d.pkl"%(cid)))
            
            nboot = bicoh_data["info"]["nboot"]
            coh_sub = bicoh_data["coh"][wid]
            freq = coh_sub["freq"]
            dcoh_avg = coh_sub["dcoh_avg"]
            coh_std = coh_sub["dcoh_std"]
            s = coh_std / np.sqrt(nboot) * 2.58
            
            c = cmap(wid//2)
            plt.plot(freq, dcoh_avg, label="%s"%(od.get_motif_labels("ver2")[wid]), color=c)
            plt.fill_between(freq, dcoh_avg-s, dcoh_avg+s, color=c, alpha=0.3, edgecolor=None)    
            num += 1
    
    plt.title("Landmark %d"%(cid_target))
    plt.hlines(0, 0, 100, colors="k", linestyles="dashed", linewidth=0.8, alpha=0.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"$\Delta$ Coherence")
    plt.xticks(np.arange(20, 81, 20))
    plt.yticks(np.arange(yl[0], yl[1]+1e-3, 0.05))
    plt.xlim([10, 90])
    plt.ylim(yl)
    plt.legend(loc="upper right", fontsize=5, edgecolor="none")
    
    return fig
    
    
if __name__ == "__main__":
    for cid in range(1, 8):
        draw_coherence(cid_target=cid, _func_label="coh%d"%(cid))