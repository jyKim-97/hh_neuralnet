import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append("../include/pytools")
import visu
import hhsignal

import oscdetector as od
import utils_te as ut
import figure_manager as fm
import tetools as tt

import hhtools
import xarray as xa

import utils_fig as uf
uf.set_plt()

te_colors = ("#d92a27", "#045894", "#a9a9a9")
kappa_dir = "../transmission_line/simulation_data/postdata/kappa_stat"
cw_pairs = [
            [(2, 2), (1, 2), (3, 2)],
            [(2, 10), (1, 5), (3, 15)],
            [(6, 10), [], (4, 2)],
            [(5, 4), [], (4, 5)],
            [(5, 10), [], (4, 10)],
            [(5, 14), [], (4, 15)],
            [(7, 2), (7, 5), (7, 15)]
        ]

cw_pairs_flat = []
for sub in cw_pairs:
    for cw in sub:
        if len(cw) > 0:
            cw_pairs_flat.append(cw)
cw_pairs_flat = sorted(cw_pairs_flat, key=lambda x: (x[0], x[1]))
            
fm.track_global("kappa_dir", kappa_dir)
fm.track_global("cw_pairs_flat", cw_pairs_flat)

def load_kappa(kappa_dir, cid, wid):
    fname = os.path.join(kappa_dir, "kappa_%d%02d.nc"%(cid, wid))
    return xa.open_dataset(fname)
            

def get_err_range(data, method="percentile", p_ranges=(5, 95), smul=1.96):
    # Assume that data is 2D (nsamples, T)
    if method == "percentile":
        m = np.median(data, axis=0)
        smin = np.percentile(data, p_ranges[0], axis=0)
        smax = np.percentile(data, p_ranges[1], axis=0)
    elif method == "std":
        m = data.mean(axis=0)
        s = data.std(axis=0) / np.sqrt(data.shape[0]) * smul
        smin = m - s
        smax = m + s
    else:
        raise ValueError("Unknown method: %s"%method)
    
    return m, smin, smax   

        
@fm.figure_renderer("tline_results", reset=True, exts=[".png", ".svg"])
def draw_tline(figsize=(12, 18), ncol=3, err_method="std", err_std=1.96, p_ranges=(5, 95)):
    num_row = int(np.ceil(len(cw_pairs_flat) / ncol))
    lb_set = ("S", "F")
    
    fig = uf.get_figure(figsize)
    axs = []
    cid_col = 1
    ax_col = []
    ylim_max = -1
    for n, (cid, wid) in enumerate(cw_pairs_flat, start=1):
        ax = plt.subplot(num_row, ncol, n)
        axs.append(ax)
        
        kappa_set = load_kappa(kappa_dir, cid, wid)
        
        yl_sub = []
        for nd in range(2):
            t = kappa_set.ndelay.data
            ym_b, ymin_b, ymax_b = get_err_range(kappa_set.kappa_base.isel(dict(ntp=1-nd)).data, method=err_method, smul=err_std, p_ranges=p_ranges)
            ym, ymin, ymax = get_err_range(kappa_set.kappa.isel(dict(ntp=1-nd)).data, method=err_method, smul=err_std, p_ranges=p_ranges)
            
            plt.plot(t, ym, color=te_colors[nd], lw=1, label=r"$\kappa_%s$"%(lb_set[nd]))
            plt.fill_between(t, ymin, ymax, color=te_colors[nd], alpha=0.3, edgecolor="none")
            plt.fill_between(t, ymin_b, ymax_b, color="k", alpha=0.5, edgecolor="none", label=r"$\kappa_{base}$")
            
            yl = max(np.abs([ymin.min(), ymax.max()]))
            ylim_cand = np.ceil(yl*10)/10
            yl_sub.append(ylim_cand)
            
        if cid != cid_col or n == len(cw_pairs_flat):
            # print(cid_col, ax_col, ylim_max)
            if n == len(cw_pairs_flat):
                ax_col.append(ax)
            for _ax in ax_col:
                _ax.set_ylim([-ylim_max, ylim_max])
                _ax.set_yticks(np.arange(-ylim_max, ylim_max+0.01, ylim_max/2), labels=[str("%d %%"%(int(np.round(100*x)))) for x in np.arange(-ylim_max, ylim_max+0.01, ylim_max/2)])

            ax_col = []
            ylim_max = -1
            cid_col = cid
        
        ylim_cand = max(yl_sub)
        if ylim_cand > ylim_max:
            ylim_max = ylim_cand
        ax_col.append(ax)
                
        # plt.ylim([-0.3, 0.3])
        plt.xticks(np.arange(0, 31, 10))
        # plt.yticks(np.arange(-0.3, 0.31, 0.1), labels=[str("%d %%"%(int(np.round(100*x)))) for x in np.arange(-0.3, 0.31, 0.1)])
        
        plt.xlabel("Delay, d (ms)")
        plt.ylabel(r"$\kappa$")
                
        plt.xticks(np.arange(0, 31, 10))
        plt.xlim([0, 31])
        # plt.ylim([-0.25, 0.25])
        lb = od.get_motif_labels("ver2")[wid]
        plt.title("#%d, %s"%(cid, lb))
        
        if n == 1:
            plt.legend(loc="upper right", edgecolor="none", fontsize=5, ncol=2)
        
    plt.tight_layout()
    
    return fig
            
            
if __name__ == "__main__":
    draw_tline()
