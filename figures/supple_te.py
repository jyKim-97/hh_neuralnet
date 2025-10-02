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

import utils_fig as uf
uf.set_plt()

reset = False
te_colors = ("#d92a27", "#045894", "#a9a9a9")
te_dir = "../information_routing/data/te_2d_newmotif_newsurr"
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
# ordering
cw_pairs_flat = sorted(cw_pairs_flat, key=lambda x: (x[0], x[1]))

fm.track_global("te_dir", te_dir)
fm.track_global("cw_pairs_flat", cw_pairs_flat)

@fm.figure_renderer("te_results", reset=reset, exts=[".png", ".svg"])
def draw_te(figsize=(12, 18), ncol=3, avg_method="median", p_ranges=(5, 95)):
    num_row = int(np.ceil(len(cw_pairs_flat) / ncol))

    opt = dict(alpha=0.5, avg_method=avg_method, p_range=p_ranges)
    opt_line = dict(linestyle="-", linewidth=0.5)
    opt_noline = dict(linestyle="none")
    ybar = [0.05, 0.085]
    ylim_max = [0.1, 0.25]
    pop_lbs = ("F", "S")
    
    fig = uf.get_figure(figsize)
    axs = []
    lobjs = {"ltrue": [None, None], "lsurr": [None, None], "lsig": [None, None]}
    for n, (cid, wid) in enumerate(cw_pairs_flat, start=1):
        ax = plt.subplot(num_row, ncol, n)
        axs.append(ax)
        
        tcut = ut.get_max_period(cid)
        te_data_2d = uf.load_pickle(os.path.join(te_dir, "te_%d%02d.pkl"%(cid, wid)))
        te_data = ut.reduce_te_2d(te_data_2d, tcut=tcut)
        tlag = te_data["tlag"]
        
        id_sig_sets = ut.identify_sig_te1d(te_data, prt=p_ranges[1])
        tsig_sets = ut.convert_sig_boundary(id_sig_sets, tlag)
        
        ymax = te_data["te"].max()
        if ymax < 0.07:
            yb, yl, dy = ybar[0], ylim_max[0], 0.02
        else:
            yb, yl, dy = ybar[1], ylim_max[1], 0.05

        for nd in range(2):
            x1 = te_data["te"][:,nd,:]
            x2 = te_data["te_surr"][:,nd,:]
            tlag = te_data["tlag"]
            
            c = te_colors[nd]
            visu.draw_with_err(tlag, x1, c=c, **opt, **opt_noline) # TE
            visu.draw_with_err(tlag, x2, c=te_colors[2], **opt, **opt_noline) # TE surrogate
            l_true, = plt.plot(tlag, np.median(x1, axis=0), c=c, linestyle='-', linewidth=0.5, label=r"$TE_{%s \rightarrow %s}$"%(pop_lbs[nd], pop_lbs[1-nd]))
            l_surr, = plt.plot(tlag, np.median(x2, axis=0), c=c, linestyle='--', linewidth=0.5, label=r"$TE_{%s \rightarrow %s}^{surr}$"%(pop_lbs[nd], pop_lbs[1-nd]))
            
            if lobjs["ltrue"][nd] is None:
                lobjs["ltrue"][nd] = l_true
            if lobjs["lsurr"][nd] is None:
                lobjs["lsurr"][nd] = l_surr

            for tsig in tsig_sets[nd]:
                l_sig, = plt.plot(tsig, [yb+dy*nd]*2, "*-", c=c, lw=1, markersize=2)
                if lobjs["lsig"][nd] is None:
                    lobjs["lsig"][nd] = l_sig
                
        plt.xticks(np.arange(0, 41, 10))
        plt.xlim([0, tcut])
        plt.ylim([0, yl])
        plt.xlabel(r"$\tau$ (ms)")
        plt.ylabel("TE (bits)")
        lb = od.get_motif_labels("ver2")[wid]
        plt.title("#%d, %s"%(cid, lb))
    
    # show legend
    lobjs_list = []
    for k in ("ltrue", "lsurr", "lsig"):
        lobjs_list.extend(lobjs[k])
    
    ax = plt.subplot(num_row, ncol, num_row*ncol)
    ax.axis("off")
    ax.legend(lobjs_list, 
              [r"$TE^{F \rightarrow S}$", r"$TE^{S \rightarrow F}$", 
               r"$TE^{F \rightarrow S}_{surr}$", r"$TE^{S \rightarrow F}_{surr}$",
               r"Significant $TE^{F \rightarrow S}$", r"Significant $TE^{S \rightarrow F}$"],
              loc="center", edgecolor="none", fontsize=6, ncol=1)
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    draw_te()





        