import numpy as np
import matplotlib.pyplot as plt
import xarray as xa

import figure_manager as fm
import os
import sys
sys.path.append("../include/pytools")
import utils_fig as uf
import oscdetector as od
uf.set_plt()

reset = True

fdir_coupling = "../coupling/postdata/cfc/"
cw_set = ((4, 15), (4, 10), (7, 10), (7, 5))
# xylb_set = (("Fast", "Fast"), ("Fast", "Slow"), ("Slow", "Fast"), ("Slow", "Slow"))
xylb_set = (("F", "F"), ("F", "S"), ("S", "F"), ("S", "S"))
ax_labels = ("ff", "fs", "sf", "ss")

fm.track_global("fdir_coupling", fdir_coupling)
fm.track_global("cw_set", cw_set)


def compute_spike_prob(prob_map):
    nspike = prob_map.coords["ny"].data
    prob_spike = (prob_map * nspike[np.newaxis,:,np.newaxis].astype(float)).sum(axis=1)
    return prob_spike


@fm.figure_renderer("coupling", reset=reset, exts=[".png", ".svg"])
def draw_coupling(figsize=(12, 15), vmax=0.3):
    
    lb_motif = od.get_motif_labels("ver2")
    
    fig = uf.get_figure(figsize)
    for n, (cid, wid) in enumerate(cw_set):
        # comap = xa.load_dataset(os.path.join(fdir_coupling, "cfc_aa_%d%02d.nc"%(cid, wid)))
        # comap0 = xa.load_dataset(os.path.join(fdir_coupling, "cfc_aa_%d00.nc"%(cid)))
        comap = xa.load_dataset(os.path.join(fdir_coupling, "cfc_sp_%d%02d.nc"%(cid, wid)))
        
        x = comap.coords["nx"].data
        y = comap.coords["fx"].data
        
        for m, lb in enumerate(ax_labels):
            
            plt.subplot(len(cw_set), 4, 4*n+m+1)
            
            prob_spike = compute_spike_prob(comap.prob.sel(dict(nax=lb, fy=1)))
            plt.imshow(prob_spike, 
                       cmap="RdBu_r", extent=(x[0], x[-1], y[0], y[-1]),
                       aspect="auto",
                       origin="lower", vmin=0, vmax=vmax)
            cbar = plt.colorbar(shrink=0.6)
            cbar.ax.text(
                0, 1.3, r"$p^{%s}_{spike}$"%(xylb_set[m][0]), 
                transform=cbar.ax.transAxes, 
                va="top", ha="left",
                fontsize=5
            )
            
            plt.yticks(np.arange(20, 81, 10))
            plt.xticks(np.arange(-np.pi, np.pi+1e-3, np.pi/2),
                       labels=[r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
            plt.ylim([20, 80])
            plt.xlim([-np.pi, np.pi])
            
            if m == 0:
                plt.ylabel("#%d, %s\n"%(cid, lb_motif[wid])+r"$f^{%s}$ (Hz)"%(xylb_set[m][1]))
            else:
                plt.ylabel(r"$f^{%s}$ (Hz)"%(xylb_set[m][1]))
            plt.xlabel(r"$\phi^{%s}$ (Hz)"%(xylb_set[m][1]))
            
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    draw_coupling()