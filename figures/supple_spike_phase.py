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

reset = False

fdir_coupling = "../coupling/postdata/cfc/"
cw_set = ((4, 15), (4, 10), (7, 10), (7, 5))
# xylb_set = (("Fast", "Fast"), ("Fast", "Slow"), ("Slow", "Fast"), ("Slow", "Slow"))
# xylb_set = (("F", "F"), ("F", "S"), ("S", "F"), ("S", "S"))
# ax_labels = ("ff", "fs", "sf", "ss")

xylb_set = (("F", "F"), ("S", "F"), ("F", "S"), ("S", "S"))
ax_labels = ("ff", "sf", "fs", "ss")

fm.track_global("fdir_coupling", fdir_coupling)
fm.track_global("cw_set", cw_set)


def compute_spike_prob(prob_map):
    nspike = prob_map.coords["ny"].data
    prob_spike = (prob_map * nspike[np.newaxis,:,np.newaxis].astype(float)).sum(axis=1)
    return prob_spike


@fm.figure_renderer("coupling", reset=reset, exts=[".png", ".svg"])
def draw_coupling(figsize=(12, 18), vmax=0.3):
    
    lb_motif = od.get_motif_labels("ver2")
    franges = ((30, 40), (60, 80))
    color_franges = ("#006492", "#c20000")
    label_franges = [r"[%d, %d] Hz"%(fr[0], fr[1]) for fr in franges]
    
    fig = uf.get_figure(figsize)
    ax_set = uf.get_custom_subplots(
        w_ratio=[1, 0.05]*4,
        h_ratio=[3, 3, 2]*2,
        w_blank_interval_set=([0.01, 0.1]*4)[:-1],
        h_blank_interval=0.05
    )
    
    nrow, ncol = len(ax_labels)+2, len(cw_set)
    for nc, (cid, wid) in enumerate(cw_set):        
        comap = xa.load_dataset(os.path.join(fdir_coupling, "cfc_sp_%d%02d.nc"%(cid, wid)))
          
        x = comap.coords["nx"].data
        y = comap.coords["fx"].data
        prob_spike_set = []
        
        for nr in range(nrow):
            ax = ax_set[nr][2*nc]
            ax_cbar = ax_set[nr][2*nc+1]
            ax_cbar.axis("off")
            
            if nr % 3 < 2:
                m = (nr//3)*2 + nr%3
                lb = ax_labels[m]
                
                prob_spike = compute_spike_prob(comap.prob.sel(dict(nax=lb, fy=1)))
                prob_spike_set.append(prob_spike.data)
                im_obj = ax.imshow(prob_spike, 
                        cmap="RdBu_r", extent=(x[0], x[-1], y[0], y[-1]),
                        aspect="auto",
                        origin="lower", vmin=0, vmax=vmax)
                ax.set_yticks(np.arange(20, 81, 10))
                ax.set_xticks(np.arange(-np.pi, np.pi+1e-3, np.pi/2))
                ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
                ax.set_ylim([20, 80])
                ax.set_xlim([-np.pi, np.pi])
                ax.set_ylabel(r"$f^{%s}$ (Hz)"%(xylb_set[m][1]))
                ax.set_xlabel(r"$\phi^{%s}$"%(xylb_set[m][1]))
                
                ax_cbar2 = ax_cbar.inset_axes([0., 0.2, 1, 0.6])
                # ax_cbar2.axis("off")
                cbar = plt.colorbar(im_obj, cax=ax_cbar2)
                cbar.ax.text(
                    -0.2, 1.3, r"$p^{%s}_{spike}$"%(xylb_set[m][0]), 
                    transform=cbar.ax.transAxes, 
                    va="top", ha="left",
                    fontsize=6
                )
                
            else:
                draw_square_axis(ax)
                for nd in range(2):
                    prob = prob_spike_set[nd]
                    for i, (c, fr) in enumerate(zip(color_franges, franges)):
                        f0 = np.mean(fr)
                        idf = (y >= fr[0]) & (y < fr[1])
                        p0 = prob[idf, :].mean(axis=0)
                        ax.plot(x/f0*1e3, p0*(-1)**nd, c=c, label=label_franges[i]) # sec -> msec
                ax.set_xticks(np.arange(-50, 51, 25))
                ax.set_xlim([-50, 50])
                ax.set_ylim([-0.4, 0.4])
                ax.set_yticks([-0.3, 0.3])
                ax.set_yticklabels([0.3, 0.3])
                ax.set_xlabel("Time from the peak in %s (ms)"%(xylb_set[m][1]), labelpad=17, fontsize=6)
                ax.text(-60, 0.15, r"$p^{F}_{spike}$", rotation=90, fontsize=5, ha="right", va="center")
                ax.text(-60, -0.15, r"$p^{S}_{spike}$", rotation=90, fontsize=5, ha="right", va="center")
                
                # if nr == 2 and nc == 0:
                ax.text(30, 0.4, label_franges[0], color=color_franges[0], fontsize=4, ha="center")
                ax.text(30, 0.28, label_franges[1], color=color_franges[1], fontsize=4, ha="center")
                #     ax.text()
                
                # if nc == 0:
                #     ax.legend(fontsize=5, edgecolor="none", loc="upper right")
                
                prob_spike_set = []
                
        ax_set[0][2*nc].set_title("#%d, %s"%(cid, lb_motif[wid]))

    return fig


def draw_square_axis(ax):
    uf.show_spline(ax, top=False, right=False, bottom=True, left=False)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    # ax.spines['bottom'].set_position('zero')
    ax.tick_params(axis="x", direction="inout", pad=-0.1, zorder=-1)
    ax.tick_params(axis="y", direction="inout", pad=-0.1, zorder=-1)
    # ax.vlines(0, -0.4, 4, color='k', linestyle='-', linewidth=0.3)
    opt = dict(head_width=10, head_length=0.05, color='k', linewidth=0.4, length_includes_head=True, overhang=0.5)
    ax.arrow(0, 0, 0, 0.4, **opt)
    ax.arrow(0, 0, 0, -0.4, **opt)
    
    # for label in ax.get_xticklabels():
    #     label.set_y(0.1)   # y=0 근처로 이동 (값을 조절해서 맞추면 됨)
    
    
    
    # ax.arrow(-5, 0, 10, 0, 
    #      length_includes_head=True, 
    #      head_width=0.2, 
    #      head_length=0.3, 
    #      fc='black', ec='black')

    # # y축 화살표
    # ax.arrow(0, -5, 0, 10, 
    #         length_includes_head=True, 
    #         head_width=0.2, 
    #         head_length=0.3, 
    #         fc='black', ec='black')

    # 라벨
    # ax.text(5.2, 0, 'x', fontsize=12)
    # ax.text(0, 5.2, 'y', fontsize=12)


if __name__ == "__main__":
    draw_coupling()