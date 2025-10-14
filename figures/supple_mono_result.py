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
import pickle as pkl
import hhtools
import xarray as xa

import utils_fig as uf
uf.set_plt()


reset=True

c_rect = "#676767"
tags = ("_mfast", "_mslow")
fdir_data_prefix = "../gen_three_pop_samples_repr/data" 
fname_umap_prefix = "./postdata/umap_coord" # + .nc
fname_cluster_prefix = "../dynamics_clustering/data/cluster_id_sub" # +.nc
fname_amp_prefix = "../extract_osc_motif/data/osc_motif"
fdir_te_prefix = "../information_routing/data/te_2d" 

cw_pairs = [
    ((1, 10), (2, 10), (3, 10), (4, 2), (4, 8), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10)),
    ((1, 10), (2, 2), (2, 8), (2, 10), (3, 10), (4, 2), (4, 8), (4, 10), (5, 10), (6, 10))
]

@fm.figure_renderer("dynamics_samples", reset=reset)
def draw_dynamics_samples(figsize=(12, 3), cnt_pairs=((5, 0), (4, 0))):
    
    fig = uf.get_figure(figsize)
    # generate PSD
    for i in range(2):
        ax_top = plt.axes((0.1+0.5*i, 0.55, 0.15, 0.4))
        ax_bot = plt.axes((0.1+0.5*i, 0.05, 0.15, 0.4))
        sobj = hhtools.SummaryLoader(fdir_data_prefix+tags[i])
        detail = sobj.load_detail(*cnt_pairs[i])
        draw_psd(detail, [ax_top, ax_bot])
    
    # draw UMAP
    for i in range(2):
        ax = plt.axes((0.3+0.5*i, 0.05, 0.2, 0.9))
        draw_cid_on_umap(tags[i], ax)
    
    return fig

@fm.figure_renderer("irp_samples", reset=reset)
def draw_irp_samples(figsize=(12, 10), prt=95):
    nrow_max = 5
    w = 0.2
    h = (0.9 - (nrow_max-1)*0.05)/nrow_max
    
    fig = uf.get_figure(figsize)
    for ntp in range(2): # fast, slow
        w0 = 0.5*ntp
        tag = tags[ntp]
        for n, (cid, wid) in enumerate(cw_pairs[ntp]):
            file_path = os.path.join(fdir_te_prefix+tag, "te_%d%02d.pkl"%(cid, wid))
            with open(file_path, "rb") as fp:
                te2d = pkl.load(fp)
            tcut = ut.get_max_period(cid, fname=os.path.join(fname_amp_prefix+tag, "amp_range_set.pkl"))
            
            i, j = nrow_max-n%nrow_max, n//nrow_max
            ax = plt.axes((w0+(w+0.05)*j, (h+0.05)*i, w, h))
            draw_irp(ax, te2d, tcut, prt=prt)
            
            if wid == 2:
                s1, s2 = "o", "x"
            elif wid == 8:
                s1, s2 = "x", "o"
            elif wid == 10:
                s1, s2 = "o", "o"
            else:
                raise NotImplemented("Not implemented wid: %d"%(wid))
            
            plt.title(r"#%d, $P_1[%s]P_2[%s]$"%(cid, s1, s2), pad=-5)
            
    return fig        
    
    
def draw_irp(ax, te2d, tcut, prt=95):
    te1d = ut.reduce_te_2d(te2d, tcut)
    tlag = te1d["tlag"]
    id_sig_sets = ut.identify_sig_te1d(te1d, prt=prt)
    
    dte = np.median(te1d["te"], axis=0) - np.percentile(te1d["te_surr"], q=prt, axis=0)
    for nd in range(2):
        for n in range(len(id_sig_sets[nd]))[::-1]:
            n0, n1 = id_sig_sets[nd][n][0], id_sig_sets[nd][n][-1]
            if n0 < 30 < n1:
                continue
            nmid = int(np.median(id_sig_sets[nd][n]))
            if dte[nd][nmid] - dte[nd][n1] < 1e-4:
                id_sig_sets[nd].pop(n)
    tsig_sets = ut.convert_sig_boundary(id_sig_sets, tlag)
    
    box_height = 2
    plt.sca(ax)
    visu.draw_te_diagram_reduce(tsig_sets, 
                                xmax=tcut, y0=2*box_height, colors=[c_rect]*2, 
                                box_height=box_height, visu_type="arrow",
                                fontsize=6, lb_text=(r"$P_1$", r"$P_2$"))



def draw_cid_on_umap(tag, ax, cmap="turbo"):
    file_umap = fname_umap_prefix + tag + ".nc"
    file_cid  = fname_cluster_prefix + tag + ".nc"
    
    cid_nc = xa.load_dataset(file_cid)
    cid_nc = cid_nc.cluster_id.values.flatten()
    
    umap_coord = xa.load_dataarray(file_umap)
    umap_coord = umap_coord.values
    
    ax.scatter(umap_coord[:, 0], umap_coord[:, 1], c=cid_nc, cmap=cmap, s=0.5, edgecolor="none")
    ax.set_xlabel("UMAP axis 1")
    ax.set_ylabel("UMAP axis 2")
    ax.set_xticks([])
    ax.set_yticks([])
    
    
def draw_psd(detail, axs, cmap="jet", vmin=0, vmax=0.8, tl=(2, 5)):
    
    t = detail["ts"]
    v = detail["vlfp"][1:]
    
    psd_set = []
    idt = t >= 1
    for i in range(2):
        psd, fpsd, tpsd = hhsignal.get_stfft(v[i][idt], t[idt], 2000, frange=(3, 100))
        psd_set.append(psd)
        
    for i in range(2):
        im_obj = axs[i].imshow(psd_set[i], 
                      cmap=cmap,
                      extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]), 
                      origin="lower", aspect="auto",
                      vmin=vmin, vmax=vmax)
        # axs[i].colorbar()
        if i == 1:
            axs[i].set_xlabel("Time (s)")
        else:
            axs[i].set_xticklabels([])
        axs[i].set_ylabel("Frequency (Hz)")
        axs[i].set_xticks(np.arange(tl[0], tl[1]+0.1))
        axs[i].set_yticks(np.arange(20, 81, 20))
        axs[i].set_ylim([10, 90])
        axs[i].set_xlim(tl)
        plt.colorbar(im_obj, ticks=np.arange(vmin, vmax+0.01, 0.2))
        
        x0 = (tl[0]+(tl[1]-tl[0])*0.2)
        axs[i].text(x0, 80, "Pop %d"%(i+1), va="center", ha="center", color="w", fontsize=6)
        

if __name__ == "__main__":
    draw_dynamics_samples()
    draw_irp_samples()