import numpy as np
import matplotlib.pyplot as plt
# import pickle as pkl
import numpy as np
from scipy.ndimage import gaussian_filter
# from 

import os
import sys
sys.path.append("../include")
sys.path.append("../include/pytools")
import utils_fig as uf
import hhtools
import hhsignal
import xarray as xa
import figure_manager as fm

uf.set_plt()


fdir_data = "../gen_three_pop_samples_repr/data"
fdir_bprops = "../gen_three_pop_samples_repr/postdata/bprops"

fm.track_global("fdir_data", fdir_data)
fm.track_global("fdir_bprops", fdir_bprops)


def get_psd_set(detail):
    psd_set = []
    for i in range(2):
        psd, fpsd, tpsd = hhsignal.get_stfft(detail["vlfp"][i+1], detail["ts"], 2000,
                                             frange=(5, 110))
        psd_set.append(psd)
    return psd_set, fpsd, tpsd


def show_psd(psd, fpsd, tpsd, vmin=0, vmax=1):
    plt.imshow(psd, aspect="auto", cmap="jet", origin="lower",
               vmin=vmin, vmax=vmax,
               extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]),
               interpolation="bicubic")
    plt.ylabel("Frequency (Hz)")


def set_ticks(tl):
    plt.xlim(tl)
    plt.xticks(np.arange(tl[0], tl[1]+1e-3))
    plt.yticks(np.arange(10, 91, 20))
    plt.ylim([10, 90])
    plt.gca().set_xticklabels([])
    
    
def set_colorbar(cticks=None):
    cbar = plt.colorbar()
    cbar.set_ticks(cticks)


@fm.figure_renderer("landmark_psd_sample", reset=True, exts=[".png", ".svg"])
def draw_PSD_sample(figsize=(3.2, 4), cid=1, nt=0, tl=(1, 9)):
    
    summary_obj = hhtools.SummaryLoader(fdir_data)
    
    detail = summary_obj.load_detail(cid, nt)
    psd_set, fpsd, tpsd = get_psd_set(detail)
    
    cmax_ticks = (0.6, 1)
    fig = uf.get_figure(figsize)
    for n in range(2):
        show_psd(psd_set[n], fpsd, tpsd, vmin=0, vmax=cmax_ticks[n])

    plt.subplot(212)
    show_psd(psd_set[1], fpsd, tpsd, vmin=0, vmax=1)
    set_ticks(tl)
    set_colorbar(cticks=((0, 0.5, 1)))
    
    return fig


def _count(matsize, idx, idy):
    num = np.zeros(matsize)
    for n in range(len(idx)):
        num[idy[n], idx[n]] += 1
    return num        


def get_mid_pts(edges):
    return (edges[1:] + edges[:-1])/2


def hist2d(y, x, yedges, xedges):
    idx = np.digitize(x, xedges, right=False)
    idy = np.digitize(y, yedges, right=False)
    
    id_nan = (idx==0) | (idx==len(xedges))
    id_nan = id_nan | (idy==0)
    id_nan = id_nan | (idy==len(yedges))

    idx = idx[~id_nan]-1
    idy = idy[~id_nan]-1
    
    matsize = (len(yedges)-1, len(xedges)-1)
    num_hist = _count(matsize, idx, idy)
    return num_hist


def draw_burst_props(bprop_f, bprop_l, nbins=21, s=0.8, vmax=0.05,
                    cticks=None):
    
    fedges = np.arange(5, 101, 5)
    ledges = np.linspace(-0.2, 0.6, 21)
    y = get_mid_pts(fedges)
    x = get_mid_pts(ledges)
    
    im = hist2d(bprop_f, bprop_l, fedges, ledges)
    im = gaussian_filter(im, 0.8)
    
    plt.contourf(x, y, im/im.sum(), 
                 np.concatenate((np.linspace(0,vmax,nbins),[1])),
                 cmap="turbo", 
                 vmax=vmax, vmin=0)
    # colorbar
    if cticks is None:
        cticks = [0, vmax/2, vmax]
    cbar = plt.colorbar(ticks=cticks)
    cbar.ax.set_ylim([0, vmax])
    
    plt.ylim([10, 90])
    plt.yticks(np.arange(10, 91, 20))
    
    # set labels
    plt.xlabel("Burst duration (s)", labelpad=2)
    plt.ylabel("Frequency (Hz)")
    

@fm.figure_renderer("landmark_bprops", reset=True, exts=[".png", ".svg"])
def draw_burst_summary(figsize=(3.2, 9.6), cid=1, nt=0, 
                       tl=(1, 9), xl=(0, 0.5),
                       cmax_psd=(0.6, 1), cmax_bprop=(0.4, 0.4),
                       ):
    
    lb_pop = ("Fast", "Slow")
    
    # get PSD
    summary_obj = hhtools.SummaryLoader(fdir_data)
    detail = summary_obj.load_detail(cid-1, nt)
    psd_set, fpsd, tpsd = get_psd_set(detail)
    
    # get bprops
    bprops = uf.load_pickle(os.path.join(fdir_bprops, f"bprops_{cid}.pkl"))
    mbin_t = bprops["attrs"]["mbin_t"]
    text_opt = dict(color="w", fontsize=9, va="center", ha="center")
    
    if "burst_len" not in bprops["burst_props"][0]:
        for n in range(2):
            bprops["burst_props"][n]["burst_len"] = []
            for r in bprops["burst_props"][n]["burst_range"]:
                bprops["burst_props"][n]["burst_len"].append(r[1]-r[0])
    
    fig = uf.get_figure(figsize)
    for n in range(2):
        plt.axes(position=(0.05, 0.76-0.21*n, 0.9, 0.17))
        show_psd(psd_set[n], fpsd, tpsd, vmin=0, vmax=cmax_psd[n])
        set_ticks(tl)
        set_colorbar(cticks=((0, cmax_psd[n]/2, cmax_psd[n])))
        plt.text(tl[0]+0.18*(tl[1]-tl[0]), 78, lb_pop[n], **text_opt)
        
        if n == 0:
            plt.title("Power spectrogram")
        
    for n in range(2):
        plt.axes(position=(0.05, 0.3-0.25*n, 0.9, 0.17))
        draw_burst_props(bprops["burst_props"][n]['burst_f'], 
                     np.array(bprops["burst_props"][n]['burst_len'])*mbin_t,
                     nbins=31, vmax=cmax_bprop[n])
        plt.xlim(xl)
        plt.xticks([0, xl[1]/2, xl[1]])
        
        if n == 0:
            plt.title("Burst feature density")
        plt.text(xl[0]+0.18*(xl[1]-xl[0]), 78, lb_pop[n], **text_opt)
    
    return fig


@fm.figure_renderer("landmark_diagram", reset=True, exts=[".png", ".svg"])
def draw_landmark(figsize=(3.2, 3.2), cid=1):
    fig = uf.get_figure(figsize)
    ax = plt.axes(position=(0.05, 0.05, 0.9, 0.9))
    uf.draw_landmark_diagram(cid=cid, **uf.landmark_points[cid-1], 
                             ax=ax)
    # plt.title("Landmark %d"%(cid))
    return fig

    
    
if __name__ == "__main__":
    draw_burst_summary(cid=1, cmax_psd=(0.3, 0.3), cmax_bprop=(0.04, 0.04), _func_label="landmark_bprops_1")
    draw_burst_summary(cid=2, cmax_psd=(0.3, 0.3), cmax_bprop=(0.04, 0.04), _func_label="landmark_bprops_2")
    draw_burst_summary(cid=3, cmax_psd=(0.4, 0.6), cmax_bprop=(0.04, 0.04), _func_label="landmark_bprops_3")
    draw_burst_summary(cid=6, cmax_psd=(0.4, 0.4), cmax_bprop=(0.04, 0.04), _func_label="landmark_bprops_6")
    draw_landmark(cid=1, _func_label="landmark_diagram_1", _transparent=True)
    draw_landmark(cid=2, _func_label="landmark_diagram_2", _transparent=True)
    draw_landmark(cid=3, _func_label="landmark_diagram_3", _transparent=True)
    draw_landmark(cid=6, _func_label="landmark_diagram_6", _transparent=True)