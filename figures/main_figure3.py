import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append("../include/pytools")
import visu
import hhsignal
import xarray as xa

import oscdetector as od
import utils_te as ut
import figure_manager as fm
import tetools as tt

import hhtools
import utils_fig as uf
uf.set_plt()


data_dir = "../gen_three_pop_samples_repr/data/"
fdir_coburst = "../gen_three_pop_samples_repr/postdata/co_burst"


def get_psd_set(detail):
    psd_set = []
    for i in range(2):
        psd, fpsd, tpsd = hhsignal.get_stfft(detail["vlfp"][i+1], detail["ts"], 2000,
                                             frange=(5, 110))
        psd_set.append(psd)
    return psd_set, fpsd, tpsd


def show_psd(ax, psd, fpsd, tpsd, vmin=0., vmax=1., cmap="jet", interpolation="bicubic"):
    im_obj = ax.imshow(psd, aspect="auto", cmap=cmap, origin="lower",
               vmin=vmin, vmax=vmax,
               extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]),
               interpolation=interpolation)
    
    return im_obj
    
    
def set_ticks(ax, tl, yl):
    plt.sca(ax)
    plt.xlim(tl)
    xt = np.arange(tl[0], tl[1]+1e-3, 0.5)
    xt_labels = np.arange(1, 1+tl[1]-tl[0]+1e-3, 0.5)
    plt.xticks(xt, xt_labels)
    plt.yticks(np.arange(20, 101, 20))
    plt.ylim(yl)
    # plt.xlabel("Time (s)")
    # ax.set_xticklabels([])
    # plt.gca().set_xticklabels([])
    
    
# def set_colorbar(cticks=None):
#     cbar = plt.colorbar()
#     cbar.set_ticks(cticks)


@fm.figure_renderer("mfop_example", reset=False)
def draw_example(figsize=(9, 6), data_dir="", cid=7, nt=93, tl=(2.2, 4.2), th_q=95):
    summary_obj = hhtools.SummaryLoader(data_dir)
    detail = summary_obj.load_detail(cid-1, nt)
    psd_set, fpsd, tpsd = get_psd_set(detail)
    
    psd_set = np.array(psd_set)
    th = np.percentile(psd_set, q=th_q, axis=2, keepdims=True)
    psd_th = psd_set > th
    
    yl = (18, 82)
    lb_set = ("Fast", "Slow")
    color_set = ("k", "#d25606")
    
    fig = uf.get_figure(figsize)
    axs = uf.get_custom_subplots([1, 1], [1, 0.05, 1.], 
                                 w_blank_interval_set=[0.02, 0.2],
                                 h_blank_interval=0.2)
    
    for i in range(2):
        ax_psd = axs[i][0]
        im_obj = show_psd(ax_psd, psd_set[i]-psd_set[i].mean(axis=1, keepdims=True), 
                          fpsd, tpsd, vmin=-0.3, vmax=0.3)
        ax_psd.set_ylabel("Frequency (Hz)")
        plt.colorbar(im_obj, cax=axs[i][1], ticks=(-0.3, 0, 0.3))
        set_ticks(ax_psd, tl, yl)
        
        ax_gray = axs[i][2]
        show_psd(ax_gray, psd_th[i].astype(float), 
                 fpsd, tpsd, vmin=0, vmax=1, cmap="gray", interpolation="none")
        set_ticks(ax_gray, tl, yl)
        
        if i == 1:
            ax_psd.set_xlabel("Time (s)")
            ax_gray.set_xlabel("Time (s)")
        
        # Put population labels
        x0 = tl[0] + (tl[1]-tl[0])/12
        y0 = yl[1] - (yl[1]-yl[0])/12
        for j, ax in enumerate((ax_psd, ax_gray)):
            ax.text(x0, y0, lb_set[i], fontsize=6, color=color_set[j],
                    va="top", ha="left")
    
    axs[0][0].set_title("Power spectrogram")
    axs[0][2].set_title("Thresholded\npower spectrogram")
    

    return fig


@fm.figure_renderer("comap_sample", reset=False)
def draw_comap_sample(figsize=(4.2, 4), cid=7, fdir_coburst="", vmax=0.02, cmap="turbo"):
    
    comap = xa.load_dataarray(os.path.join(fdir_coburst, "co_map_%d.nc"%(cid)))
    fpsd = comap.coords["f1"]
    im_comap = comap.sel(dict(mv="mean", type="fs")).values.copy()
    
    fig = uf.get_figure(figsize)
    ax = plt.axes(position=(0.05, 0.05, 0.9, 0.9))
    plt.imshow(im_comap, cmap=cmap, 
                    extent=(fpsd[0], fpsd[-1], fpsd[0], fpsd[-1]),
                    origin="lower", vmin=0, vmax=vmax)
    # plt.colorbar(ticks=np.linspace(0, vmax, 3))
    
    plt.xlim([20, 80])
    plt.ylim([20, 80])
    plt.xticks(np.arange(20, 81, 10))
    plt.yticks(np.arange(20, 81, 10))
    plt.ylabel(r"Frequency$^F$ (Hz)")
    plt.xlabel(r"Frequency$^S$ (Hz)")
    plt.title("Burst cooccurance\nprobability")
    
    return fig

@fm.figure_renderer("comap_colorbar", reset=False)
def draw_comap_colorbar(figsize=(4.2, 4), vmax=0.02, cmap="turbo"):
    
    fig1 = plt.figure()
    im = plt.imshow(np.linspace(0, vmax, 9).reshape(3, 3), cmap=cmap)
    
    fig = uf.get_figure(figsize)
    ax = plt.axes(position=(0.05, 0.05, 0.9, 0.9))
    plt.colorbar(im, cax=ax, ticks=np.linspace(0, vmax, 3))
    fig1.clf()
    
    
    return fig


if __name__ == "__main__":
    cid = 7
    im_opt = dict(vmax=0.02, cmap="turbo")
    h = 3.
    
    draw_example(figsize=(5, h), data_dir=data_dir, cid=cid, nt=93)
    draw_comap_sample(figsize=(h, h), cid=cid, fdir_coburst=fdir_coburst, **im_opt) #  type: ignore
    draw_comap_colorbar(figsize=(0.2, h), **im_opt) # type: ignore