import numpy as np
import matplotlib.pyplot as plt
import xarray as xa

import os
import sys
sys.path.append("../include/pytools")

import hhsignal
import hhtools
import utils_fig as uf
uf.set_plt()

import figure_manager as fm

file_umap="./postdata/umap_coord.nc"
file_postdata="../three_pop_mpi/simulation_data/postdata.nc"

fm.track_global("file_umap", file_umap)
fm.track_global("file_postdata", file_postdata)


reset = True


def get_psd_set(detail):
    psd_set = []
    for i in range(2):
        psd, fpsd, tpsd = hhsignal.get_stfft(detail["vlfp"][i+1], detail["ts"], 2000,
                                             frange=(5, 110))
        psd_set.append(psd)
    
    return psd_set, fpsd, tpsd


def show_psd(psd, fpsd, tpsd, vmin=0, vmax=1):
    im_obj = plt.imshow(psd, aspect="auto", cmap="jet", origin="lower",
               vmin=vmin, vmax=vmax,
               extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]),
               interpolation="bicubic")
    plt.ylabel("Frequency (Hz)")
    return im_obj


def set_ticks(tl):
    plt.xlim(tl)
    plt.xticks(np.arange(tl[0], tl[1]+1e-3))
    plt.yticks(np.arange(10, 91, 20))
    plt.ylim([10, 90])
    plt.gca().set_xticklabels([])


def set_colorbar(cticks=None):
    cbar = plt.colorbar()
    cbar.set_ticks(cticks)
    
    
@fm.figure_renderer("spec_sample", reset=reset)
def draw_spec(figsize=(2.2, 2.8), cid=0, nt=0, tl=None, vmin=0, vmax=1, cticks=None):
    
    summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data/")
    detail = summary_obj.load_detail(cid-1, nt)
    psd_set, fpsd, tpsd = get_psd_set(detail)
    
    fig = uf.get_figure(figsize)
    
    ax1 = plt.axes((0.01, 0.1, 0.45, 0.8))
    show_psd(psd_set[0], fpsd, tpsd, vmin=vmin, vmax=vmax)
    x0 = tl[0]+1
    uf.show_scalebar(ax1, size=1, label="1 s", anchor_pos=(x0, 25), lw=1, pad=5, color='w', fontsize=5)
    plt.text(x0, 80, "Fast", color="w", fontsize=5)
    set_ticks(tl)

    ax2 = plt.axes((0.5, 0.1, 0.45, 0.8))
    obj = show_psd(psd_set[1], fpsd, tpsd, vmin=vmin, vmax=vmax)
    uf.show_scalebar(ax2, size=1, label="1 s", anchor_pos=(x0, 25), lw=1, pad=5, color='w', fontsize=5)
    plt.text(x0, 80, "Slow", color="w", fontsize=5)
    set_ticks(tl)
    plt.yticks([])
    plt.ylabel("")
    
    ax3 = plt.axes((0.96, 0.1, 0.03, 0.8))
    cbar = plt.colorbar(obj, cax=ax3)
    cbar.set_ticks(cticks)  
    
    
    # set_colorbar(cticks=cticks)
    
    return fig


if __name__ == "__main__":
    draw_spec(cid=7, nt=0, vmin=0, vmax=0.6, tl=(1, 9), cticks=(0, 0.3, 0.6), _func_label="spec_sample_7", _transparent=True)
    draw_spec(cid=5, nt=0, vmin=0, vmax=0.6, tl=(1, 9), cticks=(0, 0.3, 0.6), _func_label="spec_sample_5", _transparent=True)
    draw_spec(cid=4, nt=10, vmin=0, vmax=0.6, tl=(1, 9), cticks=(0, 0.3, 0.6), _func_label="spec_sample_4", _transparent=True)