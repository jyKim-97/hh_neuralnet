import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../include/pytools")
import utils_fig as uf
import oscdetector as od
uf.set_plt()

import figure_manager as fm


@fm.figure_renderer("num_mfops", reset=True)
def draw_num_mfops(figsize=(6, 3), prefix_motif=None):
    
    num_winfos_tot = np.zeros((7, 16))
    for nc in range(7):
        cid = nc + 1
        winfos = uf.load_pickle(prefix_motif+"_%d.pkl"%(cid))["winfo"]
        num_winfos_tot[nc] = [len(w) for w in winfos]
    num_winfos_tot[:,0] = 0
    num_winfos_tot = num_winfos_tot / num_winfos_tot.sum(axis=1, keepdims=True)
    num_winfos_tot[:,0] = np.nan
    
    fig = uf.get_figure(figsize)
    
    w0, w = 0.1, 0.8
    plt.axes(position=(0.1, 0.2, 0.8, 0.75))
    cobj = plt.imshow(num_winfos_tot, cmap="hot", origin="lower", extent=(-0.5, 15.5, 0.5, 7.5), vmin=0, vmax=0.4, aspect="auto")
    plt.xticks(np.arange(16), labels=[])
    plt.xlim([0.5, 15.5])
    plt.yticks(np.arange(8))
    plt.ylim([0.5, 7.5])
    plt.ylabel("Landmark ID")
    
    
    lb_set = od.get_motif_labels()[1:]
    
    for n, lb in enumerate(lb_set):
        _w  = w / len(lb_set)
        _w0 = w0 + n * _w
        plt.axes(position=(_w0-0.11, 0.02, _w+0.22, 0.17))
        uf.draw_motif_pictogram(lb, rcolor=[0.5, 0.5, 0.5])
    
    ax = plt.axes(position=(0.95, 0.2, 0.02, 0.75))
    plt.colorbar(cobj, cax=ax, shrink=1)
    
    return fig

    
    
# @fm.figure_renderer("colorbar")
# def draw_colorbar():
#     global cobj
    
#     fig, ax = plt.subplots(1, 1, figsize=(0.15 * uf.cm, 2 * uf.cm), dpi=200)
#     plt.colorbar(cobj, cax=ax, shrink=0.4)
#     return fig


# def main():
    # draw_num_mfops(prefix_motif="../extract_osc_motif/data/osc_motif/motif_info")
    # draw_colorbar()


if __name__ == "__main__":
    draw_num_mfops(prefix_motif="../extract_osc_motif/data/osc_motif/motif_info", _dpi=600)