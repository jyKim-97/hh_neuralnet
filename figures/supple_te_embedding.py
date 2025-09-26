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

cw_pairs = ((4, 15), (7, 15))
max_dim = 5
te_dir = "../information_routing/data/te_2d_newmotif_newsurr"
te_emb_dir = "../information_routing/data/te_2d_embedding/"
c_rect = "#676767"

fm.track_global("cw_pairs", cw_pairs)
fm.track_global("max_dim", max_dim)
fm.track_global("te_dir", te_dir)
fm.track_global("te_emb_dir", te_emb_dir)

@fm.figure_renderer("te_embedding", reset=True, exts=[".png", ".svg"])
def show_te_embedding(figsize=(3.5, 10), cid=1, wid=0, p_ranges=(5, 95)):
    
    tcut = ut.get_max_period(cid)
    y0 = 4
    box_height = 2
    
    fig = uf.get_figure(figsize)
    
    for i in range(max_dim):
        
        # plt.axes(position=(0.1, y0 - (i+1)*(box_height+0.5), 0.8, box_height))
        plt.subplot(max_dim, 1, i+1)
        
        if i == 0:
            print("Collecting from %d, %d"%(cid, wid))
            te = uf.load_pickle(os.path.join(te_dir, "te_%d%02d.pkl"%(cid, wid)))
        else:
            te = uf.load_pickle(os.path.join(te_emb_dir, "te%d%02d_%d.pkl"%(cid, wid, i+1)))
        te1d = ut.reduce_te_2d(te, tcut=tcut)
        tlag = te1d["tlag"]
        
        id_sig = ut.identify_sig_te1d(te1d, prt=p_ranges[1])
        tsig = ut.convert_sig_boundary(id_sig, tlag)
        
        visu.draw_te_diagram_reduce(tsig, xmax=tcut, y0=2*box_height, colors=[c_rect]*2,
                                    box_height=box_height, visu_type="arrow", fontsize=6)
        plt.ylim([-2, y0+2])
        
        if i == 0:
            title = "Original IRP in %s"%(od.get_motif_labels("ver2")[wid])
        else:
            title = r"$IRP^{emb}_{%d}$"%(i+1)
        plt.title(title)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    for cid, wid in cw_pairs:
        show_te_embedding(cid=cid, wid=wid, p_ranges=(2.5, 97.5), _func_label="te_embedding_%d%02d"%(cid, wid))
