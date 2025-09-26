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

fdir_coupling = "../coupling/postdata/cfc/"
cw_set = ((4, 15), (4, 10), (7, 10), (7, 5))
# xylb_set = (("Fast", "Fast"), ("Slow", "Slow"), ("Fast", "Slow"))
xylb_set = (("F", "F"), ("S", "S"), ("F", "S"))
ax_labels = ("ff", "ss", "fs")

fm.track_global("fdir_coupling", fdir_coupling)
fm.track_global("cw_set", cw_set)


@fm.figure_renderer("coupling", reset=True, exts=[".png", ".svg"])
def draw_coupling(figsize=(12, 15), vmaxs=(0.2, 0.2, 0.05), mode="aa"):
    
    lb_motif = od.get_motif_labels("ver2")
    
    fig = uf.get_figure(figsize)
    for n, (cid, wid) in enumerate(cw_set):
        # comap = xa.load_dataset(os.path.join(fdir_coupling, "cfc_aa_%d%02d.nc"%(cid, wid)))
        # comap0 = xa.load_dataset(os.path.join(fdir_coupling, "cfc_aa_%d00.nc"%(cid)))
        comap = xa.load_dataset(os.path.join(fdir_coupling, "cfc_%s_%d%02d.nc"%(mode, cid, wid)))
        fpsd = comap.coords["fy"]
        
        # for m, nax in enumerate((0, 1, 3)):
        for m, lb in enumerate(ax_labels):
            
            plt.subplot(len(cw_set), 3, 3*n+m+1)
            # print(comap.mut.coords, comap0.mut.coords)
            # im_mut = comap.mut.sel(dict(nax=nax)).values.copy()
            # im_base = comap0.mut.sel(dict(nax=nax)).values.copy()
            # im_mut -= im_base
            im_mut = comap.mut.sel(dict(nax=lb))
            
            if mode != "ap" and m < 2:
                for i in range(im_mut.shape[0]):
                    im_mut[i,:i+1] = np.nan
            
            plt.imshow(im_mut, 
                    cmap="turbo", extent=(fpsd[0], fpsd[-1], fpsd[0], fpsd[-1]),
                    origin="lower", vmin=0, vmax=vmaxs[m])
            plt.colorbar(shrink=0.6)
            
            plt.xlim([20, 80])
            plt.ylim([20, 80])
            plt.xticks(np.arange(20, 81, 10))
            plt.yticks(np.arange(20, 81, 10))

            if m == 0: 
                plt.ylabel("#%d, %s\n"%(cid, lb_motif[wid])+r"$f^{%s}$ (Hz)"%(xylb_set[m][0]))
            else:
                plt.ylabel(r"$f^{%s}$ (Hz)"%(xylb_set[m][0]))
            plt.xlabel(r"$f^{%s}$ (Hz)"%(xylb_set[m][1]))
            
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # draw_aa_coupling()
    draw_coupling(mode="aa", _func_label="aa_coupling")
    draw_coupling(mode="pp", vmaxs=(0.6, 0.6, 0.4), _func_label="pp_coupling")
    draw_coupling(mode="ap", vmaxs=(0.01, 0.01, 0.01), _func_label="ap_coupling")
    # draw_coupling(mode="sp", _func_label="sp_coupling")