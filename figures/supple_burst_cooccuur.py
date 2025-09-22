import numpy as np
import matplotlib.pyplot as plt
import xarray as xa

import figure_manager as fm
import os
import sys
sys.path.append("../include/pytools")
import utils_fig as uf
uf.set_plt()

fdir_coburst = "../gen_three_pop_samples_repr/postdata/co_burst"

fm.track_global("fdir_coburst", fdir_coburst)


@fm.figure_renderer("burst_cooccuur_map", reset=True, exts=[".png", ".svg"])
def burst_cooccuur_map(figsize=(12, 20), vmax_set=(0.04, 0.04, 0.02)):
    
    tp_set = ("ff", "ss", "fs")
    xylb_set = (("Fast", "Fast"), ("Slow", "Slow"), ("Fast", "Slow"))
    
    fig = uf.get_figure(figsize)
    for cid in range(1, 8):
        comap = xa.load_dataarray(os.path.join(fdir_coburst, "co_map_%d.nc"%(cid)))
        fpsd = comap.coords["f1"]
        
        for ntp in range(3):
            tp = tp_set[ntp]
            
            im_true = comap.sel(dict(mv="mean", type=tp)).values.copy()
            im_perm = comap.sel(dict(mv="thr",  type=tp)).values.copy()
            sig_map = (im_true > im_perm).astype(float)
            sig_map[sig_map == 0] = 0.2
            if ntp < 2:
                for i in range(len(fpsd)):
                    sig_map[i,:i] = 0

            plt.subplot(7, 3, (cid-1)*3 + ntp + 1)
            # vmax = 0.05 if ntp < 2 else 0.03
            vmax = vmax_set[ntp]
            plt.imshow(im_true, alpha=sig_map, cmap="turbo", 
                    extent=(fpsd[0], fpsd[-1], fpsd[0], fpsd[-1]),
                    origin="lower", vmin=0, vmax=vmax)
            plt.colorbar(shrink=0.6, ticks=np.arange(0, vmax+0.01, 0.01))
            
            plt.xlim([20, 80])
            plt.ylim([20, 80])
            plt.xticks(np.arange(20, 81, 10))
            plt.yticks(np.arange(20, 81, 10))
            
            xlb, ylb = xylb_set[ntp]
            if ntp == 0: 
                plt.ylabel("Landmark %d\n"%(cid)+r"$f_{%s}$ (Hz)"%(xlb))
            else:
                plt.ylabel(r"$f_{%s}$ (Hz)"%(xlb))
            plt.xlabel(r"$f_{%s}$ (Hz)"%(ylb))

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    burst_cooccuur_map()