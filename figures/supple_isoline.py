import numpy as np
import matplotlib.pyplot as plt
import xarray as xa
from scipy.interpolate import RegularGridInterpolator

import figure_manager as fm
import os
import sys
sys.path.append("../include/pytools")
import utils_fig as uf
uf.set_plt()


@fm.figure_renderer("isoline", reset=True, exts=[".png", ".svg"])
def draw_isoline(figsize=(12, 3.5), 
                 file_prefix="../two_pop_mpi/postdata/pe_nu_", 
                 pl_set=((0.051, 0.234), (0.028, 0.105)), 
                 d_set=(14142.14, 15450.48),
                 npoints=31,
                 keys=("chi", "fr", "cv", "fnet"),
                 keylabels=(r"$\chi$", "firing rate (Hz)", "CV", "network frequency (Hz)"),
                 yl_set=((0.1, 0.45), (5, 13), (0., 1), (20, 80))):
    
    fig = uf.get_figure(figsize)
    for n, (key, keylabel) in enumerate(zip(keys, keylabels)):
        plt.subplot(1, len(keys), n+1)
        
        for i, key_pop in enumerate(("fast", "slow")):
            file_path = file_prefix + key_pop + ".nc"
            da = xa.load_dataarray(file_path)
            
            x = da.pe.data
            y = da.nu.data
            z = da.sel(dict(vars=key)).data
            interp = RegularGridInterpolator((x, y), z, method="linear")
            
            pl = pl_set[i]
            d = d_set[i]
            
            xsub = np.linspace(pl[0], pl[1], npoints)
            ysub = d * np.sqrt(xsub)
            zsub = interp(np.vstack((xsub, ysub)).T)
            
            _x = np.linspace(0, 1, npoints)
            plt.plot(_x, zsub, label=key_pop)
        
        plt.xlabel("Echelon")
        plt.xticks(np.arange(0, 1.1, 0.2))
        plt.ylabel(keylabel)
        plt.ylim(yl_set[n])
        plt.xlim([0, 1])
        
        if n == 0:
            plt.legend(loc="upper left", fontsize=5, edgecolor="none")
    
    plt.tight_layout()
    return fig
    


if __name__ == "__main__":
    draw_isoline(npoints=21)
    # draw_isoline(figsize=(5,4), file_path="../two_pop_mpi/postdata/pe_nu_fast.nc", 
    #              pl=(0.051, 0.234), d=14142.14, _func_label="isoline_fast")
    # draw_isoline(figsize=(5,4), file_path="../two_pop_mpi/postdata/pe_nu_slow.nc", 
    #              pl=(0.028, 0.105), d=15450.48, _func_label="isoline_slow")
    