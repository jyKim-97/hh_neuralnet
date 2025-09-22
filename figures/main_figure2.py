import numpy as np
import matplotlib.pyplot as plt
import xarray as xa

import os
import sys
sys.path.append("../include/pytools")
import utils_fig as uf
uf.set_plt()

import figure_manager as fm

file_umap="./postdata/umap_coord.nc"
file_postdata="../three_pop_mpi/simulation_data/postdata.nc"

fm.track_global("file_umap", file_umap)
fm.track_global("file_postdata", file_postdata)


def sel_params(postdata, key=None, pop=None, type=None):
    return postdata.sel(dict(key=key, pop=pop, type=type)).data.flatten()


def draw_values(value, da, title=None,
                s=1, edgecolor="none", 
                vmin=None, vmax=None,
                cmap="viridis", shrink=0.5, cticks=None,
                **plot_opt):
    plt.scatter(da[:,0], da[:,1], s=s, edgecolor=edgecolor, cmap=cmap, c=value, vmin=vmin, vmax=vmax, rasterized=True, **plot_opt)
    plt.colorbar(shrink=shrink, ticks=cticks)
    plt.axis("off")
    plt.title(title, fontsize=12)
    
    
@fm.figure_renderer("draw_structure_embed", reset=True, exts=[".png", ".svg"])
def draw_echelon(figsize=(10, 5), s=1, vmin=0, vmax=1, cmap="viridis", shrink=0.5, cticks=None):
    da = xa.load_dataarray(file_umap)
    postdata = xa.load_dataarray(file_postdata)
    
    out_tmp = postdata.isel(dict(key=0, type=0, pop=0))
    
    _, _, echelon_grid, _ = xa.broadcast(out_tmp["alpha"], out_tmp["beta"], out_tmp["rank"], out_tmp["w"])
    keys = (r"$\alpha$", r"$\beta$", r"$\omega$")
    
    fig = uf.get_figure(figsize)
    for n, x in enumerate((alpha_grid, beta_grid, w_grid)):
        xf = x.data.flatten()
        
        plt.subplot(1,3,n+1)
        draw_values(xf, da, title=keys[n], s=s, vmin=np.min(xf), vmax=np.max(xf))
        
        # plt.scatter(da[:,0], da[:,1], s=s, edgecolor="none", cmap="viridis", c=xf, vmin=np.min(xf), vmax=np.max(xf), rasterized=True)
        # plt.colorbar(shrink=0.5)
        # plt.title(keys[n])
        # plt.axis("off")
    
    return fig