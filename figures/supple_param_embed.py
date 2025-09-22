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
def draw_structure_embed(figsize=(10, 3), s=0.5):
    da = xa.load_dataarray(file_umap)
    postdata = xa.load_dataarray(file_postdata)
    
    out_tmp = postdata.isel(dict(key=0, type=0, pop=0))
    
    alpha_grid, beta_grid, _, w_grid = xa.broadcast(out_tmp["alpha"], out_tmp["beta"], out_tmp["rank"], out_tmp["w"])
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

    
@fm.figure_renderer("draw_dynamic_embed", reset=True, exts=[".png", ".svg"])
def draw_dynamic_embed(figsize=(10, 3), s=1):
    da = xa.load_dataarray(file_umap)
    postdata = xa.load_dataarray(file_postdata)
    
    fig = uf.get_figure((20, 25))
    axs = [fig.add_subplot(6,4,i+1) for i in range(24)]
    for ax in axs:
        ax.axis("off")
    
    out_tmp = postdata.isel(dict(key=0, type=0, pop=0))
    alpha_grid, beta_grid, _, w_grid = xa.broadcast(out_tmp["alpha"], out_tmp["beta"], out_tmp["rank"], out_tmp["w"])
    keys = (r"$\alpha$", r"$\beta$", r"$\omega$")
    for n, x in enumerate((alpha_grid, beta_grid, w_grid)):
        plt.sca(axs[n])
        draw_values(x.data.flatten(), da, title=keys[n], s=s, vmin=np.min(x), vmax=np.max(x))
        
    target_key = ("ac2p_large", "tlag_large", "ac2p_1st", "tlag_1st", "cc1p", "tlag_cc")
    target_pop = ("F", "S")
    target_type = ("mean", "var")
    
    # amplitude
    # amopt = dict(vmin=0, vmax=0.6, cticks=(0, 0.3, 0.6), cmap='jet')
    # asopt = dict(vmin=0, vmax=0.1, cticks=(0, 0.05, 0.1), cmap="jet")
    # fmopt = dict(vmin=30, vmax=70, cticks=(30, 50, 70), cmap="jet")
    # fsopt = dict(vmin=0, vmax=0.004, cticks=(0, 0.002, 0.002), cmap="jet")
    
    nstack = 4
    for k1 in target_key:
        for k2 in target_pop:
            for k3 in target_type:
                if "cc" in k1 and k2 == "S": continue
                
                x = sel_params(postdata, key=k1, pop=k2, type=k3)
                
                if "ac" in k1:
                    if "large" in k1:
                        kt = r"$AC^{%s}_{M}$"%(k2)
                    else:
                        kt = r"$AC^{%s}_1$"%(k2)
                    if k3 == "mean":
                        opt = dict(vmin=0, vmax=0.6, cticks=(0, 0.3, 0.6), cmap='jet')
                    else:
                        opt = dict(vmin=0, vmax=0.1, cticks=(0, 0.05, 0.1), cmap="jet")
                elif "tlag" in k1 and "cc" not in k1:
                    if "large" in k1:
                        kt = r"$f^{%s}_{M}$"%(k2)
                    else:
                        kt = r"$f^{%s}_{1}$"%(k2)
                    if k3 == "mean":
                        x = 1/x
                        opt = dict(vmin=30, vmax=70, cticks=(30, 50, 70), cmap="jet")
                    else:
                        opt = dict(vmin=0, vmax=0.01, cticks=(0, 0.005, 0.01), cmap="jet")
                elif "cc" in k1:
                    if "1p" in k1:
                        kt = r"$C_M$"
                        if k3 == "mean":
                            opt = dict(vmin=0, vmax=1, cticks=(0, 0.5, 1), cmap="jet")
                        else:
                            opt = dict(vmin=0, vmax=0.1, cticks=(0, 0.05, 0.1), cmap="jet")
                    else:
                        kt = r"$\tau_C$"
                        if k3 == "mean":
                            opt = dict(vmin=-0.01, vmax=0.01, cticks=(-0.01, 0, 0.01), cmap="jet")
                        else:
                            opt = dict(vmin=0, vmax=0.006, cticks=(0, 0.003, 0.006), cmap="jet")

                else:
                    print(k1, k2, k3)
                    raise ValueError("Unexpected key combination")

                if k3 == "mean":
                    kt = "E[" + kt + "]"
                else:
                    kt = r"$\sigma$[" + kt + "]"
                    x[x<0] = 0
                    x = np.sqrt(x)
                        
                plt.sca(axs[nstack])
                draw_values(x, da, s=s, title=kt, **opt)
                nstack += 1
        
    plt.tight_layout()    
    
    return fig
    
    
    
    # plt.figure(figsize=figsize, dpi=120)
    # plt.scatter(da[:,0], da[:,1], s=s, edgecolor=edgecolor, cmap=cmap, c=d, vmin=vmin, vmax=vmax, rasterized=True, **plot_opt)
    # print(postdata.keys())
    # print(postdata.alpha.shape)
    # print(postdata.shape)
    # print(da.shape)
    # x_grid, y_grid, z_grid = xa.broadcast(da['x'], da['y'], da['z'])
    
    # print(out.shape)
    
    
    
    # print(x_grid.shape)



if __name__ == "__main__":
    # draw_structure_embed()
    draw_dynamic_embed()