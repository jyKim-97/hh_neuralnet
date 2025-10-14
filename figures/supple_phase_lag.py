import numpy as np
import matplotlib.pyplot as plt
import xarray as xa

from tqdm import tqdm
import figure_manager as fm
import os
import sys
sys.path.append("../include/pytools")
import utils_fig as uf
import oscdetector as od
uf.set_plt()

reset = False

fdir_coupling = "../coupling/postdata/cfc/"
cw_set = ((4, 15), (4, 10), (7, 10), (7, 5))
vmax_set = (0.2, 0.2, 0.1, 0.1)


def compute_corr_from_joint(joint_prob: xa.DataArray):
    
    phi_x = joint_prob.nx.values
    phi_y = joint_prob.ny.values
    assert len(phi_x) == len(phi_y), "x and y should have the same dim"
    
    lag2idx = {dn:n for n, dn in enumerate(np.arange(-len(phi_x)+1, len(phi_x)))}
    num_count = np.zeros((len(lag2idx)))
    px = joint_prob / joint_prob.sum()
    phi_lags = np.zeros(len(lag2idx))
    for i in range(len(phi_x)):
        for j in range(len(phi_y)):
            idx = lag2idx[i-j]
            phi_lags[idx] += px.isel(dict(nx=i, ny=j))
            num_count[idx] += 1
    
    assert np.all(phi_lags[num_count==0] == 0)
    num_count[num_count==0] = 1
    phi_lags = phi_lags / num_count
    phi_lags *= len(phi_x)
    
    dp = (phi_x[1:] - phi_x[:-1]).mean()
    dphi = np.sort(np.array(list(lag2idx.keys())) * dp)
    
    return phi_lags, dphi


def compute_corr(comap):
    phi_set  = []
    for n in range(len(comap.fx.values)):
        joint = comap.isel(dict(fx=n, fy=n))
        phi_lags, dphi = compute_corr_from_joint(joint)
        phi_set.append(phi_lags)
        # fset.append(comap.fx.values[n])
    # fset = np.array(fset)
    phi_set = np.array(phi_set)
    return phi_set, dphi


def build_phase_lag(cw_set=None, tmin=-50, tmax=50, nbin=21, use_cache=True):
    
    fcache = "./postdata/phi_lag_cache.nc"
    flag_compute = True
    if use_cache:
        if not os.path.exists(fcache):
            flag_compute = True
        else:
            data = xa.load_dataarray(fcache)
            flag_compute = False
            if data.tlag.values[0] != tmin or data.tlag.values[-1] != tmax:
                flag_compute = True
            elif len(data.tlag) != nbin:
                flag_compute = True
            elif tuple(data.cset) != tuple([cw[0] for cw in cw_set]):
                flag_compute = True
            elif tuple(data.wset) != tuple([cw[1] for cw in cw_set]):
                flag_compute = True
    
    if not use_cache or flag_compute:
        print("Cache does not exist, compute phase lag")
        phi_interp_set = []
        tq = np.linspace(tmin, tmax, nbin)
        for cid, wid in tqdm(cw_set):
            comap = xa.load_dataset(os.path.join("../coupling/postdata/cfc","cfc_pp_%d%02d.nc"%(cid, wid)))
            comap_sub = comap.sel(dict(nax="fs")).prob
            phi_lags, dphi = compute_corr(comap_sub) # dphi < 0: F lead, > 0: S lead
            fset = comap_sub.fx.values
            
            phi_lags_q = np.zeros((len(phi_lags), len(tq)))
            for i in range(len(fset)):
                tlag = dphi/fset[i]*1e3
                phi_lags_q[i] = np.interp(tq, tlag, phi_lags[i])
            phi_interp_set.append(phi_lags_q)
        phi_interp_set = np.array(phi_interp_set)
        
        data = xa.DataArray(
            phi_interp_set,
            coords=dict(ncw=np.arange(len(cw_set)), fset=fset, tlag=tq),
            # attrs=dict(cw_set=cw_set)
            attrs=dict(cset=tuple([cw[0] for cw in cw_set]),
                       wset=tuple([cw[1] for cw in cw_set]))
        )
        data.to_netcdf(fcache)
        
    return data
    



@fm.figure_renderer("phase_lag", reset=False, exts=(".png", ".svg"))
def draw_phase_lag(figsize=(12, 3), cw_set=None, tmin=-50, tmax=50, vmax_set=None, use_cache=True):

    data = build_phase_lag(cw_set, tmin=tmin, tmax=tmax, nbin=51, use_cache=use_cache)
    N = len(cw_set)
    assert N == 4
    
    fig = uf.get_figure(figsize)
    # axs = uf.get_custom_subplots(w_ratio=[1,0.05]*2, h_ratio=[1,1], 
    #                              w_blank_interval_set=[0.02, 0.15, 0.02], 
    #                              h_blank_interval=0.15)
    axs = uf.get_custom_subplots(w_ratio=[1]*N, h_ratio=[1], 
                                 w_blank_interval=0.12,
                                #  w_blank_interval_set=[0.02, 0.15, 0.02], 
                                 h_blank_interval=0.15)
    tset, fset = data.tlag, data.fset
    lb_set = []
    for n in range(N):
        # i, j = n%2, n//2
        # ax = axs[n%2][2*(n//2)]
        # cbar_ax = axs[n%2][2*(n//2)+1]
        ax = axs[0][n]
        # cbar_ax.axis("off")

        vmax = vmax_set[n]
        phi_lag = data.isel(dict(ncw=n))
        im_obj = ax.imshow(phi_lag, extent=(tset[0], tset[-1], fset[0], fset[-1]), aspect="auto",
                           origin="lower", cmap="RdBu_r", vmax=vmax, vmin=0,
                           interpolation="none")
        # cbar = plt.colorbar(im_obj, cax=cbar_ax, ticks=np.linspace(0, vmax, 3), shrink=0.6)
        # cbar = ax.colorbar(im_obj, ticks=np.linspace(0, vmax, 3), shrink=0.6)
        cbar = plt.colorbar(im_obj, ticks=np.linspace(0, vmax, 3), shrink=0.6)
        # # if n // 2 == 1:
        if n == N-1:
            cbar.set_label(r"$CC_{V^F_{[f_1, f_2]}, V^S_{[f_1, f_2]}}(\tau)$", 
                           fontsize=6, rotation=270, labelpad=10)
            # cbar.ax.set_title(r"$CC_{V^F_{[f_1, f_2]}, V^S_{[f_1, f_2]}}(\tau)$", 
            #                   )
        # cbar.ax.set_title(r"$CC$")
        
        cid, wid = cw_set[n]
        lb = "#%d"%(cid) + od.get_motif_labels("ver2")[wid]
        lb_set.append(lb)
        ax.set_title(lb)
        ax.set_xlabel(r"$\tau$ (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xticks(np.linspace(tmin, tmax, 5))
        
        y0 = fset[0] + (fset[-1]-fset[0])/15
        opt = dict(va="center", ha="center", fontsize=5)
        ax.text(tmin/2, y0, "F lead", **opt)
        ax.text(tmax/2, y0, "S lead", **opt)
        
    # for i in range(N//2):
    #     ax = axs[2][2*i]
    #     axs[2][2*i+1].axis("off")
    #     for j in range(2):
    #         phi_lag = data.isel(dict(ncw=2*i+j)).values
    #         # tlag_max.append(np.argmax(phi_lag, axis=1))
    #         tlag_max = tset[np.argmax(phi_lag, axis=1)]
    #         ax.plot(tlag_max, fset, label=lb_set[2*i+j])
    #     ax.set_xlabel(r"$\tau_{max}$ (ms)")
    #     ax.set_ylabel("Frequency (Hz)")
    #     ax.axvline(0, color='k', linestyle='--', linewidth=1)
    #     ax.set_xlim([tmin, tmax])
    #     ax.set_ylim([20, 80])
    #     ax.legend(fontsize=5, loc="center right", edgecolor="none")
    
    return fig
        
    
    # cache_set = []
    
    # for n, (cid, wid) in enumerate(cw_set):
    #     comap = xa.load_dataset(os.path.join("../coupling/postdata/cfc","cfc_pp_%d%02d.nc"%(cid, wid)))
    #     comap_sub = comap.sel(dict(nax="fs")).prob
    #     phi_lags, dphi = compute_corr(comap_sub)
    #     fset = comap_sub.fx.values
        
    #     phi_lags_q = np.zeros((len(phi_lags), len(tq)))
    #     for i in range(len(fset)):
    #         tlag = dphi/fset[i]*1e3
    #         phi_lags_q[i] = np.interp(tq, tlag, phi_lags[i])
    #     cache_set.append(phi_lags_q)
        
    #     im_obj = axs[0][n].imshow(phi_lags_q, extent=(tq[0], tq[-1], fset[0], fset[-1]), 
    #                      aspect="auto", origin="lower", cmap="RdBu_r",
    #                      vmin=0, vmax=vmax)
    #     plt.colorbar(im_obj, ax=axs[0][n])
        
    
        
    # return fig


if __name__ == "__main__":
    draw_phase_lag(cw_set=cw_set, vmax_set=vmax_set)