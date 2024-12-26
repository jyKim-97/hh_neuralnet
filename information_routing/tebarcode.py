import numpy as np
from numba import njit
# import matplotlib.pyplot as plt
import stats
# import matplotlib.ticker as mticker


# def draw_barcode(binfo, cmap="RdBu_r", dots="kp",
#                  vmax=None, vmin=None,
#                  figsize=(6.5, 1), ax=None, xlb=r"$\tau$ (ms)",
#                  show_cbar=False):
    
#     pos_cbar = (0.04, 0.1, 0.02, 0.85)
#     pos_ax = (0.08, 0.1, 0.72, 0.85)
    
#     if ax is None:
#         fig = plt.figure(figsize=figsize)
#         ax_cbar = plt.axes(position=pos_cbar)
#         ax_main = plt.axes(position=pos_ax)
#     else:
#         fig = None
#         ax.axis("off")
#         plt.sca(ax)
#         ax_cbar = ax.inset_axes(pos_cbar)
#         ax_main = ax.inset_axes(pos_ax)
      
#     plt.axes(ax_main)
#     tbar = binfo["tbar"]
    
#     if vmax is None:
#         if vmin is None:
#             vmax = np.percentile(binfo["barcode"][binfo["barcode"] > 0], 80)
#             vmin = -vmax
#         else:
#             vmax = -vmin
            
#     if vmin is None:
#         vmin = -vmax
    
#     plt.yticks([])

#     dt = tbar[1] - tbar[0]
#     extent = (tbar[-1]+dt/2, -dt/2, 0, 1)
#     # (-tbar[-1]-dt/2, dt/2, 0, 1)
#     plt.imshow(binfo["barcode"][:,::-1], aspect="auto", cmap=cmap,
#                extent=extent, vmax=vmax, vmin=-vmax)

#     bpeaks = binfo["bpeaks"]
#     for ntp in range(2):
#         if len(bpeaks[ntp]) == 0: continue
#         plt.plot(tbar[bpeaks[ntp]], [0.75-0.5*ntp]*len(bpeaks[ntp]), dots)
        
#     plt.gca().yaxis.tick_right()
#     plt.yticks([0.25, 0.75], labels=(r"$S \rightarrow F$", r"$F \rightarrow S$"))
#     plt.plot([tbar[-1], tbar[0]], [0.5, 0.5], 'k-', lw=0.2, alpha=0.5)
#     plt.xlabel(xlb, fontsize=14)
    
#     xl = plt.xlim()
#     xt, xtt = plt.xticks()
#     assert 0 in xt
#     id0 = np.where(xt == 0)[0][0]
#     xtt[id0].set_text("NOW")
#     plt.xticks(xt, labels=xtt)
#     plt.xlim(xl)
    
#     if show_cbar:
#         plt.colorbar(location="left", ax=ax_main, cax=ax_cbar,
#                      format=mticker.FormatStrFormatter("%d %%"))
        
#     else:
#         ax_cbar.axis("off")
    
#     return fig


def is_diff_te(te_data, stat_method="conf", nw=3):
    sig_arr = stats.te_stat_test(te_data, method=stat_method, alpha=0.05)
    hinfo = find_te_hill(te_data, verbose=False)
    
    bpeaks = []
    is_diff = np.zeros(te_data["te"].shape[1:], dtype=bool)
    
    for ntp in range(2):
        bpeaks.append([])
        num = len(hinfo['id_set'][ntp])
        for nh in range(num):
            n0 = hinfo["id_peak"][ntp][nh]
            nset = np.arange(n0-nw, n0+nw+1)
            nset = nset[(nset >= 0) & (nset < te_data["te"].shape[-1])]

            if sig_arr[ntp][nset].sum() >= len(nset)-2:
                nset = np.arange(n0-1, n0+2)
                nset = nset[(nset >= 0) & (nset < te_data["te"].shape[-1])]
                is_diff[ntp][nset] = True
                bpeaks[ntp].append(n0)

        bpeaks[ntp] = np.array(bpeaks[ntp])
        
    return is_diff, bpeaks


def get_barcode_boost(te_data, te_base, stat_method="conf"):
    is_bar1, bpeaks_te = is_diff_te(te_data, stat_method=stat_method)
    
    # dte_data = te_data["te"] - te_data["te_surr"].mean(axis=0, keepdims=True)
    # dte_base = te_base["te"] - te_base["te_surr"].mean(axis=0, keepdims=True)
    
    dte_data = te_data["te"]
    dte_base = te_base["te"]
    
    te_tmp = dict(te=dte_data, te_surr=dte_base, tlag=te_data["tlag"], info=te_data["info"])
    is_bar2, _ = is_diff_te(te_tmp)
    
    is_bar = is_bar1 & is_bar2
    bpeaks = [np.array([nb for nb in bpeaks_te[n] if is_bar[n][nb]]) for n in range(2)]
    
    return get_barcode(te_tmp, is_bar=is_bar, bpeaks=bpeaks, percentile=True)
    

def get_barcode(te_data, stat_method='conf', bth=2, is_bar=None, bpeaks=None, percentile=False):
    
    nw = 3
    if is_bar is None:
        is_bar, bpeaks = is_diff_te(te_data, stat_method=stat_method, nw=nw)
        
    if "info" not in te_data.keys():
        te_data["info"] = None
        
    tbar = te_data["tlag"]
    barcode = np.zeros(te_data["te"].shape[1:])
    m = te_data['te_surr'].mean(axis=0, keepdims=True)
    if np.sum(is_bar) == 0:
        return dict(barcode=barcode, tbar=tbar, bpeaks=[[], []], info=te_data["info"])
    
    # # remove periodic values
    # te_val = (te_data["te"] - m).mean(axis=0)
    # for ntp in range(2):
    #     bp, bp_out = _remove_periodic_barcode(bpeaks[ntp], te_val[ntp], bth=bth)
    #     for n in bp_out:
    #         is_bar[ntp, n-nw:n+nw+1] = False
    #     # print(bpeaks[ntp], bp)
    #     bpeaks[ntp] = np.array(bp)
            
    if percentile:
        te = te_data["te"].mean(axis=0)[is_bar]
        tem = m.mean(axis=0)[is_bar]
        barcode[is_bar] = (te - tem) / np.max(([np.abs(te), np.abs(tem)])) * 100
    else:
        barcode[is_bar] = (te_data["te"] - m).mean(axis=0)[is_bar]

    # fill
    if tbar[0] != 0:
        dt = tbar[1] - tbar[0]
        nt = int(tbar[0]/dt)
        bp = np.zeros([te_data["te"].shape[1], nt])
        
        barcode = np.hstack((bp, barcode))
        tbar = np.concatenate((np.arange(0, tbar[0], dt), tbar))
        bpeaks = [b + nt for b in bpeaks]
        
    info = dict(te_info=te_data["info"], stat_method=stat_method)
    return dict(barcode=barcode, tbar=tbar, bpeaks=bpeaks, info=info)



def _remove_periodic_barcode(bpeaks, yval, bth=2):
    bpeaks = list(bpeaks)[:]
    bpeaks_out = []
    
    n = 0
    while n < len(bpeaks)-1:
        if bpeaks[n] < 5:
            n += 1
            continue
        db = bpeaks[n+1] - bpeaks[n]

        for i in np.arange(n+2, len(bpeaks))[::-1]:
            c1 = (bpeaks[i] - bpeaks[n])%db <= bth
            c2 = (bpeaks[i] - bpeaks[n] + bth)%db <= bth
            
            if (c1 or c2) and (yval[bpeaks[i]] <= yval[bpeaks[n]]):
                bpeaks_out.append(bpeaks.pop(i))
        n += 1
    
    if len(bpeaks) > 2:
        db1 = bpeaks[1]-bpeaks[0]
        db2 = bpeaks[2]-bpeaks[1]
        
        if abs(db1 - db2) <= bth:
            bpeaks_out.append(bpeaks.pop(2))
        
    return bpeaks, bpeaks_out


def find_te_hill(te_data, verbose=False):
    te = te_data["te"].mean(axis=0)
    hinfo = find_hill(te)
    
    if verbose:
        t = te_data['tlag']
        # plt.figure(figsize=(4, 3))
        for ntp in range(2):
            for c in hinfo["id_set"][ntp]:
                idh = hinfo["id"][ntp] == c
                # plt.plot(t[idh], te[ntp][idh])
                
    return hinfo


def find_hill(xs):
    from scipy.signal import find_peaks

    assert len(xs.shape) <= 2
    if len(xs.shape) == 1:
        N, nlen = 1, xs.shape[0]
        xs = np.reshape(xs, (1, -1))
    else:
        N, nlen = xs.shape
    
    hill_id = np.zeros_like(xs)
    hill_id_set = []
    id_peak = []
    
    for n in range(N):
        idh = 1
        idp_set, _ = find_peaks(xs[n], prominence=1e-3)
        id_peak.append(list(idp_set))
        
        for idp in idp_set:
            is_low = fill_lower(xs[n], idp)
            hill_id[n][is_low] = idh
            idh += 1
            
        if hill_id[n][-1] == 0:
            n0 = np.where(hill_id[n] > 0)[0][-1]
            if np.all(hill_id[n][-4:] == 0):
                hill_id[n][n0:] = idh
                id_peak[-1].append(np.argmax(xs[n][n0:]) + n0)
            else:
                hill_id[n][n0:] = idh-1
                
        if hill_id[n][0] == 0:
            n0 = np.where(hill_id[n] > 0)[0][0]
            if np.all(hill_id[n][:4] == 0):
                id_peak[-1] = [np.argmax(xs[n][:n0])] + id_peak[-1]
                hill_id[n] += 1
            else:
                hill_id[n][:n0] = 1
        
        # hill_id[n][hill_id[n] == 0] = np.nan
        id_unique = np.unique(hill_id[n])
        id_unique = id_unique[id_unique > 0]
        hill_id_set.append(id_unique)
        hill_id[n][hill_id[n] == 0] = np.nan
        
        assert len(hill_id_set[-1]) == len(id_peak[-1])
    assert not np.any(hill_id == 0)
            
    return dict(id=hill_id,
                id_set=hill_id_set,
                id_peak=id_peak)
            
                            
def fill_lower(x, n0):
    xl = fill_lower_dir(x, n0, ndir=-1)
    xr = fill_lower_dir(x, n0, ndir=1)
    return (xl == 1) | (xr == 1)

@njit
def fill_lower_dir(x, n0, ndir=-1):
    # ndir in (-1, 1)
    
    xfill = np.zeros(len(x))
    
    n_cur, n_prv = n0, n0+ndir
    while x[n_prv] <= x[n_cur]:
        xfill[n_cur] = 1
        n_cur += ndir
        n_prv += ndir
        
        if n_prv < 0 or n_prv >= len(x):
            break
    
    xfill[n_cur] = 1
    
    return xfill