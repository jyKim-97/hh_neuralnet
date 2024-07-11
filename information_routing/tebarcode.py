import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import stats


def draw_barcode(binfo, cmap="Reds", dots="kp",
                 figsize=(6, 1), xlb=r"$\tau$ (ms)"):
    
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.gcf()
    # fig = None
    
    tbar = binfo["tbar"]
    vmax = np.percentile(binfo["barcode"][binfo["barcode"] > 0], 80)
    
    plt.yticks([])

    dt = tbar[1] - tbar[0]
    extent = (tbar[-1]+dt/2, -dt/2, 0, 1)
    # (-tbar[-1]-dt/2, dt/2, 0, 1)
    plt.imshow(binfo["barcode"][:,::-1], aspect="auto", cmap=cmap,
            extent=extent, vmax=vmax, vmin=-vmax)

    bpeaks = binfo["bpeaks"]
    for ntp in range(2):
        if len(bpeaks[ntp]) == 0: continue
        plt.plot(tbar[bpeaks[ntp]], [0.75-0.5*ntp]*len(bpeaks[ntp]), dots)
        
    plt.gca().yaxis.tick_right()
    plt.yticks([0.25, 0.75], labels=(r"$S \rightarrow F$", r"$F \rightarrow S$"))
    plt.plot([tbar[-1], tbar[0]], [0.5, 0.5], 'k-', lw=0.2, alpha=0.5)
    plt.xlabel(xlb, fontsize=14)
    
    return fig
    

def get_barcode(te_data, stat_method='conf'):
    sig_arr = stats.te_stat_test(te_data, method=stat_method, alpha=0.05)
    hinfo = find_te_hill(te_data, verbose=False)
    
    nw = 3
    bpeaks = []
    is_bar = np.zeros(te_data["te"].shape[1:], dtype=bool)
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
                is_bar[ntp][nset] = True
                bpeaks[ntp].append(n0)

        bpeaks[ntp] = np.array(bpeaks[ntp])
        
    tbar = te_data["tlag"]
    barcode = np.zeros(te_data["te"].shape[1:])
    m = te_data['te_surr'].mean(axis=0, keepdims=True)
    barcode[is_bar] = (te_data["te"] - m).mean(axis=0)[is_bar]

    # fill
    if tbar[0] != 0:
        dt = tbar[1] - tbar[0]
        nt = int(tbar[0]/dt)
        bp = np.zeros([te_data["te"].shape[1], nt])
        
        barcode = np.hstack((bp, barcode))
        tbar = np.concatenate((np.arange(0, tbar[0], dt), tbar))
        bpeaks = [b + nt for b in bpeaks]
    
    if "info" not in te_data.keys():
        te_data["info"] = None
    info = dict(te_info=te_data["info"], stat_method=stat_method)
    return dict(barcode=barcode, tbar=tbar, bpeaks=bpeaks, info=info)


def find_te_hill(te_data, verbose=False):
    te = te_data["te"].mean(axis=0)
    hinfo = find_hill(te)
    
    if verbose:
        t = te_data['tlag']
        plt.figure(figsize=(4, 3))
        for ntp in range(2):
            for c in hinfo["id_set"][ntp]:
                idh = hinfo["id"][ntp] == c
                plt.plot(t[idh], te[ntp][idh])
                
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
        idp_set, _ = find_peaks(xs[n])
        id_peak.append(list(idp_set))
        
        for idp in idp_set:
            is_low = fill_lower(xs[n], idp)
            hill_id[n][is_low] = idh
            idh += 1
            
        if np.all(hill_id[n][-3:] == 0):
            n0 = np.where(hill_id[n] > 0)[0][-1]
            hill_id[n][n0:] = idh
            id_peak[-1].append(np.argmax(xs[n][n0:]) + n0)
        
        if np.all(hill_id[n][:3] == 0):
            n0 = np.where(hill_id[n] > 0)[0][0]
            id_peak[-1] = [np.argmax(xs[n][:n0])] + id_peak[-1]
            hill_id[n] += 1
            
        hill_id_set.append(np.unique(hill_id[n]))
        
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