import matplotlib.pyplot as plt
import numpy as np
from functools import partial


""" Parameters for plot """

# cs = ["#d10000", "#0021ea", "#ffc9c0", "#b5bcea"]
cs = ["#d10000", "#0021ea", "#510404", "#193559"]


fpeaks = [[27, 40],
          [25, 43],
          [25, 38],
          [33, 60],
          [30, 66],
          [25, 55],
          [-1, 66],
          [35, 70]]


def set_colorset(cs_new):
    global cs
    
    print("color set is changed from", end=" ")
    print(cs)
    print("to")
    print(cs_new)
    
    cs = cs_new


def draw_with_err(t, xset, p_range=(5, 95), tl=None, linestyle="-", c='k', avg_method="median", label=None):
    """
    t: time
    xset: (N x T)
    """
    
    if tl is None:
        idt = np.ones_like(t, dtype=bool)
    else:
        idt = (t >= tl[0]) & (t <= tl[1])
    
    _t = t[idt]
    _xset = xset[:, idt]
    N = xset.shape[0]
    
    if avg_method == "median":
        xtop = np.percentile(_xset, p_range[1], axis=0)
        xbot = np.percentile(_xset, p_range[0], axis=0)
        x50  = np.median(_xset, axis=0)
    else:
        x50 = np.average(_xset, axis=0)
        s = np.std(_xset, axis=0)
        xtop = x50 + 1.65*s # 5%
        xbot = x50 - 1.65*s # 95%
        # xtop = x50 + 1.96*s/np.sqrt(N) # 5%
        # xbot = x50 - 1.96*s/np.sqrt(N) # 95%
    
    # x50 = xset.mean(axis=0)
    # s = xset.std(axis=0)*1.96/np.sqrt(xset.shape[0])
    # xtop = x50 + s
    # xbot = x50 - s

    plt.plot(_t, x50, c=c, lw=1.5, linestyle=linestyle, label=label)
    plt.fill_between(_t, xtop, xbot, color=c, alpha=0.2, edgecolor=c,
                    linewidth=0.5, linestyle="--")
    
    
def show_te_summary(te_data, figsize=(3.5, 3), dpi=120, xl=None, yl=None,
                    title=None, key="te", xlb=r"$\tau$ (ms)", ylb=r"$TE$ (bits)",
                    avg_method="median",
                    subtract_surr=False):
    
    te_labels = (
        r"$TE_{F \rightarrow S}$",
        r"$TE_{S \rightarrow F}$",
        r"$TE^{surr}_{F \rightarrow S}$",
        r"$TE^{surr}_{S \rightarrow F}$"
    )
    
    assert avg_method in ("mean", "median")
    _draw_err = partial(draw_with_err, avg_method=avg_method, tl=xl)
    
    fig = None
    if figsize is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    
        
    if subtract_surr:
        m0 = np.median(te_data["%s_surr"%(key)][:,0,:], axis=0)
        m1 = np.median(te_data["%s_surr"%(key)][:,1,:], axis=0)
        
        _draw_err(te_data["tlag"], te_data[key][:,0,:]-m0, c=cs[0], label=te_labels[0]+"-"+te_labels[2])
        _draw_err(te_data["tlag"], te_data[key][:,1,:]-m1, c=cs[1], label=te_labels[1]+"-"+te_labels[3])
        
        _draw_err(te_data["tlag"], te_data["%s_surr"%(key)][:,0,:]-m0, c=cs[2], linestyle=None)
        _draw_err(te_data["tlag"], te_data["%s_surr"%(key)][:,1,:]-m1, c=cs[3], linestyle=None)
    
    else:
        _draw_err(te_data["tlag"], te_data[key][:,0,:], c=cs[0], label=te_labels[0])
        _draw_err(te_data["tlag"], te_data[key][:,1,:], c=cs[1], label=te_labels[1])
        
        _draw_err(te_data["tlag"], te_data["%s_surr"%(key)][:,0,:], c=cs[2], linestyle='--', label=te_labels[2])
        _draw_err(te_data["tlag"], te_data["%s_surr"%(key)][:,1,:], c=cs[3], linestyle='--', label=te_labels[3])
    
    
    plt.xlim(xl)
    plt.ylim(yl)
    plt.xlabel(xlb, fontsize=16)
    plt.ylabel(ylb, fontsize=16)
    plt.title(title, fontsize=16)
    
    plt.legend(loc="upper right", edgecolor='none')
    
    return fig


def show_spec_summary(spec_data, figsize=(3.5, 3), dpi=120, xl=None, yl=None,
                      xlb=None, ylb=None, title=None, avg_method="median"):
    cs = ["#d10000", "#0021ea"]
    
    assert avg_method in ("mean", "median")
    _draw_err = partial(draw_with_err, avg_method=avg_method, tl=xl)
    
    fig = None
    if figsize is not None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    
    _draw_err(spec_data["fpsd"], spec_data["spec_boot"][:,0,:], c=cs[0], label=r"$A_{F}$")
    _draw_err(spec_data["fpsd"], spec_data["spec_boot"][:,1,:], c=cs[1], label=r"$A_{S}$")
    
    plt.xlim(xl)
    plt.ylim(yl)
    plt.xlabel(r"$frequency$ (Hz)", fontsize=16)
    plt.ylabel(r"$A$ (a.u.)", fontsize=16)
    plt.title(title, fontsize=16)
    
    plt.legend(loc="upper right", edgecolor='none')
    
    return fig


def show_te_summary_2d(te_data, tl=None, vmax=None, vmin=None, vdmax=None):
    # row: \tau_{src}, col: \tau_{dst}

    te = np.mean(te_data["te"], axis=0)
    te_surr = np.mean(te_data["te_surr"], axis=0)

    dte = [
        te[0]-te_surr[0],
        (te[1]-te_surr[1]).T
    ]
    
    titles = [
        r"$\Delta TE_{F \rightarrow S}$",
        r"$\Delta TE_{F \rightarrow S} - \Delta TE_{S \rightarrow F}$",
        r"$\Delta TE_{S \rightarrow F}$"
    ]

    extent = [te_data["tlag"][0], te_data["tlag"][-1]]*2

    fig = plt.figure(figsize=(14, 3))
    for ntp in range(3):
        plt.subplot(1,3,ntp+1)
        if ntp == 1:
            if vdmax is None:
                _vmax, _vmin = None, None
            else:
                _vmax, _vmin = vdmax, -vdmax
            
            plt.imshow(dte[0]-dte[1], origin="lower", extent=extent,
                        cmap="RdBu_r", vmax=_vmax, vmin=_vmin)
        else:
            plt.imshow(dte[ntp//2], origin="lower", extent=extent,
                        cmap="jet", vmax=vmax, vmin=vmin)
            
        plt.colorbar()
        plt.xlim(tl)
        plt.ylim(tl)
        plt.ylabel(r"$\tau_F$ (ms)", fontsize=15)
        plt.xlabel(r"$\tau_S$ (ms)", fontsize=15)
        plt.title(titles[ntp], fontsize=15)

    return fig


orders = [0, 2, 10, 6, 14, 4, 5, 7, 15, 8, 13]


def gen_background(dpi=200):
    
    import oscdetector as od
    
    we = 0.01
    wc = 0.025
    wl = 0.01
    num_c = 8

    fig = plt.figure(figsize=(16, 9), dpi=dpi)

    # boundaries
    w = 1 - we*2 - wc
    h = 1 - (we + wc)
    dw = w / len(orders)

    for n in range(len(orders)):
        plt.axes(position=(we+wc+dw*n, h, dw, wc))
        lb = od.get_motif_labels()[orders[n]]
        plt.text(0.5, 0.5, lb, va="center", ha="center", fontsize=12)
        plt.xticks([])
        plt.yticks([])

    # h = 1 - (2*we + wc)
    dh = (h - we) / num_c
    for n in range(num_c):
        # plt.axes(position=(we, 1-we-wc-dh*(n+1), wc, dh))
        plt.axes(position=(we, 1-we-wc-dh*(n+1), wc, dh))
        lb = "%d"%(n+1)
        plt.text(0.5, 0.5, lb, va="center", ha="center", fontsize=16)
        plt.xticks([])
        plt.yticks([])

    # generate axis
    wtot = 1 - 2*we - wc
    htot = 1 - 2*we - wc

    dw = wtot/len(orders)
    dh = htot/num_c
    
    coords = []

    for nc in range(num_c):
        coords.append([])
        h0 = 1 - we - wc - dh*(nc+1)
        for i in range(len(orders)):
            w0 = we + wc + dw*i

            plt.axes(position=(w0, h0, dw, dh))
            plt.xticks([]); plt.yticks([])
            
            coords[-1].append((w0, h0, dw, dh))

            # plt.axes(position=(w0+wl, h0+wl, dw-2*wl, dh-2*wl))
            
    return fig, coords


def draw_freq_indicator(cid, yl=None):
    if yl is None: yl = [0, 1]
    
    fopt = dict(va="center", ha="center", fontsize=10)
    lopt = dict(linestyle=":", linewidth=1, alpha=0.5)
    tp_labels = (r"$T_s/2$", r"$T_s$", r"$T_f/2$", r"$T_f$")
    
    for tp in range(2):
        f0 = fpeaks[cid-1][tp]
        if f0 == -1: continue
        plt.vlines(1e3/f0, yl[0], yl[1], color=cs[1-tp], **lopt)
        plt.vlines(1e3/f0/2, yl[0], yl[1], color=cs[1-tp], **lopt)
        if 1e3/f0*3/2 < 40:
            plt.vlines(1e3/f0/2*3, yl[0], yl[1], color=cs[1-tp], **lopt)
        
        h = yl[0] + (yl[1]-yl[0])/30
        plt.text(1e3/f0/2, h, tp_labels[2*tp], **fopt)
        plt.text(1e3/f0, h, tp_labels[2*tp+1], **fopt)
        
        
def draw_syn_indicator(yl=None):
    if yl is None: yl = [0, 1]
    tau1 = [0.3, 0.5, 1] # E, If, Is
    tau2 = [1. , 2.5, 8]
    cs_set = ("#b02323", "#004fb2", "#107f0a")
    tp_set = [tau1[n]*tau2[n]/(tau2[n]-tau1[n])*np.log(tau2[n]/tau1[n]) for n in range(3)]
    labels = (r"$t^*_{E}$",
              r"$t^*_{I_F}$",
              r"$t^*_{I_S}$")
    
    fopt = dict(va="center", ha="center", fontsize=10)
    lopt = dict(linestyle="dotted", linewidth=1)
    
    for n in range(3):
        cs, tp, lb = cs_set[n], tp_set[n], labels[n]
        plt.vlines(tp, yl[0], yl[1], color=cs, **lopt)
        h = yl[0] + (yl[1]-yl[0])/30 * (5-2*n)
        plt.text(tp, h, lb, color=cs, **fopt)
    
    
    
def draw_cfc_indicator(cid, yl=None):
    if yl is None: yl = [0, 1]
    dp_set = (-1/2, 0, -1/4, 1/4, 1/8, 1/4, 0, 1/4) # \time \pi; \phi^S_S(V^F_f)
    # -: Ff -> Ss / +: Ff <- Ss
    
    dt = dp_set[cid-1]/2 * 1e3/fpeaks[cid-1][0]
    if dt < 0: # f -> s
        c = cs[0]
        txt = r"$T^{CFC}_{F \rightarrow S}$"
    elif dt > 0: # s -> f
        c = cs[1]
        txt = r"$T^{CFC}_{S \rightarrow F}$"
    else:
        return
        # r"$T_{\phi^S_s, V^F_f}$"
    plt.vlines(abs(dt), yl[0], yl[1], color=c, linestyle="-", linewidth=0.5, alpha=0.8)
    h = yl[0] + (yl[1]-yl[0])/30 * 28
    plt.text(abs(dt), h, txt, color=c, va="center", ha="center", fontsize=10)
    
    # plt.vlines(dt, yl[0], yl[1], color=)
    
    