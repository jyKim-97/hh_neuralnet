import matplotlib.pyplot as plt
import numpy as np
from functools import partial

import matplotlib.ticker as mticker


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


def draw_with_err(t, xset, p_range=(5, 95), tl=None, linestyle="-", c='k',
                  avg_method="median", label=None):
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

    plt.plot(_t, x50, c=c, lw=1.5, linestyle=linestyle, label=label)
    plt.fill_between(_t, xtop, xbot, color=c, alpha=0.2, edgecolor=c,
                    linewidth=0.5, linestyle="--")
    
    
def show_te_summary(te_data, figsize=(3.5, 3), dpi=120, ax=None, xl=None, yl=None,
                    title=None, key="te", xlb=r"$\tau$ (ms)", ylb=r"$TE$ (bits)",
                    avg_method="median",
                    stat_test=False,
                    subtract_surr=False):
    
    te_labels = (
        r"$TE_{F \rightarrow S}$",
        r"$TE_{S \rightarrow F}$",
        r"$TE^{surr}_{F \rightarrow S}$",
        r"$TE^{surr}_{S \rightarrow F}$"
    )
    
    assert avg_method in ("mean", "median")
    _draw_err = partial(draw_with_err, avg_method=avg_method, tl=xl)
    
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig = None
        plt.sca(ax)
    
    key_surr = "%s_surr"%(key)
    if subtract_surr:
        if avg_method == "median":
            ms = np.median(te_data[key_surr], axis=0, keepdims=True)
        else:
            ms = np.average(te_data[key_surr], axis=0, keepdims=True)
            
        xy = te_data[key] - ms
        xy_s = te_data[key_surr] - ms
        labels = (te_labels[0]+"-"+te_labels[2],
                  te_labels[1]+"-"+te_labels[3],
                  None, None)
    
    else:
        xy = te_data[key]
        xy_s = te_data[key_surr]
        labels = te_labels
    
    t = te_data["tlag"]
    _draw_err(t, xy[:,0,:], c=cs[0], label=labels[0])
    _draw_err(t, xy[:,1,:], c=cs[1], label=labels[1])
    _draw_err(t, xy_s[:,0,:], c=cs[2], linestyle='--', label=labels[2])
    _draw_err(t, xy_s[:,1,:], c=cs[3], linestyle='--', label=labels[3])
    
        
    if stat_test:
        from stats import conf_test
        
        is_sig = np.zeros((2, xy.shape[-1]), dtype=bool)
        for ntp in range(2):
            for nd in range(xy.shape[-1]):
                is_sig[ntp, nd] = conf_test(xy[:,ntp,nd], xy_s[:,ntp,nd], alpha=0.05)
            
        ms = xy.mean(axis=0)
        for i in range(2):
            plt.plot(t[is_sig[i]], ms[i][is_sig[i]], '.', c=cs[i], markersize=5)
    
    
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


def draw_indicator(xind, yl=None, color='k',
                   txt=None, htxt=None, fontsize=10,
                   linestyle="--", linewidth=1, alpha=0.5,
                   flip=False):
    
    if yl is None:
        yl = plt.ylim()
    
    if flip: xind = -xind

    plt.vlines(xind, yl[0], yl[1], color=color,
               linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    
    if txt is not None:
        if htxt is None:
            htxt = yl[0] + (yl[1]-yl[0])/30*2
        plt.text(xind, htxt, txt, va="center", ha="center",
                 fontsize=fontsize, color=color)
        

def draw_cfc_indicator(cid, yl=None, flip=False, h=None):
    _pi = np.pi
    
    trad = [0, 0]
    for i in range(2):
        f = max(fpeaks[cid-1][i], 30)
        trad[i] = 1e3/f/(2*_pi)
    
    # trad = [1e3/f/(2*_pi) for f in fpeaks[cid-1]]
    
    dt_set_tot = [[-_pi/2 * trad[0]],
                  [0],
                  [-_pi/4 * trad[0]],
                  [_pi/4 * trad[0]],
                  [_pi/8 * trad[0], -_pi/8 * trad[1]],
                  [_pi/4 * trad[0]],
                  [0],
                  [_pi/4 * trad[0], _pi*5/8 * trad[1]]]
    
    dt_set = dt_set_tot[cid-1]
    for dt in dt_set:
        if dt == 0: continue
        if dt < 0: # f -> s
            dt = -dt
            c = cs[0]
        else:
            c = cs[1]
            
        # print(dt)
        draw_indicator(dt, yl, color=c, linestyle="-.", txt=r"$T^{CFC}$",
                       flip=flip, htxt=h)
    

def draw_freq_indicator(cid, yl=None, flip=False, h=None,
                        lw=1, alpha=1):
    
    xl = plt.xlim()
    yl = plt.ylim() if yl is None else yl
    
    lopt = dict(linestyle="--", linewidth=lw, alpha=alpha)
    tp_labels = (r"$T_s/2$", r"$T_s$", r"$T_f/2$", r"$T_f$")
    
    for tp in range(2):
        f0 = fpeaks[cid-1][tp]
        if f0 == -1: continue
        
        c = cs[1-tp]
        h = yl[0] + (yl[1]-yl[0])/30 if h is None else h
        
        draw_indicator(1e3/f0/2, yl, color=c, txt=tp_labels[2*tp], htxt=h, flip=flip, **lopt)
        draw_indicator(1e3/f0, yl, color=c, txt=tp_labels[2*tp+1], htxt=h, flip=flip, **lopt)
        
        n = 3
        while 1e3/f0*n/2 < max(xl):
            draw_indicator(1e3/f0/2*n, yl, color=c, flip=flip, **lopt)
            n += 1
            
    plt.xlim(xl)
        
        
        
def draw_syn_indicator(yl=None, flip=None, h=None):
    
    xl = plt.xlim()
    if yl is None: yl = [0, 1]
    tau1 = [0.3, 0.5, 1] # E, If, Is
    tau2 = [1. , 2.5, 8]
    cs_set = ("#b02323", "#004fb2", "#107f0a")
    tp_set = [tau1[n]*tau2[n]/(tau2[n]-tau1[n])*np.log(tau2[n]/tau1[n]) for n in range(3)]
    labels = (r"$t^*_{E}$",
              r"$t^*_{I_F}$",
              r"$t^*_{I_S}$")
    
    for n in range(3):
        h = yl[0] + (yl[1]-yl[0])/30 * (5-2*n) if h is None else h
        draw_indicator(tp_set[n], yl, color=cs_set[n],
                       txt=labels[n], htxt=h,
                       linestyle='dotted', linewidth=1,
                       flip=flip)
    
    plt.xlim(xl)


def draw_barcode(binfo, cmap="RdBu_r", dots="kp",
                 vmax=None, vmin=None,
                 figsize=(6.5, 1), ax=None, xlb=r"$\tau$ (ms)",
                 show_cbar=False, show_pline=False):
    
    pos_cbar = (0.04, 0.1, 0.02, 0.85)
    pos_ax = (0.08, 0.1, 0.72, 0.85)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax_cbar = plt.axes(position=pos_cbar)
        ax_main = plt.axes(position=pos_ax)
    else:
        fig = None
        plt.sca(ax)
        ax.axis("off")
        ax_cbar = ax.inset_axes(pos_cbar)
        ax_main = ax.inset_axes(pos_ax)
      
    plt.sca(ax_main)
    tbar = binfo["tbar"]
    
    if vmax is None:
        if vmin is None:
            vmax = np.percentile(binfo["barcode"][binfo["barcode"] > 0], 80)
            vmin = -vmax
        else:
            vmax = -vmin
            
    if vmin is None:
        vmin = -vmax
    
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
    
    xl = plt.xlim()
    xt, xtt = plt.xticks()
    assert 0 in xt
    id0 = np.where(xt == 0)[0][0]
    xtt[id0].set_text("NOW")
    plt.xticks(xt, labels=xtt)
    plt.xlim(xl)
    
    if show_cbar:
        plt.colorbar(location="left", ax=ax_main, cax=ax_cbar,
                     format=mticker.FormatStrFormatter("%d %%"))
    else:
        ax_cbar.axis("off")
        
    if show_pline:
        nb = np.concatenate(bpeaks).astype(int)
        if len(nb) > 0:
            ax_p = ax_main.inset_axes((0, 0, 1, 1.15))
            # print(nb)
            xb = tbar[nb]
            ax_p.vlines(xb, 0, 1.1, colors='k', linestyle=':', linewidth=1)
            ax_p.set_ylim((0, 1.1))
            ax_p.set_xlim(xl)
            ax_p.axis("off")
    
    plt.sca(ax_main)
    return fig, ax_main
