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


def draw_with_err(t, xset, p_range=(5, 95), tl=None, linestyle="-", linewidth=1.2, c='k',
                  avg_method="median", label=None, alpha=0.2, err_linestyle="--"):
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

    plt.plot(_t, x50, c=c, linestyle=linestyle, label=label, linewidth=linewidth)
    plt.fill_between(_t, xtop, xbot, color=c, alpha=alpha, edgecolor="none",
                    linewidth=0.5, linestyle=err_linestyle)
    
    
def show_te2d_summary(te_data_2d, figsize=(6, 3), cmap="turbo", dpi=120, vmax=None, vmin=None):
    # For TE_{X-> Y}
    te = te_data_2d["te"].mean(axis=0) # (2, X, Y)
    lb_labels = ("F", "S")
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    for ntp in range(2):
        plt.sca(axs[ntp])
        plt.imshow(te[ntp], origin="lower", cmap=cmap, 
                        extent=(te_data_2d["tlag"][0], te_data_2d["tlag"][-1],
                                te_data_2d["tlag"][0], te_data_2d["tlag"][-1]),
                        vmin=vmin, vmax=vmax)
        plt.xlabel(r"$\tau_{dst}$ (ms)", fontsize=12)
        plt.ylabel(r"$\tau_{src}$ (ms)", fontsize=12)
        plt.title(r"$TE_{%s \rightarrow %s}$"%(lb_labels[ntp], lb_labels[1-ntp]), fontsize=14)
        plt.colorbar(shrink=0.5)
    plt.tight_layout()
        
    return fig    
    
    
def show_te_summary(te_data, figsize=(3.5, 3), dpi=120, ax=None, xl=None, yl=None,
                    title=None, key="te", xlb=r"$\tau$ (ms)", ylb=r"$TE$ (bits)",
                    avg_method="median",
                    stat_test=False,
                    subtract_surr=False):
    
    if key=="te":
        te_labels = (
            r"$TE_{F \rightarrow S}$",
            r"$TE_{S \rightarrow F}$",
            r"$TE^{surr}_{F \rightarrow S}$",
            r"$TE^{surr}_{S \rightarrow F}$"
        )
    else:
        te_labels = ("F", "S")
    
    assert avg_method in ("mean", "median")
    _draw_err = partial(draw_with_err, avg_method=avg_method, tl=xl)
    
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig = None
        plt.sca(ax)
        
    xy = te_data[key]
    labels = te_labels
    
    if key == "te":
        key_surr = "%s_surr"%(key)
        if subtract_surr:
            if avg_method == "median":
                ms = np.median(te_data[key_surr], axis=0, keepdims=True)
            else:
                ms = np.average(te_data[key_surr], axis=0, keepdims=True)
            
            xy = xy - ms
            xy_s = te_data[key_surr] - ms
            labels = (te_labels[0]+"-"+te_labels[2], 
                      te_labels[1]+"-"+te_labels[3],
                      None, None)
        else:
            xy_s = te_data[key_surr]
    else:
        stat_test=False

    
    t = te_data["tlag"]
    _draw_err(t, xy[:,0,:], c=cs[0], label=labels[0])
    _draw_err(t, xy[:,1,:], c=cs[1], label=labels[1])
    if key != "mi":
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
        plt.figure(dpi=dpi)
    
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


def gen_background_full(num_c=9, dpi=200):
    import oscdetector as od
    
    we = 0.01
    wc = 0.025
    wl = 0.01
    num_w = 16
    
    figsize = (20, num_c+1)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # boundaries
    w = 1 - we*2 - wc
    h = 1 - (we + wc)
    dw = w / num_w
    
    for n in range(num_w):
        plt.axes(position=(we+wc+dw*n, h, dw, wc))
        lb = od.get_motif_labels()[n]
        plt.text(0.5, 0.5, lb, va="center", ha="center", fontsize=12)
        plt.xticks([])
        plt.yticks([])
        
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

    dw = wtot/num_w
    dh = htot/num_c
    
    coords = []
    for nc in range(num_c):
        coords.append([])
        h0 = 1 - we - wc - dh*(nc+1)
        for i in range(num_w):
            w0 = we + wc + dw*i

            plt.axes(position=(w0, h0, dw, dh))
            plt.xticks([]); plt.yticks([])
            
            coords[-1].append((w0, h0, dw, dh))
            
    return fig, coords


orders = [0, 2, 10, 6, 14, 4, 5, 7, 15, 8, 13]


def gen_background(num_c=8, dpi=200):
    
    import oscdetector as od
    
    we = 0.01
    wc = 0.025
    wl = 0.01
    # num_c = 8

    figsize = (16, num_c+1)
    fig = plt.figure(figsize=figsize, dpi=dpi)

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
    plt.ylim(yl)
        

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


def draw_freq_indicator(cid=None, yl=None, flip=False, h=None,
                        lw=1, alpha=1, f0_set=None):
    
    xl = plt.xlim()
    yl = plt.ylim() if yl is None else yl
    
    lopt = dict(linestyle="--", linewidth=lw, alpha=alpha)
    tp_labels = (r"$T_s/2$", r"$T_s$", r"$T_f/2$", r"$T_f$")
    
    if f0_set is None:
        if cid is None: raise ValueError("cid or f0_set need to be determined")
        f0_set = fpeaks[cid-1]
    
    for tp in range(2):
        f0 = f0_set[tp]
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


# def draw_barcode(binfo, cmap="RdBu_r", dots="kp",
#                  vmax=None, vmin=None,
#                  figsize=(6.5, 1), ax=None, xlb=r"$\tau$ (ms)",
#                  show_cbar=False, show_pline=False):
    
#     pos_cbar = (0.04, 0.1, 0.02, 0.85)
#     pos_ax = (0.08, 0.1, 0.72, 0.85)
    
#     if ax is None:
#         fig = plt.figure(figsize=figsize)
#         ax_cbar = plt.axes(position=pos_cbar)
#         ax_main = plt.axes(position=pos_ax)
#     else:
#         fig = None
#         plt.sca(ax)
#         ax.axis("off")
#         ax_cbar = ax.inset_axes(pos_cbar)
#         ax_main = ax.inset_axes(pos_ax)
        
#     # print(ax_main)
#     plt.axes(ax_main)
#     # plt.sca(ax_main)
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
#     # plt.plot([tbar[-1], tbar[0]], [0.5, 0.5], 'k-', lw=0.2, alpha=0.5)
#     plt.plot([tbar[-1], 0], [0.5, 0.5], 'k-', lw=0.2, alpha=0.5)
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
        
#     if show_pline:
#         nb = np.concatenate(bpeaks).astype(int)
#         if len(nb) > 0:
#             ax_p = ax_main.inset_axes((0, 0, 1, 1.15))
#             # print(nb)
#             xb = tbar[nb]
#             ax_p.vlines(xb, 0, 1.1, colors='k', linestyle=':', linewidth=1)
#             ax_p.set_ylim((0, 1.1))
#             ax_p.set_xlim(xl)
#             ax_p.axis("off")
    
#     plt.sca(ax_main)
#     return fig, ax_main


def draw_barcode(binfo, cmap="RdBu_r", dots="kp",
                 vmax=None, vmin=None,
                 figsize=(5.5, 1), ax=None, ax_cbar=None, xlb=r"$\tau$ (ms)",
                 show_cbar=False, show_pline=False):
    
    # pos_cbar = (0.04, 0.1, 0.02, 0.85)
    # pos_ax = (0.08, 0.1, 0.72, 0.85)
    pos_cbar = (0.74, 0.1, 0.02, 0.85)
    pos_ax = (0.04, 0.1, 0.68, 0.85)
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax_cbar = plt.axes(position=pos_cbar)
        ax_main = plt.axes(position=pos_ax)
    else:
        fig = None
        if ax_cbar is None:
            plt.sca(ax)
            ax.axis("off")
            ax_cbar = ax.inset_axes(pos_cbar)
            ax_main = ax.inset_axes(pos_ax)
        else:
            ax_main, ax_cbar = ax, ax_cbar
        
    # print(ax_main)
    plt.axes(ax_main)
    # plt.sca(ax_main)
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
    extent = (-dt/2, tbar[-1]+dt/2, 0, 1)
    plt.imshow(binfo["barcode"], aspect="auto", cmap=cmap,
               extent=extent, vmax=vmax, vmin=-vmax)

    bpeaks = binfo["bpeaks"]
    for ntp in range(2):
        if len(bpeaks[ntp]) == 0: continue
        plt.plot(tbar[bpeaks[ntp]], [0.75-0.5*ntp]*len(bpeaks[ntp]), dots)
        
    # plt.gca().yaxis.tick_right()
    plt.yticks([0.25, 0.75], labels=(r"$S \rightarrow F$", r"$F \rightarrow S$"))
    # plt.plot([tbar[-1], tbar[0]], [0.5, 0.5], 'k-', lw=0.2, alpha=0.5)
    plt.plot([tbar[-1], 0], [0.5, 0.5], 'k-', lw=0.2, alpha=0.5)
    plt.xlabel(xlb, fontsize=14)
    
    xl = plt.xlim()
    xt, xtt = plt.xticks()
    assert 0 in xt
    id0 = np.where(xt == 0)[0][0]
    xtt[id0].set_text("NOW")
    plt.xticks(xt, labels=xtt)
    plt.xlim(xl)
    
    if show_cbar:
        plt.colorbar(location="right", ax=ax_main, cax=ax_cbar,
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


# --- summarizing diagrams

import matplotlib.patches as patches
import matplotlib.colors as mcolors
from typing import List, Optional

def draw_arrow_axis(xmax=30, h=1, width=0.02, head_length=1, head_width=0.05, fontsize=4):
    ax = plt.gca()
    for n in range(2):
        ar = patches.FancyArrow(
                    x=xmax*1.1, y=h*n,      # Start point
                    dx=-xmax*1.15, dy=0,      # Horizontal direction (no vertical movement)
                    width=width,        # Shaft width
                    head_width=head_width,   # Height of the head
                    head_length=head_length,  # Length of the arrowhead
                    color='k',
                    length_includes_head=False,
                )
        ax.add_patch(ar)
    
    w = (head_width + width)*0.8
    plt.vlines(0, -w, w, color='k', lw=0.5)
    plt.vlines(0, h-w, h+w, color='k', lw=0.5)
    plt.vlines(xmax, -w, w, color='k', lw=0.5)
    plt.vlines(xmax, h-w, h+w, color='k', lw=0.5)
    plt.text(0, h+w, r"$\tau=0$", fontsize=fontsize, color='k', ha="center", va="bottom")
    plt.text(xmax, h+w, r"$\tau=T_{s}$", fontsize=fontsize, color='k', ha="center", va="bottom")

        
# def draw_round_rect(tsets: List, xmax=30, y=0, h=0.1, color="k", rect_type="round", lw=0.5, alpha=1):
#     """
#     tsets: List of tuples, each tuple contains (start_time, end_time)
#     xmax: Maximum x-axis value for the rectangle
#     y: y-coordinate for the rectangle
#     h: Height of the rectangle
#     color: Color of the rectangle
#     alpha: Transparency of the rectangle
#     """
    
#     ax = plt.gca()
    
#     if rect_type == "round":
#         rounding_size = xmax * 0.01
#         opt = dict(boxstyle=f"round,pad=0.01,rounding_size={rounding_size}")
#     elif rect_type == "sharp":
#         opt = dict(boxstyle="Square")
#     else:
#         raise ValueError(f"{rect_type} is invalid")
    
#     for ts in tsets:
#         w = ts[1] - ts[0]
        
#         rrect = patches.FancyBboxPatch(
#             (ts[0], y-h/2), w, h,
#             **opt,
#             edgecolor='k',
#             linewidth=lw,
#             facecolor=color,
#             alpha=alpha
#         )
#         ax.add_patch(rrect)

def draw_round_rect(tsets: List, xmax=30, y=0, h=0.1, color="k",
                    rect_type="round", lw=0.5, alpha=1,
                    show_arrow=False, arrow_h=0):
    """
    tsets: List of tuples, each tuple contains (start_time, end_time)
    xmax: Maximum x-axis value for the rectangle
    y: y-coordinate for the rectangle
    h: Height of the rectangle
    color: Color of the rectangle
    alpha: Transparency of the rectangle
    """
    
    ax = plt.gca()
    
    if rect_type == "round":
        rounding_size = xmax * 0.01
        opt = dict(boxstyle=f"round,pad=0.01,rounding_size={rounding_size}")
    elif rect_type == "sharp":
        opt = dict(boxstyle="Square")
    elif rect_type == "none":
        opt = dict(boxstyle="Square")
    else:
        raise ValueError(f"{rect_type} is invalid")
    
    for ts in tsets:
        w = min(ts[1]-ts[0], xmax-ts[0])
        
        # rrect = patches.FancyBboxPatch(
        #     (ts[0], y-h/2), w, h,
        #     **opt,
        #     edgecolor='k',
        #     linewidth=lw,
        #     facecolor=color,
        #     alpha=alpha
        # )
        # ax.add_patch(rrect)
        
        points = [
            (ts[0], y+h/2),   # Top left
            (ts[0], y-h/2),  # Bottom left
            (ts[0]+w, y-h/2),  # Bottom right
            (ts[0]+w, y+h/2)  # Top right
        ]

        if show_arrow or (rect_type == "none"):
            if arrow_h < 0:
                points[0], points[1] = points[1], points[0]
                points[2], points[3] = points[3], points[2]
            # points.append((ts[0]+w/2, points[3][1]+arrow_h))
                
            if rect_type == "none":
                points = points
                points = [
                    points[0],
                    points[-1],
                    (ts[0]+w/2, points[3][1]+arrow_h)
                ]
                # print(points)
            else:
                points.append((ts[0]+w/2, points[3][1]+arrow_h))
        
        polygon = patches.Polygon(points, closed=True, fill=True, color=color, lw=lw, alpha=alpha)
        ax.add_patch(polygon)
                                  

    
def draw_arrow_line(tsets: List, y=0, dy=30, width=0.02, head_length=1, head_width=0.05, color="k"):
    ax = plt.gca()
    for ts in tsets:
        x0 = (ts[0]+ts[1])/2
        ar = patches.FancyArrow(
            x=x0, y=y,
            dx=-x0, dy=dy,
            width=width,
            head_width=head_width,
            head_length=head_length,
            edgecolor="k",
            facecolor=color,
            alpha=1,
            linestyle='-', lw=0.2,
            # linestyle="none",
            length_includes_head=True)
        ax.add_patch(ar)


def draw_te_diagram_full(tsig_sets, xmax=30, y0=30, colors_arrow=None, colors_rect=None, fontsize=4):
    if colors_arrow is None:
        colors_arrow = ("r", "b")
    if colors_rect is None:
        colors_rect = ("r", "b")
    
    # axes
    draw_arrow_axis(xmax=xmax, h=y0, head_length=2, width=0.5, head_width=2, fontsize=fontsize)
    # bbox
    draw_arrow_line(tsig_sets[0], y=y0, dy=-y0, width=0.5, head_length=4, head_width=1.5, color=colors_arrow[0])
    draw_arrow_line(tsig_sets[1], y=0, dy=y0, width=0.5, head_length=4, head_width=1.5, color=colors_arrow[1])
    draw_round_rect(tsig_sets[0], xmax=xmax, y=y0, h=4, color=colors_rect[0])
    draw_round_rect(tsig_sets[1], xmax=xmax, y=0, h=4, color=colors_rect[1])

    xl = [-5, xmax+1]
    plt.ylim([-3, 33])
    plt.xlim(xl)
    plt.axis("off")
    
    
def draw_reduce_axis(y=30, xmax=30, head_length=2, width=0.5, fs=5, lw=1, lb_text=("F", "S")):
    ax = plt.gca()
    for n in range(2):
        h0 = y*n
        points = (
            (-head_length, h0),
            (-lw/4, h0+width/2-lw/4),
            (xmax, h0+width/2-lw/4),
            (xmax, h0-width/2+lw/4),
            (-lw/4, h0-width/2+lw/4)
        )
        polygon = patches.Polygon(points, closed=True, fill=False, color='k', lw=lw)
        ax.add_patch(polygon)
        
    plt.plot([0, 0], [-width/2+lw/4, width/2-lw/4], 'k', lw=lw)
    plt.plot([0, 0], [y-width/2+lw/4, y+width/2-lw/4], 'k', lw=lw)
    
    plt.text(xmax-width, h0, lb_text[0], va="center", ha="right", fontsize=fs)
    plt.text(xmax-width, 0, lb_text[1], va="center", ha="right", fontsize=fs)
        
    plt.axis("off")

    
def draw_te_diagram_reduce(tsig_sets, xmax=30, y0=5, y0_set=None, colors=None, box_height=2., 
                           alpha=1, visu_type="box", fontsize=4, show_axis=True, arrow_ratio=0.5,
                           lb_text=("F", "S")):
    
    assert visu_type in ("box", "arrow", "arrow_only"), "visu_type must be 'box' or 'arrow'"
    
    if colors is None:
        colors = ("r", "b")
        
    if y0_set is not None:
        ymin = y0_set[0]
        ymax = y0_set[1]
    else:
        ymin = 0
        ymax = y0

    if visu_type == "box":
        draw_round_rect(tsig_sets[0], xmax=xmax, y=ymax, h=box_height, color=colors[0], rect_type="sharp", lw=0, alpha=alpha)
        draw_round_rect(tsig_sets[1], xmax=xmax, y=ymin, h=box_height, color=colors[1], rect_type="sharp", lw=0, alpha=alpha)
    elif visu_type == "arrow":
        draw_round_rect(tsig_sets[0], xmax=xmax, y=ymax, h=box_height, color=colors[0], rect_type="sharp", lw=0, alpha=alpha, show_arrow=True, arrow_h=-box_height*arrow_ratio)
        draw_round_rect(tsig_sets[1], xmax=xmax, y=ymin, h=box_height, color=colors[1], rect_type="sharp", lw=0, alpha=alpha, show_arrow=True, arrow_h=box_height*arrow_ratio)
    elif visu_type == "arrow_only":
        draw_round_rect(tsig_sets[0], xmax=xmax, y=ymax, h=box_height, color=colors[0], rect_type="none", lw=0, alpha=alpha, show_arrow=True, arrow_h=-box_height*arrow_ratio)
        draw_round_rect(tsig_sets[1], xmax=xmax, y=ymin, h=box_height, color=colors[1], rect_type="none", lw=0, alpha=alpha, show_arrow=True, arrow_h=box_height*arrow_ratio)
    
    xl = [-5, xmax+1]
    yl = [-3, y0+3]
    dx = xl[1] - xl[0]
    
    if show_axis:
        draw_reduce_axis(width=2.5, y=y0, xmax=xmax, head_length=dx*0.05, fs=fontsize, lb_text=lb_text)
    
    plt.ylim(yl)
    plt.xlim(xl)
    
    
