import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.append("../include/pytools")
import visu
import hhsignal

import oscdetector as od
import utils_te as ut
import figure_manager as fm
import tetools as tt

import hhtools

import utils_fig as uf
uf.set_plt()

empty_files = False
te_colors = ("#d92a27", "#045894", "#a9a9a9")
c_circ = "#7d0000"
c_rect = "#676767"
te_dir = "../information_routing/data/te_2d_newmotif_newsurr"
data_dir = "../gen_three_pop_samples_repr/data/"

fm.track_global("te_colors", te_colors)
fm.track_global("te_dir", te_dir)
fm.track_global("data_dir", data_dir)
fm.track_global("c_rect", c_rect)

cw_pairs = [
            [(2, 2), (1, 2), (3, 2)],
            [(2, 10), (1, 5), (3, 15)],
            [(6, 10), [], (4, 2)],
            [(5, 4), [], (4, 5)],
            [(5, 10), [], (4, 10)],
            [(5, 14), [], (4, 15)],
            [(7, 2), (7, 5), (7, 15)]
        ]
cw_id = (1, 3, 5, 2, 4, 6, 11, 7, 12, 8, 13, 9, 14, 10, 15, 16, 17)


def gen_signal(tmax, t_pts, f):
    srate = 2000
    t = np.arange(0, tmax, 1/srate)
    y = np.zeros_like(t)
    
    s = 0.2
    for t0 in t_pts:
        assert t0 < tmax
        n0 = int(t0*srate)
        nw = int(10*s*srate)
        
        tsub = np.arange(-nw, nw+1)/srate
        ysub = np.cos(2*np.pi*f*tsub) * np.exp(-tsub**2/s) * 0.5
        
        y[n0-nw:n0+nw+1] += ysub
    
    return t, y


def show_schem_spec(psd, tpsd, fpsd, yl=None, pop_txt="Fast", cmap="jet"):
    vmin, vmax = None, None
    if "RdBu" in cmap:
        # vmin, vmax = -5, 5
        vmin, vmax = -0.2, 0.2
        # psd = (psd - psd.mean(axis=1, keepdims=True))/psd.std(axis=1, keepdims=True)
    
    plt.imshow(psd, extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]), aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    # plt.yticks([freq_slow, freq_fast], labels=["slow", "fast"])
    plt.xticks([])
    plt.yticks([])
    # plt.ylim([freq_slow-5, freq_fast+5])
    plt.ylim(yl)
    plt.ylabel("Frequency (Hz)", fontsize=5)
    
    # put pop text
    xl = plt.xlim()
    yl = plt.ylim()
    
    x0 = xl[0]+0.15*(xl[1]-xl[0])
    y0 = yl[1]-0.15*(yl[1]-yl[0])
    
    plt.text(x0, y0, pop_txt, color="w", ha="center", va="center", fontsize=6, fontweight="bold")
    

@fm.figure_renderer("schem_mfop", reset=empty_files)
def schem_mfop(figsize=(6, 3.3), seed=42):
    freq_fast = 40
    freq_slow = 30
    srate = 2000
    tmax = 20
    an = 0.5
    
    np.random.seed(seed)
    
    t, yf1 = gen_signal(tmax, [4.8, 8, 13.8,], freq_fast)
    t, yf2 = gen_signal(tmax, [8.2, 14, 17.2], freq_slow)
    yf = yf1 + yf2 + an*np.random.randn(len(t))

    t, ys1 = gen_signal(tmax, [8.1, 14], freq_fast)
    t, ys2 = gen_signal(tmax, [2, 2.8, 8, 14, 17], freq_slow)
    ys = ys1 + ys2 + an*np.random.randn(len(t))
    
    psd_s, fpsd, tpsd = hhsignal.get_stfft(ys, t, srate, mbin_t=0.1, wbin_t=1, frange=(1, 100))
    psd_f, fpsd, tpsd = hhsignal.get_stfft(yf, t, srate, mbin_t=0.1, wbin_t=1, frange=(1, 100))
    
    yl = (freq_slow-5, freq_fast+5)
    fig = uf.get_figure(figsize)
    plt.subplot(211)
    show_schem_spec(psd_f, tpsd, fpsd, yl=yl, pop_txt="", cmap="RdBu_r")
    plt.subplot(212)
    show_schem_spec(psd_s, tpsd, fpsd, yl=yl, pop_txt="", cmap="RdBu_r")
    
    return fig


def get_psd_set(detail):
    psd_set = []
    for i in range(2):
        psd, fpsd, tpsd = hhsignal.get_stfft(detail["vlfp"][i+1], detail["ts"], 2000,
                                             frange=(5, 110))
        psd_set.append(psd)
    return psd_set, fpsd, tpsd


def show_psd(psd, fpsd, tpsd, vmin=0, vmax=1):
    plt.imshow(psd, aspect="auto", cmap="jet", origin="lower",
               vmin=vmin, vmax=vmax,
               extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]),
               interpolation="bicubic")
    plt.ylabel("Frequency (Hz)")
    
    
def set_ticks(tl):
    plt.xlim(tl)
    plt.xticks(np.arange(tl[0], tl[1]+1e-3, 0.5))
    plt.yticks(np.arange(20, 101, 20))
    plt.ylim([18, 82])
    plt.gca().set_xticklabels([])
    
    
def set_colorbar(cticks=None):
    cbar = plt.colorbar()
    cbar.set_ticks(cticks)


@fm.figure_renderer("mfop_example", reset=empty_files)
def draw_example(figsize=(2, 4), cid=7, nt=93, tl=(2.25, 4.15)):
    summary_obj = hhtools.SummaryLoader(data_dir)
    detail = summary_obj.load_detail(cid-1, nt)
    psd_set, fpsd, tpsd = get_psd_set(detail)
    
    fig = uf.get_figure(figsize)
    plt.subplot(211)
    show_psd(psd_set[0]-psd_set[0].mean(axis=1, keepdims=True), fpsd, tpsd, vmin=-0.3, vmax=0.3)
    set_colorbar(cticks=((-0.3, 0., 0.3)))
    set_ticks(tl)

    
    plt.subplot(212)
    show_psd(psd_set[1]-psd_set[1].mean(axis=1, keepdims=True), fpsd, tpsd, vmin=-0.3, vmax=0.3)
    set_ticks(tl)
    set_colorbar(cticks=((-0.3, 0., 0.3)))
    # plt.ylim([15, 85])
    
    return fig


@fm.figure_renderer("te_example", reset=empty_files)
def draw_example_te(figsize=(2.8, 11.8), cid=5, wid=10, p_ranges=(5, 95), te_dir=None):
# def draw_example_te(figsize, cid=5, wid=10, p_ranges=(5, 95), te_dir=None):
    # read te_data    
    tcut = ut.get_max_period(cid)
    te_data_2d = uf.load_pickle(os.path.join(te_dir, "te_%d%02d.pkl"%(cid, wid)))
    te_data = ut.reduce_te_2d(te_data_2d, tcut=tcut)
    # te_data_2d_b = uf.load_pickle(os.path.join(te_dir, "te_%d%02d.pkl"%(cid, 0)))
    # te_data_b = tt.reduce_te_2d(te_data_2d_b, tcut=tcut)
    tlag = te_data["tlag"]
    
    # get full TE
    lb_pop = ("F", "S")
    ybar = 0.085
    fig = uf.get_figure(figsize)

    # ax 1-2) draw TE results
    id_sig_sets = ut.identify_sig_te1d(te_data, prt=p_ranges[1])
    tsig_sets = ut.convert_sig_boundary(id_sig_sets, tlag)

    for nd in range(2):
        plt.axes([0.12, 0.75-0.28*nd, 0.8, 0.2])
    
        x1 = te_data["te"][:,nd,:]
        x2 = te_data["te_surr"][:,nd,:]
        tlag = te_data["tlag"]
        
        opt = dict(alpha=0.5, avg_method='median', p_range=p_ranges)
        opt_line = dict(linestyle="-", linewidth=0.5)
        opt_noline = dict(linestyle="none")
        # visu.draw_with_err(tlag, x1, c=te_colors[nd], **opt, **opt_noline) # TE
        plt.plot(tlag, np.median(x1, axis=0), **opt_line)
        visu.draw_with_err(tlag, x2, c=te_colors[2], **opt, **opt_noline) # TE surrogate
        ax1, = plt.plot(tlag, np.median(x1, axis=0), c=te_colors[nd], **opt_line)
        ax2, = plt.plot(tlag, np.median(x2, axis=0), c=te_colors[2], **opt_line)
        
        for tsig in tsig_sets[nd]:
            plt.plot(tsig, [ybar]*2, c="k", lw=1)
    
        plt.xlim([1, tcut])
        plt.ylim([-0.001, 0.1])
        # plt.gca().set_yscale("log")
        
        plt.xlabel(r"$\tau$ (ms)")
        plt.ylabel(r"$TE_{%s \rightarrow %s}$ (bits)"%(lb_pop[nd], lb_pop[1-nd]))
        
        plt.legend([ax1, ax2], ("TE", r"TE$_{surr}$"), fontsize=5,
                   loc="upper right", edgecolor="none", facecolor="none",
                   borderpad=0, borderaxespad=0, handlelength=1, handletextpad=0.8, labelspacing=0.25)
        
    # ax3) Full IRP
    plt.axes([0.01, 0.22, 0.93, 0.15])
    visu.draw_te_diagram_full(tsig_sets, xmax=tcut, y0=30,
                              colors_arrow=te_colors,
                              colors_rect=[c_rect]*2)
    
    # ax4) reduced IRP
    box_height = 2
    plt.axes([0.01, 0.05, 0.93, 0.12])
    visu.draw_te_diagram_reduce(tsig_sets, xmax=tcut, y0=2*box_height, colors=[c_rect]*2,
                                box_height=box_height, visu_type="arrow")

    return fig


def _span_bbox(r0, r1, c0, c1, wr, wc, ws_row, ws_col):
    # r0..r1, c0..c1 범위를 덮는 큰 bbox (figure 좌표)
    x = (c0+1)*ws_col + c0*wc
    y = 1 - ((r1+1)*ws_row + r1*wr) - wr
    w = (c1 - c0 + 1)*wc + (c1 - c0)*ws_col
    h = (r1 - r0 + 1)*wr + (r1 - r0)*ws_row
    return [x, y, w, h]


@fm.figure_renderer("entire_irp", reset=empty_files)
def draw_entire_irp(figsize=(9.5, 11.5), p_ranges=(5, 95)):
    
    num_row = 7
    num_col = 3

    # draw IRP set
    ws_row = 0.002
    ws_col = 0.02
    # ws_row = 0.01
    # ws_col = 0.05

    wr = (1-(num_row+1)*ws_row)/num_row
    wc = (1-(num_col+1)*ws_col)/num_col
    box_height = 2
    
    fig = uf.get_figure(figsize=figsize)
    k = 0
    for nr in range(num_row):
        for nc in range(len(cw_pairs[nr])):
            if len(cw_pairs[nr][nc]) == 0:
                continue
            
            cid, wid = cw_pairs[nr][nc]
            max_period = ut.get_max_period(cid)

            # compute peaks
            te_data_2d = uf.load_pickle(os.path.join(te_dir, "te_%d%02d.pkl"%(cid, wid)))    
            te_data = ut.reduce_te_2d(te_data_2d, tcut=max_period)
            tlag = te_data["tlag"]

            id_sig_sets = ut.identify_sig_te1d(te_data, prt=p_ranges[1])
            tsig_sets = ut.convert_sig_boundary(id_sig_sets, tlag)
                
            pos = ((nc+1)*ws_col+nc*wc, 1-((nr+1)*ws_row+nr*wr)-wr, wc, wr)
            ax = plt.axes(pos)
                
            ax_te = ax.inset_axes([0.23, 0., 0.77, 1])
            fig.add_axes(ax_te)
            plt.sca(ax_te)
            
            box_height = 2
            visu.draw_te_diagram_reduce(tsig_sets, 
                                        xmax=max_period, y0=2*box_height, colors=[c_rect]*2, 
                                        box_height=box_height, visu_type="arrow",
                                        fontsize=6)

            # draw indicator
            ax_pict = ax.inset_axes([0.05, 0.1, 0.2, 0.8])
            fig.add_axes(ax_pict)
            plt.sca(ax_pict)
            uf.draw_motif_pictogram(od.get_motif_labels()[wid], rcolor=uf.get_cid_color(cid))

            ax.axis("off")
            ax.text(-1, 1, "%d"%(cw_id[k]), fontsize=6, ha='center', va='center')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1.2, 1.2])
            
            k += 1
            
            
    # draw pictograms
    empty_cells = [(r,c) for r in range(num_row) for c in range(num_col)
                   if c < len(cw_pairs[r]) and len(cw_pairs[r][c]) == 0]
    c_edges = "#565654"
    if empty_cells:
        rows = sorted({r for r,_ in empty_cells})
        cols = sorted({c for _,c in empty_cells})
        r0, r1 = min(rows), max(rows)         # 세로로 연속한 빈줄 범위
        c0, c1 = min(cols), max(cols)         # 가로로 연속한 빈칸 범위
        
        ws_row_e = ws_row*0.6
        ws_col_e = ws_col*0.6
        
        big_pos = _span_bbox(r0, r1, c0, c1, wr+ws_row-ws_row_e, wc+ws_col-ws_col_e, ws_row_e, ws_col_e)
        ax_big = fig.add_axes(big_pos)
        ax_big.axis('off')

        # ---- big 영역 안에 4×2 서브-axes 만들기 ----
        sub_R, sub_C = 4, 2
        # pad = 0.04  # 내부 여백 비율
        pad = 0.02
        h = (1 - (sub_R+1)*pad) / sub_R
        w = (1 - (sub_C+1)*pad) / sub_C
        
        cid_list = (2, 1, 6, 3, 5, 4, 7)

        k = 0
        for rr in range(sub_R):
            for cc in range(sub_C):
                subpos = [(cc+1)*pad + cc*w,
                          1 - ((rr+1)*pad + rr*h) - h,
                          w, h]
                subpos[0] += pad*1.2
                ax_sub = ax_big.inset_axes(subpos)
                fig.add_axes(ax_sub)
                if k < len(cid_list):
                    cid = cid_list[k]
                    uf.draw_landmark_diagram(cid=cid, ax=ax_sub, rot=90, text_pops=("F", "S"),
                                             color_e=c_edges, colors_i=(c_edges, c_edges), box_color=uf.get_cid_color(cid))
                    
                    xl = plt.xlim()
                    yl = plt.ylim()
                    plt.text(xl[0]+(xl[1]-xl[0])*0.1, (yl[0]+yl[1])/2, "Landmark %d"%(cid), fontsize=6, va="center", ha="center", rotation=90)
                    k+=1
                else:
                    ax_sub.axis("off")
    
    return fig


def gen_sample1(T=1000):
    tau, s = 9, 3
    # create X as AR(1) noise
    X = np.zeros(T)
    for t in range(1, T):
        X[t] = 0.6*X[t-1] + 0.4*np.random.randn()

    # create Y that depends on past Y (s) and past X (tau)
    Y = np.zeros(T)
    for t in range(T):
        y_term = 0.1*Y[t-s] if t-s >= 0 else 0.0
        # x_term = 0.6*X[t-tau] if t-tau >= 0 else 0.0
        x_term = 0.5*X[t-tau] if t-tau >= 0 else 0.0
        Y[t] = y_term + 0.2*np.random.randn()
        
    valid = np.arange(max(tau, s), T)  # ensure valid lags
    Y_t   = Y[valid]
    X_tau = X[valid - tau]
    Y_s   = Y[valid - s]
    
    return Y_t, Y_s, X_tau


def gen_sample2(T=1000):
    tau, s = 9, 3
    # create X as AR(1) noise
    X = np.zeros(T)
    for t in range(1, T):
        X[t] = 0.8*X[t-1] + 0.2*np.random.randn()

    # create Y that depends on past Y (s) and past X (tau)
    Y = np.zeros(T)
    for t in range(T):
        y_term = 0.1*Y[t-s] + 0.5*X[t-tau] if t-s >= 0 else 0.0
        x_term = 0.6*X[t-tau] if t-tau >= 0 else 0.0
        # x_term = 0.5*X[t-tau] if t-tau >= 0 else 0.0
        Y[t] = y_term + 2*x_term + 0.2*np.random.randn()
        
    valid = np.arange(max(tau, s), T)  # ensure valid lags
    Y_t   = Y[valid]
    X_tau = X[valid - tau]
    Y_s   = Y[valid - s]
    
    return Y_t, Y_s, X_tau


def gen_sample3(T=1000):
    tau, s = 9, 3
    # create X as AR(1) noise
    X = np.zeros(T)
    for t in range(1, T):
        X[t] = 0.8*X[t-1] + 0.2*np.random.randn()

    # create Y that depends on past Y (s) and past X (tau)
    Y = np.zeros(T)
    for t in range(T):
        y_term = 0.2*Y[t-s] if t-s >= 0 else 0.0
        x_term = 0.5*X[t-tau] if t-tau >= 0 else 0.0
        Y[t] = y_term + 0.2*x_term + 0.2*np.random.randn()
        
    valid = np.arange(max(tau, s), T)  # ensure valid lags
    Y_t   = Y[valid]
    X_tau = X[valid - tau]
    Y_s   = Y[valid - s]
    
    return Y_t, Y_s, X_tau


def draw_hist(ax, Y_t, Y_s, X_tau):
    bins = 21  # adjust bin granularity
    data = np.vstack([Y_t, X_tau, Y_s]).T
    H, edges = np.histogramdd(data, bins=bins)
    
    # Compute bin centers for plotting
    centers = [(e[:-1] + e[1:]) / 2 for e in edges]
    cy, cx, cz = np.meshgrid(centers[0], centers[1], centers[2], indexing='ij')

    # Flatten for plotting only nonzero bins
    mask = H > 0
    xs = cx[mask].ravel()
    ys = cy[mask].ravel()
    zs = cz[mask].ravel()
    cs = H[mask].ravel()
    
    # Normalize counts to [0,1] for marker size/alpha scaling
    cs_norm = (cs - cs.min()) / (cs.max() - cs.min() + 1e-12)
    
    # Each point = one occupied bin center; size & alpha ~ count
    sizes =  0.5 + 0.2*cs_norm
    alphas = 0.2 + 0.7 * cs_norm
    # s
    # for xi, yi, zi, si, ai in zip(xs, ys, zs, sizes, alphas):
    #     ax.scatter(yi, xi, zi, s=si, alpha=0.7, color='k', rasterized=True, edgecolors="none", zorder=0)
    pts = ax.scatter(
        ys, xs, zs, 
        s=sizes, c='k', alpha=1.0, edgecolors="none", zorder=0
    )
    # pts.set_alpha(None)           # allow per-point alpha
    pts.set_facecolors([[0,0,0,a] for a in alphas])  # apply alpha individually
    # pts.set_rasterized(True)      # rasterize only the scatter points
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Thicken axis lines and ticks
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        for line in axis.get_ticklines():
            line.set_linewidth(0.2)
        for line in axis.get_ticklines(minor=True):
            line.set_linewidth(0.2)

    # Also thicken main axis lines
    ax.xaxis.line.set_linewidth(0.2)
    ax.yaxis.line.set_linewidth(0.2)
    ax.zaxis.line.set_linewidth(0.2)


@fm.figure_renderer("irp_4dhist_reduce", reset=empty_files, exts=[".png", ".svg"])
def irp_4dhist_reduce(figsize=(2, 4)):

    np.random.seed(42)

    T = 300
    
    fig = uf.get_figure(figsize)
    axs = []
    for n in range(3):
        axs.append(fig.add_subplot(3, 1, n+1, projection='3d'))

    draw_hist(axs[0], *gen_sample1(T=T))
    draw_hist(axs[1], *gen_sample2(T=T))
    draw_hist(axs[2], *gen_sample3(T=T))

    return fig
        

def main():
    p_ranges = (2.5, 97.5)
    schem_mfop(figsize=(6, 3.3), seed=42)
    
    # for nr in range(len(cw_pairs)):
    #     for nc in range(len(cw_pairs[nr])):
    #         if len(cw_pairs[nr][nc]) == 0:
    #             continue
    #         cid, wid = cw_pairs[nr][nc]
    #         draw_example_te(te_dir=te_dir, cid=cid, wid=wid, p_ranges=p_ranges, _func_label="check_te_%d%02d"%(cid, wid))
            
    draw_example_te(te_dir=te_dir, cid=4, wid=10, p_ranges=p_ranges, _func_label="draw_example_te_410")
    draw_example_te(te_dir=te_dir, cid=4, wid=15, p_ranges=p_ranges, _func_label="draw_example_te_415")
    draw_entire_irp(figsize=(9.5, 11.5), p_ranges=p_ranges)
    draw_example()
    irp_4dhist_reduce()



if __name__ == "__main__":
    main()
