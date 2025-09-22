import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

import os
import sys
sys.path.append("../include/pytools")
import visu
import hhsignal
import xarray as xa

import oscdetector as od
import utils_te as ut
import figure_manager as fm
import tetools as tt

import utils_fig as uf
uf.set_plt()

empty_files = False
prob_spk_dir = "../transmission_line/simulation_data/postdata"
kappa_dir = "../transmission_line/simulation_data/postdata/kappa_stat"
te_dir = "../information_routing/data/te_2d_newmotif_newsurr"
te_colors = ("#d92a27", "#045894", "#a9a9a9")
# tl_colors = ("#e60000", "#0008ca")
tl_colors = te_colors[:2]
c_rect = "#676767"
box_height = 2


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

fm.track_global("cw_pairs", cw_pairs)

# fm.track_global("te_dir", te_dir)
# fm.track_global("kappa_dir", kappa_dir)


def load_kappa(kappa_dir, cid, wid):
    fname = os.path.join(kappa_dir, "kappa_%d%02d.nc"%(cid, wid))
    return xa.open_dataset(fname)


def load_te(te_dir, cid, wid):
    tmax = ut.get_max_period(cid)
    te_data_2d = uf.load_pickle(os.path.join(te_dir, "te_%d%02d.pkl"%(cid, wid)))
    te_data = ut.reduce_te_2d(te_data_2d, tcut=tmax)
    return te_data


def get_err_range(data, method="percentile", p_ranges=(5, 95), smul=1.96):
    # Assume that data is 2D (nsamples, T)
    if method == "percentile":
        m = np.median(data, axis=0)
        smin = np.percentile(data, p_ranges[0], axis=0)
        smax = np.percentile(data, p_ranges[1], axis=0)
    elif method == "std":
        m = data.mean(axis=0)
        s = data.std(axis=0) / np.sqrt(data.shape[0]) * smul
        smin = m - s
        smax = m + s
    else:
        raise ValueError("Unknown method: %s"%method)
    
    return m, smin, smax    


@fm.figure_renderer("single_resp_sample", reset=empty_files)
def show_single_spike_resp(figsize=(2.5, 2), prob_spk_dir=None, cid=0, wid=10, ntp=1, s=1.5):
    lb = "F" if ntp == 0 else "S"
    
    prob_set = xa.load_dataset(os.path.join(prob_spk_dir, "prob_spk_set_nc%d_sample.nc"%(cid-1)))
    prob_set = prob_set.sel(dict(nw=wid))
    
    t = prob_set.t
    # ndelay = prob_set.ndelay
    cmap = plt.get_cmap("turbo")
    
    fig = uf.get_figure(figsize)
    
    # for i, wid in enumerate((wid1, wid2)):
    #     plt.subplot(1,2,i+1)
    #     prob_set = prob_set.sel(dict(nw=wid))
        
    for nd in prob_set.ndelay:
        prob_sub = prob_set.sel(dict(ndelay=nd, ntp=ntp))
        p = prob_sub.prob.data
        p0 = prob_sub.prob0.data
        
        # print(p.shape, p0.shape)
        dy = p.mean(axis=0)-p0.mean(axis=0)
        dy = gaussian_filter1d(dy, s)
        # dy = gaussian_filter1d((p - p0).mean(axis=0), s)
        ds = p.std(axis=0)/np.sqrt(p.shape[0]) * 0.98
        
        plt.plot(t, dy, c=cmap(nd/prob_set.ndelay.max()))
        plt.fill_between(t, dy-ds, dy+ds, color=cmap(nd/prob_set.ndelay.max()), alpha=0.3, edgecolor="none")

    plt.xlabel("Time from" + "\n" + "transmitter firing (ms)")
    plt.ylabel(r"$P_%s - P_{%s, 0}$"%(lb, lb))
    plt.ylim([-0.001, 0.02])

    return fig


@fm.figure_renderer("show_tline_sample", reset=empty_files)
def show_tline_sample(figsize=(2.5, 8), kappa_dir=None, cid=0, wid=0, err_method="std", err_std=1.96, p_ranges=(5, 95)):
    lb_set = ("S", "F")
    
    kappa_set = load_kappa(kappa_dir, cid, wid)
    
    tmax = ut.get_max_period(cid)
    
    fig = uf.get_figure(figsize)

    axs = []
    for npop in range(2):
        ax = plt.axes([0.12, 0.7-0.35*npop, 0.8, 0.25])
        axs.append(ax)
        
        t = kappa_set.ndelay.data
        ym_b, ymin_b, ymax_b = get_err_range(kappa_set.kappa_base.isel(dict(ntp=1-npop)).data, method=err_method, smul=err_std, p_ranges=p_ranges)
        ym, ymin, ymax = get_err_range(kappa_set.kappa.isel(dict(ntp=1-npop)).data, method=err_method, smul=err_std, p_ranges=p_ranges)

        plt.plot(t, ym, color=te_colors[npop], lw=1, label=r"$\kappa_%s$"%(lb_set[npop]))
        plt.fill_between(t, ymin, ymax, color=te_colors[npop], alpha=0.3, edgecolor="none")
        plt.fill_between(t, ymin_b, ymax_b, color="k", alpha=0.5, edgecolor="none", label=r"$\kappa_{base}$")
        plt.xlim([0, tmax])
        plt.legend(loc="lower right", fontsize=4.5, edgecolor="none", facecolor="none", ncol=2)
        
        plt.ylim([-0.22, 0.22])
        plt.xticks(np.arange(0, 31, 10))
        plt.yticks(np.arange(-0.2, 0.3, 0.1), labels=[str("%d %%"%(int(100*x))) for x in np.arange(-0.2, 0.3, 0.1)])
        
        plt.xlabel("Delay, d (ms)")
        plt.ylabel(r"$\kappa_%s$"%(lb_set[npop]))
    
    ax = plt.axes([0.01, 0.05, 0.95, 0.2])
    
    id_sig_pos, id_sig_neg, tq = ut.identify_sig_tline(kappa_set, err_method=err_method, err_std=err_std, p_ranges=p_ranges, num_min=2)
    tsig_pos_set = ut.convert_sig_boundary(id_sig_pos, tq)
    tsig_neg_set = ut.convert_sig_boundary(id_sig_neg, tq)

    box_height = 2
    visu.draw_te_diagram_reduce(tsig_pos_set, tmax, y0=2*box_height, colors=[tl_colors[0]]*2, box_height=box_height, alpha=0.5)
    visu.draw_te_diagram_reduce(tsig_neg_set, tmax, y0=2*box_height, colors=[tl_colors[1]]*2, box_height=box_height, alpha=0.5)

    ax.axis("off")

    return fig


def _span_bbox(r0, r1, c0, c1, wr, wc, ws_row, ws_col):
    # r0..r1, c0..c1 범위를 덮는 큰 bbox (figure 좌표)
    x = (c0+1)*ws_col + c0*wc
    y = 1 - ((r1+1)*ws_row + r1*wr) - wr
    w = (c1 - c0 + 1)*wc + (c1 - c0)*ws_col
    h = (r1 - r0 + 1)*wr + (r1 - r0)*ws_row
    return [x, y, w, h]


@fm.figure_renderer("draw_entire_irp", reset=empty_files)
def draw_entire_tl(figsize=(7.5, 10.5), kappa_dir=None, te_dir=None, err_std=2.58, p_ranges=(5, 95)):

    num_row = 7
    num_col = 3
    
    ws_row = 0.002
    ws_col = 0.02

    # ws_row = 0.01
    # ws_col = 0.05
    
    wr = (1-(num_row+1)*ws_row)/num_row
    wc = (1-(num_col+1)*ws_col)/num_col

    fig = uf.get_figure(figsize)
    
    k = 0
    for nr in range(num_row):
        for nc in range(len(cw_pairs[nr])):
            if len(cw_pairs[nr][nc]) == 0:
                continue
            
            cid, wid = cw_pairs[nr][nc]
            max_period = ut.get_max_period(cid)
            
            # compute significant points in TE
            te_data = load_te(te_dir, cid, wid)
            tlag = te_data["tlag"]
            id_sig_sets = ut.identify_sig_te1d(te_data, prt=p_ranges[1])
            tsig_sets = ut.convert_sig_boundary(id_sig_sets, tlag)
            
            # compute significant points in Tline
            kappa_set = load_kappa(kappa_dir, cid, wid)
            id_sig_pos, id_sig_neg, tq = ut.identify_sig_tline(kappa_set, err_method="std", err_std=err_std, p_ranges=p_ranges, num_min=4)
            tline_sig_pos = ut.convert_sig_boundary(id_sig_pos, tq)
            tline_sig_neg = ut.convert_sig_boundary(id_sig_neg, tq)
                
            # show axis
            pos = ((nc+1)*ws_col+nc*wc, 1-((nr+1)*ws_row+nr*wr)-wr, wc, wr)
            ax = plt.axes(pos)
                
            ax_te = ax.inset_axes([0.23, 0., 0.77, 1])
            fig.add_axes(ax_te)
            plt.sca(ax_te)
            
            box_opt = dict(xmax=max_period, y0=2*box_height, box_height=box_height)
            visu.draw_te_diagram_reduce(tsig_sets, colors=[c_rect]*2, 
                                        xmax=max_period,
                                        y0_set=[box_height/4, 7/4*box_height],
                                        box_height=box_height/2,
                                        show_axis=False, visu_type="arrow")
            
            # draw Tline results
            tline_opt = dict(visu_type="box", show_axis=False, alpha=0.5)
            visu.draw_te_diagram_reduce(tline_sig_pos, colors=[tl_colors[0]]*2, 
                                        y0_set=[-box_height/4, 9/4*box_height],
                                        xmax=max_period,
                                        box_height=box_height/2,
                                        **tline_opt)
            visu.draw_te_diagram_reduce(tline_sig_neg, colors=[tl_colors[1]]*2, 
                                        y0_set=[-box_height/4, 9/4*box_height],
                                        xmax=max_period,
                                        box_height=box_height/2,
                                        **tline_opt)
            # show axis
            visu.draw_te_diagram_reduce([[], []], colors=[c_rect]*2, **box_opt, visu_type="box", show_axis=True, fontsize=6)

            # draw indicator
            ax_pict = ax.inset_axes([0, 0.1, 0.2, 0.8])
            fig.add_axes(ax_pict)
            plt.sca(ax_pict)
            uf.draw_motif_pictogram(od.get_motif_labels()[wid], rcolor=uf.get_cid_color(cid), )

            ax.axis("off")
            ax.text(-1, 1, "%d"%(cw_id[k]), fontsize=6, ha='center', va='center')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1.2, 1.2])
            
            k+= 1
            
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

    

def main():
    show_single_spike_resp(prob_spk_dir=prob_spk_dir, cid=4, wid=10, ntp=0, _func_label="show_single_spike_resp_410")
    show_single_spike_resp(prob_spk_dir=prob_spk_dir, cid=4, wid=15, ntp=0, _func_label="show_single_spike_resp_415")
    show_tline_sample(kappa_dir=kappa_dir, cid=4, wid=10, err_method="std", err_std=1.96, _func_label="show_tline_example_410")
    show_tline_sample(kappa_dir=kappa_dir, cid=4, wid=15, err_method="std", err_std=1.96, _func_label="show_tline_example_415")
    
    # for nr in range(len(cw_pairs)):
    #     for nc in range(len(cw_pairs[nr])):
    #         if len(cw_pairs[nr][nc]) == 0:
    #             continue
    #         cid, wid = cw_pairs[nr][nc]
    #         show_tline_sample(kappa_dir=kappa_dir, cid=cid, wid=wid, err_method="std", err_std=1.96, _func_label="check_tline_%d%02d"%(cid, wid))

    draw_entire_tl(kappa_dir=kappa_dir, te_dir=te_dir, err_std=1.96, p_ranges=(2.5, 97.5))

if __name__ == "__main__":
    main()