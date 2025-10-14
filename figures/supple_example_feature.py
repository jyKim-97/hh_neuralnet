import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../include/pytools")
import hhtools
import hhsignal

import figure_manager as fm
import os
import sys
sys.path.append("../include/pytools")
import utils_fig as uf
import oscdetector as od
uf.set_plt()


data_dir = "../gen_three_pop_samples_repr/data"
cid, nt = 5, 10


@fm.figure_renderer("show_lfp")
def show_lfp(figsize=(6, 4), cid=7, nt=10, data_dir="", tl=(5, 6)):
    def set_ax(rm_xticklabe=False, ylb="V"):
        plt.xlim(tl)
        plt.xticks(np.arange(tl[0], tl[-1]+0.1, 0.2))
        if rm_xticklabe:
            plt.gca().set_xticklabels([])
        plt.ylabel("%s (mV)"%(ylb))
        plt.ylim([-67, -53])
    
    sobj = hhtools.SummaryLoader(data_dir)
    detail = sobj.load_detail(cid-1, nt)
    
    t = detail["ts"]
    idt = t >= 1
    t = t[idt]
    vm = [vlfp[idt] for vlfp in detail["vlfp"][1:]]
    
    fig = uf.get_figure(figsize)
    plt.axes((0.1, 0.52, 0.8, 0.4))
    plt.plot(t, vm[0], c='k', lw=0.8)
    set_ax(rm_xticklabe=True, ylb=r"$V^F$")
    
    plt.axes((0.1, 0.1, 0.8, 0.38))
    plt.plot(t, vm[1], c='k', lw=0.8)
    set_ax(rm_xticklabe=False, ylb=r"$V^S$")
    plt.xlabel("Time (s)")
    
    return fig
    

@fm.figure_renderer("show_corr")
def show_corr(figsize=(3, 8), cid=7, nt=10, data_dir="", tl=(5, 6)):
    from scipy.signal import find_peaks
    
    sobj = hhtools.SummaryLoader(data_dir)
    detail = sobj.load_detail(cid-1, nt)
    cs = ("r", "b")
    
    t = detail["ts"]
    idt = (t >= tl[0]) & (t < tl[1])
    # t_sub = t[idt]
    vm_sub = [vlfp[idt] for vlfp in detail["vlfp"][1:]]
    
    fig = uf.get_figure(figsize)
    axs = uf.get_custom_subplots(h_ratio=(1, 1, 1), w_ratio=[1],
                           h_blank_boundary=0.1, w_blank_boundary=0.05)
    
    yl = [-1.2, 1.2]
    xl = (-40, 40)
    
    for n, (nax1, nax2) in enumerate(((0, 0), (0, 1), (1, 1))):
        corr, tlag = hhsignal.get_correlation(vm_sub[nax1], vm_sub[nax2], srate=2000, max_lag=0.1)
        tlag *= 1e3
        ax = axs[n][0]
        ax.plot(tlag, corr, 'k', lw=.8)
        ax.set_xlim([-50, 50])        
        
        # organize peaks
        if n == 0 or n == 2:
            for i in range(2):
                idp_set = hhsignal.detect_peak(corr, mode=i+1)
                idp1 = idp_set[1]
                ax.plot([xl[0], tlag[idp1], tlag[idp1]], 
                         [corr[idp1], corr[idp1], yl[0]], color=cs[i], linestyle='--', linewidth=0.5)
                
        else:
            idp_set = hhsignal.detect_peak(corr, mode=1)
            idp0 = idp_set[0]
            ax.plot([xl[0], tlag[idp0], tlag[idp0]], 
                         [corr[idp0], corr[idp0], yl[0]], color='k', linestyle='--', linewidth=0.5)
        
        ax.set_xticks(np.arange(-40, 41, 20))
        ax.set_xlim([-40, 40])
        if n == 2:
            ax.set_xlabel(r"$\tau$ (ms)")
        
        if n == 0:
            ax.set_ylabel(r"$AC_{V^F, V^F}(\tau)$", labelpad=5)
        elif n == 1:
            ax.set_ylabel(r"$CC_{V^F, V^S}(\tau)$", labelpad=5)
        elif n == 2:
            ax.set_ylabel(r"$AC_{V^S, V^S}(\tau)$", labelpad=5)
        
        ax.set_ylim([-1.1, 1.1])
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["-1.0", "0.0", "1.0"])
        
        # if n != 1:
        #     ax.set_ylim([-1.1, 1.1])
        #     ax.set_yticks([-1, 0, 1])
        #     ax.set_yticklabels(["-1.0", "0.0", "1.0"])
        # else:
            # ax.set_ylim([-.8, .8])
            # ax.set_yticks(np.arange(-0.1, 0.5+1e-3, 0.5))
    
    return fig


if __name__ == "__main__":
    show_lfp(data_dir=data_dir, cid=cid, nt=nt, tl=(5, 5.6))
    show_corr(data_dir=data_dir, cid=cid, nt=nt, tl=(3, 5))