# from tqdm.notebook import tqdm
from tqdm import trange
import numpy as np
import pickle as pkl
from scipy import signal
from scipy.signal import savgol_filter

import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import hhsignal
from numba import jit


fpeaks = (([39.520], [34.517], [31.016], [60.030], [34.517, 61.531], [61.031], [66.533], [68.534]), # Fast subpop
          ([25.513], [25.513], [27.014], [35.018], [34.017, 65.033], [29.515], [66.533, 22.011], [34.017, 68.034])) # Slow subpop


tmax = 10.5
teq = 0.5
wbin_t = 0.2
srate = 2000

nsample = 100
frange = (2/wbin_t, 90)
wbin = int(wbin_t * srate)

num_r, num_c = 42, 41

cc_edges = np.linspace(0.1, 1, num_r)
amp_edges = np.linspace(0.1, 1.5, num_c)

sobj = hhtools.SummaryLoader("./data")


def main():
    np.random.seed(2000)
    
    joint_map = []
    for cid in range(1, 9):
        joint = get_corr(cid)
        joint_map.append(joint)
        
    with open("./joint_corr_amp.pkl", "wb") as fp:
        pkl.dump({
            "corr": joint_map,
            "cc_edges": cc_edges,
            "amp_edges": amp_edges,
            "fpeaks": fpeaks,
            },
                 fp)


def get_pop_rate(step_spk, fs=2000, tmax=10.5, dt=1e-5):
    l = int(tmax * fs)
    pop_rate = np.zeros([2, l])
    for n, _nt in enumerate(step_spk):
        nt = (np.array(_nt) * dt * fs).astype(int)
        pop_rate[n//1000][nt] += 1
    return pop_rate / len(step_spk) * fs, np.arange(l)/fs


@jit(nopython=True)
def is_out(nx, xrange):
    if (nx < xrange[0]) or (nx >= xrange[1]):
        return True
    else:
        return False


# @jit(nopython=True)
def get_bin_id(x, xedges):
    dx = xedges[1] - xedges[0]
    return np.int16((x - xedges[0]) / dx)


@jit(nopython=True)
def get_yf_avg(yf, f, fpeaks):
    yf_avg = np.zeros(len(fpeaks))
    for nf in range(len(fpeaks)):
        # frange = fpeaks[nf]
        for i in range(len(f)):
            f0 = f[i]
            # print(fpeaks[nf], f0)
            if (f0 >= fpeaks[nf]-5) and (f0 < fpeaks[nf]+5):
                yf_avg[nf] += yf[i]
                
    return yf_avg


def get_corr(cid):
    
    
    joint = np.zeros([2, 2, num_r-1, num_c-1])

    r_range = np.array([0, num_r-1])
    c_range = np.array([0, num_c-1])

    for ni in trange(200):
        detail = sobj.load_detail(cid-1, ni)

        pop_rate, t = get_pop_rate(detail["step_spk"])
        pop_rate = savgol_filter(pop_rate, 11, 1)
        
        for n in range(nsample):
            n0 = int((np.random.rand() * (tmax - teq - wbin_t) + teq) * srate)
            
            cc_p, t = hhsignal.get_correlation(pop_rate[0][n0:n0+wbin], pop_rate[1][n0:n0+wbin], srate=srate, max_lag=0.05)
            cc_p_max = np.max(cc_p)
            
            nr = get_bin_id(cc_p_max, cc_edges)
            if is_out(nr, r_range):
                continue
            
            yf1, f = hhsignal.get_fft(detail["vlfp"][1][n0:n0+wbin], srate, frange=frange)
            amp_peaks1 = get_yf_avg(yf1, f, np.array(fpeaks[0][cid-1]))
            na1 = get_bin_id(amp_peaks1, amp_edges)
            
            for i, nc in enumerate(na1):
                if not is_out(nc, c_range):
                    joint[0,i,nr,nc] += 1
            
            yf2, f = hhsignal.get_fft(detail["vlfp"][2][n0:n0+wbin], srate, frange=frange)
            amp_peaks2 = get_yf_avg(yf2, f, np.array(fpeaks[1][cid-1]))
            na2 = get_bin_id(amp_peaks2, amp_edges)
            
            for i, nc in enumerate(na2):
                if not is_out(nc, c_range):
                    joint[1,i,nr,nc] += 1
            

    joint /= np.sum(joint)
    
    return joint


if __name__ == "__main__":
    main()