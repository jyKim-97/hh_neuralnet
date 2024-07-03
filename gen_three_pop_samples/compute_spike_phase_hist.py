# from tqdm.notebook import tqdm
from tqdm import trange
import numpy as np
import pickle as pkl
from scipy import signal

import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools


fpeaks = (([39.520], [34.517], [31.016], [60.030], [34.517, 61.531], [61.031], [66.533], [68.534]), # Fast subpop
          ([25.513], [25.513], [27.014], [35.018], [34.017, 65.033], [29.515], [66.533, 22.011], [34.017, 68.034])) # Slow subpop


teq = 500
srate = 2000
fo = 5
fw_half = 5


def main(fdir="./data", fout="./spike_phs_hist.pkl"):
    sobj = hhtools.SummaryLoader(fdir)
    edges = np.linspace(-np.pi, np.pi, 51)
    
    fp_cid_set = []
    phs_bin_set = []
    
    for cid in range(1, 9):
        fp_cid = scan_fpeaks(cid)
        phs_set, amp_set = compute_pa_at_spike_all_trial(sobj, cid, fp_cid, sobj.num_controls[1])
        phs_bin = compute_histogram(phs_set, amp_set, edges, p_upper=90, p_lower=10)

        fp_cid_set.append(fp_cid)
        phs_bin_set.append(phs_bin)
        
        del phs_set, amp_set
        
    with open(fout, "wb") as fp:
        pkl.dump(dict(
            phs_bin_set=phs_bin_set, fp_cid_set=fp_cid_set, edges=edges,
            attrs=dict(fo=fo, teq=teq, fpeaks=fpeaks)
        ), fp)
        
        
def compute_histogram(phs_set, amp_set, edges, p_upper=90, p_lower=10):
    
    # compute threshold
    ath = [[], []]  
    for n in range(len(phs_set)):
        
        ath[0].append([
            np.percentile(np.concatenate(amp_set[n][:2]), p_upper),
            np.percentile(np.concatenate(amp_set[n][2:]), p_upper)
        ])
        
        ath[1].append([
            np.percentile(np.concatenate(amp_set[n][:2]), p_lower),
            np.percentile(np.concatenate(amp_set[n][2:]), p_lower)
        ])
    
    # discretize
    f = [
        lambda phs, amp, ath: np.array(phs)[np.array(amp) >= ath],
        lambda phs, amp, ath: np.array(phs)[np.array(amp) < ath]
    ]
    
    phs_bins = np.zeros([2, len(phs_set), 4, len(edges)-1]) # up/down, fp_info, id_types, edges
    for ma in range(2):
        for n in range(len(phs_set)):
            for ntp in range(4):
                phs = f[ma](phs_set[n][ntp], amp_set[n][ntp], ath[ma][n][ntp//2])
                phs_bins[ma, n, ntp], _ = np.histogram(phs, edges)

    return phs_bins
        
        
def compute_pa_at_spike_all_trial(sobj, cid, fp_info, nitr):
    phs_set = [[[] for _  in range(4)] for _ in fp_info]
    amp_set = [[[] for _  in range(4)] for _ in fp_info]
    
    
    itr_comb = [[popid, nfp] for popid in range(4) for nfp in range(len(fp_info))]
    
    
    for n in trange(nitr, desc="cid%d"%(cid)):
        detail = sobj.load_detail(cid-1, n)
        
        phs_set_single, amp_set_single = compute_pa_at_spike(detail, fp_info)
        
        for popid, nfp in itr_comb:
            phs_set[nfp][popid].extend(phs_set_single[nfp])
            amp_set[nfp][popid].extend(amp_set_single[nfp])
        
    return phs_set, amp_set


def compute_pa_at_spike(detail, fp_info):
    
    phs_set = [[[] for _  in range(4)] for _ in fp_info]
    amp_set = [[[] for _  in range(4)] for _ in fp_info]
    
    v_phs_set, v_amp_set = [], []
    for fp in fp_info:
        ntype, frange = fp["ntype"], fp["frange"]
        v_phs, v_amp = decompose_signal(detail["vlfp"][ntype+1], frange, fo, srate)
        v_phs_set.append(v_phs)
        v_amp_set.append(v_amp)
        
    # get spike
    for idcell, step_spk in enumerate(detail["step_spk"]):
        popid = get_popid(idcell)
        id_spk = convert_spike_index(step_spk, dt=1e-5, teq=0.5, srate=2000)
        
        for i in range(len(fp_info)):
            phs_set[i][popid].extend(v_phs_set[i][id_spk])
            amp_set[i][popid].extend(v_amp_set[i][id_spk])
    
    return phs_set, amp_set
    
    
def scan_fpeaks(cid):
    fp_info = []
    f0_prev = []
    for ntype in range(2):
        for f0 in fpeaks[ntype][cid-1]:
            
            flag_prev = False
            for f0_p in f0_prev:
                if abs(f0 - f0_p) < 1:
                    f0_r = f0_p
                    flag_prev = True
                    break
                
            if not flag_prev:
                f0_r = np.round(f0, 0)
                f0_prev.append(f0_r)
            
            frange = [f0_r-fw_half, f0_r+fw_half]
            fp_info.append({"ntype": ntype, "frange": frange})
    return fp_info

    
def filt_between(sig, f1, f2, fs=2000, fo=10):
    b1, a1 = signal.butter(fo, f1, 'hp', fs=fs, output='ba')
    filtered = signal.filtfilt(b1, a1, sig, padlen=50)
    
    b2, a2 = signal.butter(fo, f2, 'lp', fs=fs, output='ba')
    filtered = signal.filtfilt(b2, a2, filtered, padlen=50)
    
    return filtered

    
def decompose_signal(sig, frange, fo=5, srate=2000):
    yf = filt_between(sig, frange[0], frange[1], fs=srate, fo=fo)
    yh = signal.hilbert(yf)
    yamp = np.abs(yh).astype(np.float32)
    yphs = np.angle(yh).astype(np.float32)
    return yphs, yamp


def convert_spike_index(step_spk, dt=0.01*1e-3, teq=0.5, srate=2000):
    t_spk = step_spk * dt
    return np.array(t_spk[t_spk > teq] * srate).astype(int)
    
    
def get_popid(n):
    if n > 1800:
        return 3
    elif n > 1000:
        return 2
    elif n > 800:
        return 1
    else:
        return 0
    
    
if __name__ == "__main__":
    main()