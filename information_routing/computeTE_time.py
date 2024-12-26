"""
Thiu version uses Gaussian coupula method with frites
"""

import numpy as np
import argparse

import sys 
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import pickle as pkl
from functools import partial

import hhsignal
from scipy.signal import hilbert

sys.path.append("/home/jungyoung/Project/hh_neuralnet/extract_osc_motif")
import utils

from tqdm import tqdm

import tetools as tt
tag = ""


num_process = 4
srate = 2000
fdir_summary = "/home/jungyoung/Project/hh_neuralnet/gen_three_pop_samples_repr/data"+tag
famp_range = "/home/jungyoung/Project/hh_neuralnet/osc_detector/data/osc_motif/amp_range_set.pkl"

chunk_size = 100
nchunks = 400


def main(cid=0, wid=0, nadd=80, nphase=21):
    mua_sample, pid_sample = sample_data_wphase(cid, wid, summary_obj=summary_obj,
                                                nadd=nadd, nphase=nphase)


def compute_te_2d(v_sample: np.ndarray,
                  nmove: int=5):
    pass


# def sample_true_wphase(cid, wid,
#                        summary_obj=None,
#                        nsamples=2000,
#                        target="mua", srate=2000, nadd=80, 
#                        norm=True):
#     if summary_obj is None:
#         pass    
    
#     # sample target 
#     mua_sample
    
    
# def sample_data_wid(target_pid, mua_sample, pid_sample, nsample):

def sample_target_pid(target_pid, nsample, pid_table, mua_sample, nlag_max):
    nid = np.random.randint(low=0, high=len(pid_table[target_pid]), size=nsample)
    # mua_sub = np.zeros((2, ))


def build_pid_table(pid_sample, nphase=21):
    pid_table = [[[] for _ in range(nphase)] for _ in range(2)]
    nskip_max = 5
    
    for ntp in range(2):
        for n in range(pid_sample.shape[1]):
            pid_prv = -1
            nskip = 0
            for i in range(pid_sample.shape[2]):
                pid = pid_sample[n, i]
                if np.isnan(pid):
                    break
                
                if pid != pid_prv:
                    nskip = 0
                else:
                    nskip += 1
                
                if nskip == nskip_max:
                    nskip = 0
                
                if nskip == 0:
                    pid_table[ntp, pid].append((n, i))
            
    return pid_table
    

def sample_data_wphase(cid, wid, summary_obj=None, nadd=80, nphase=21):
    
    if summary_obj is None:
        pass
    
    # get frequency range
    freq_range, amp_range_sub = get_freq_range(cid)
    
    # get winfo    
    winfo = utils.load_osc_motif(cid, wid, tag=tag)[0]
    wlen_max = get_max_len(winfo) + nadd
    mua_sample = np.zeros(2, (len(winfo), wlen_max)) * np.nan
    pid_sample = np.zeros(2, (len(winfo), wlen_max, 2), dtype=int) * np.nan
    
    # sample dataset
    nt_prv = -1
    for n in tqdm(range(len(winfo)), ncols=100):
        nt, tl = winfo[n]
        if nt != nt_prv:
            
            detail_data = summary_obj.load_detail(cid-1, nt)
            pid_set = get_pid_signal(detail_data, freq_range, nbins=nphase, fo=5)
            mua_set = detail_data["mua"]
            
            nt_prv = nt
            
        if tl[0] < 0.6 or tl[1] > detail_data["ts"][-1]-0.5:
            continue
        
        nl = [int(t*srate) for t in tl]
        mua_sample[:, n, :(nl[1]-nl[0]+nadd)] = mua_set[:, (nl[0]-nadd):nl[1]]
        pid_sample[:, n, :(nl[1]-nl[0]+nadd)] = pid_set[:, (nl[0]-nadd):nl[1]]

    return mua_sample, pid_sample

    
def get_pid_signal(detail_data, freq_range, nbins=21, fo=5):
    """
    
    Outputs:
    pid_set (4, n): int
        (fpop-s, fpop-f, spop-s, spop-f)
    
    """
    # filt_signal = [[], []] # fpop, spop
    
    pid_set = np.zeros((4, len(detail_data["ts"])), dtype=int)
    e = np.linspace(-np.pi, np.pi, nbins+1)
    sos_set = [hhsignal.get_sosfilter(freq_range[nf], srate, fo=fo) for nf in range(2)]
    
    for n in range(4):
        i, j = n//2, n%2 # fpop/spop, slow/fast
        v = detail_data["vlfp"][i+1]
        vf = hhsignal.filt(v, sos_set[j])
        phs = np.angle(hilbert(vf))
        pid_set[n] = np.digitize(phs, e)
        
    return pid_set
        
        
def get_filt_signal(detail_data, freq_range, fo=5):
    filt_signal = [[], []] # fpop, spop
    # print(srate, freq_range)
    for nf in range(2):
        # print(freq_range[nf])
        sos = hhsignal.get_sosfilter(freq_range[nf], srate, fo=fo)
        for n in range(2):
            v = detail_data["vlfp"][n+1]
            vf = hhsignal.filt(v, sos)
            filt_signal[n].append(vf)
    # print(sos.shape)
    # filt_signal[0], filt_signal[1] = filt_signal[1], filt_signal[0]
    return np.array(filt_signal) # F (s, f), S (s, f)    
    

def get_max_len(winfo):
    wlen_set = [int((w[1][1] - w[1][0])*srate) for w in winfo]
    return np.max(wlen_set)
    
    
def get_freq_range(cid):
    amp_range = utils.load_pickle(famp_range)["amp_range_set"]
    amp_range_sub = amp_range[cid-1]
    
    freq_range = []
    for n in range(2):
        fr_f = amp_range_sub["fpop"][n]
        fr_s = amp_range_sub["spop"][n]
        
        assert len(fr_f) != 0
        assert len(fr_s) != 0
        
        fr = [min(fr_f[0], fr_s[0]), max(fr_f[1], fr_s[1])]
        freq_range.append(fr)

    return freq_range, amp_range_sub




if __name__=="__main__":
    pass