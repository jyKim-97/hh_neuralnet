import numpy as np
import sys
sys.path.append("../include")
import hhsignal

def get_spec_subset(detail, mbin_t=0.05, wbin_t=0.5, srate=2000, frange=(5, 200)):
    psd_set = [None, None]
    psd_set[0], fpsd, tpsd = hhsignal.get_stfft(detail["vlfp"][1], detail["ts"], srate, mbin_t=mbin_t, wbin_t=wbin_t, frange=frange)
    psd_set[1], fpsd, tpsd = hhsignal.get_stfft(detail["vlfp"][2], detail["ts"], srate, mbin_t=mbin_t, wbin_t=wbin_t, frange=frange)
    psd_set = np.array(psd_set)
    
    idt = tpsd >= 0.5
    psd_set = psd_set[:,:,idt]
    tpsd = tpsd[idt]
    
    return psd_set, fpsd, tpsd


def get_spec_line(detail, amp_range, params_spec=None):
    # Fs-Ff-Ss-Sf
    if params_spec is None:
        params_spec = dict(mbin_t=0.05, wbin_t=0.5, srate=2000, frange=(5, 200))
    
    psd_set, fpsd, tpsd = get_spec_subset(detail, **params_spec)
    
    psd_line = np.zeros((4, len(tpsd)))
    for tp, k in enumerate(("fpop", "spop")):
        for i in range(2):
            if len(amp_range[k][i]) == 0: continue
            
            n = 2*tp + i
            idf = (fpsd >= amp_range[k][i][0]) & (fpsd < amp_range[k][i][1])
            psd_line[n,:] = psd_set[tp,idf,:].mean(axis=0)
            
    psd_dict = dict(psd=psd_set, tpsd=tpsd, fpsd=fpsd)
    
    return psd_line, tpsd, psd_dict
            
            
def norm_minmax(arr):
    amax = arr.max(axis=1, keepdims=True)
    amax[amax == 0] = 1
    amin = arr.min(axis=1, keepdims=True)
    
    return (arr-amin) / (amax-amin)


def digitize(arr, nlevel=10):
    if np.all(arr == 0):
        return arr
    
    e = np.linspace(0,1,nlevel+1)
    e[-1] += 0.05
    ad = np.digitize(arr, e) - 1
    assert (np.min(ad) == 0) and (np.max(ad) == nlevel-1)
    return ad


def identify_long_seg(arr, min_len=100):
    arr = np.asarray(arr)
    id_seg = []

    start = 0
    while start < len(arr):
        value = arr[start]
        end = start + 1
        while end < len(arr) and arr[end] == value:
            end += 1
        # Check the length of the segment
        if (end - start) > min_len:
            id_seg.append((start, end, arr[start]))
            # result[start:end] = -1
        start = end

    return id_seg