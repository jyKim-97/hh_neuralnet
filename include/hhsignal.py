# Signal processing library
import numpy as np
from scipy.signal import correlate


# Fourier transform
def get_fft(x, fs, nbin=None, nbin_t=None, frange=None):
    if nbin is None and nbin_t is None:
        N = len(x)
    elif nbin_t is not None:
        N = int(nbin_t*fs)
    elif nbin is not None:
        N = nbin

    yf = np.fft.fft(x, axis=0, n=N)
    yf = 2/N * np.abs(yf[:N//2])
    freq = np.linspace(0, 1/2*fs, N//2)

    if frange is not None:
        if frange[0] is None:
            frange[0] = freq[0]
        if frange[1] is None:
            frange[1] = freq[-1]
        idf = (freq >= frange[0]) & (freq <= frange[1])
        yf = yf[idf]
        freq = freq[idf]

    return yf, freq


def get_stfft(x, t, fs, mbin_t=0.1, wbin_t=1, frange=None, buf_size=100):

    wbin = int(wbin_t * fs)
    mbin = int(mbin_t * fs)
    window = np.hanning(wbin)
    
    ind = np.arange(wbin//2, len(t)-wbin//2, mbin, dtype=int)
    psd = np.zeros([wbin//2, len(ind)])
    
    n_id = 0
    while n_id < len(ind):
        n_buf = min([buf_size, len(ind)-n_id])
        y = np.zeros([wbin, n_buf])

        for i in range(n_buf):
            n = i + n_id
            n0 = max([0, ind[n]-wbin//2])
            n1 = min([ind[n]+wbin//2, len(t)])
            y[n0-(ind[n]-wbin//2):wbin-(ind[n]+wbin//2)+n1, i] = x[n0:n1]
        y = y * window[:,np.newaxis]
        yf, fpsd = get_fft(y, fs)
        psd[:, n_id:n_id+n_buf] = yf

        n_id += n_buf
    
    if frange is not None:
        idf = (fpsd >= frange[0]) & (fpsd <= frange[1])
        psd = psd[idf, :]
        fpsd = fpsd[idf]
    # tpsd = ind / fs
    tpsd = t[ind]
    
    return psd, fpsd, tpsd


def get_frequency_peak(vlfp, fs=2000):
    from scipy.signal import find_peaks

    yf, freq = get_fft(vlfp, fs)
    idf = (freq >= 2) & (freq < 200)
    yf = yf[idf]
    freq = freq[idf]

    inds = find_peaks(yf)[0]
    n = np.argmax(yf[inds])
    return freq[inds[n]]


# Get cross-correlation
def get_correlation(x, y, srate, max_lag=None):
    # positive: y leads x
    # negative: x leads y
    
    xn = x - np.average(x)
    yn = y - np.average(y)
    std = [np.std(xn), np.std(yn)]

    if max_lag is None:
        max_lag = len(x)/srate
    max_pad = max_lag * srate

    if (std[0] == 0) or (std[1] == 0):  
        return np.zeros(2*max_pad+1)
    
    pad = np.zeros(int(max_pad))
    xn = np.concatenate((pad, xn, pad))
    cc = correlate(xn, yn, mode="valid", method="fft")/std[0]/std[1]
    
    # cc = correlate(xn, yn, mode="full", method="fft")/std[0]/std[1]
    tlag = np.arange(-max_lag, max_lag+1/srate/10, 1/srate)
    
    # get normalization block
    # nc = len(cc)
    # num_use = np.zeros(nc)
    # num_use[:nc//2+1] = len(yn) - np.arange(0, nc//2+1)[::-1]
    # num_use[nc//2:] = len(yn) - np.arange(0, nc//2+1)
    
    num_use = len(yn)
    cc = cc/num_use
    
    return cc, tlag


def detect_peak(c, prominence=0.01, mode=0):
    # mode: full(0) / 2nd peak (criteria: distance, 1) / 2nd peak (criteria: amplitude, 2) / 2nd peak (distance, amp)
    from scipy.signal import find_peaks

    if (mode < 0) or (mode > 3):
        raise ValueError("Invalid mode: %d"%(mode))
    
    ind_peaks, _ = find_peaks(c, prominence=prominence)
    amp_peaks = c[ind_peaks]
    n0 = ind_peaks[np.argmax(amp_peaks)] # find the center

    # align
    dn = np.abs(ind_peaks - n0)
    ind_sort = np.argsort(dn)
    
    if mode == 0:
        return ind_peaks[ind_sort]

    # use only 5 peaks (two peaks for each side)
    tmp_peaks = ind_peaks[ind_sort[:5]]
    dn2 = np.abs(tmp_peaks - n0)

    if dn2[1] == dn2[2]:
        if dn2[3] == dn2[4]:
            ind_peaks = [tmp_peaks[0], n0+dn2[1], n0+dn2[3]]
        else:
            ind_peaks = [tmp_peaks[0], n0+dn2[1], tmp_peaks[3]]
    else:
        ind_peaks = tmp_peaks[:3]

    if mode == 1:
        return ind_peaks[:2]
    else: 
        c1, c2 = c[ind_peaks[1]], c[ind_peaks[2]]
        if c1 < c2:
            ind_peaks_l = [ind_peaks[0], ind_peaks[2]]
        else:
            ind_peaks_l =  ind_peaks[:2]
        
        if mode == 2:
            return ind_peaks_l
        else:
            return ind_peaks[:2], ind_peaks_l
    

""" # Legacy
# def detect_peak(c, srate=2000, tol_t=1e-2, tol_c=0.2, prominence=0.01, mode=0):
#     # mode: full(0) / largest (1) / first (2) / all (3)
    
#     from scipy.signal import find_peaks
    
#     def sort_peaks(ind_p, n0, del_overlap=True):
#         dn = np.abs(ind_p - n0)
#         ind = np.argsort(dn)
#         tmp_peaks = ind_p[ind[:10]] # use ~10 points
        
#         if not del_overlap:
#             return tmp_peaks
        
#         dn_abs = np.abs(dn[ind])
#         sort_ind_p = [tmp_peaks[0]]
#         dn0 = dn_abs[0]
#         for n in range(1, len(tmp_peaks)):
#             if dn0 == dn_abs[n]:
#                 sort_ind_p[-1] = dn0 + n0
#             else:
#                 dn0 = dn_abs[n]
#                 sort_ind_p.append(tmp_peaks[n])

#         return sort_ind_p
    
#     ind_peaks, _ = find_peaks(c, prominence=prominence)
#     if mode == 0:
#         return sort_peaks(ind_peaks, len(c)//2, del_overlap=False)
    
#     ind_peaks = sort_peaks(ind_peaks, len(c)//2, del_overlap=True)
    
#     # check assumption
#     ind_max = ind_peaks[np.argmax(c[ind_peaks])]
#     if ind_max != ind_peaks[0]:
#         nid = np.where(ind_peaks == ind_max)[0][0]
#         ind_peaks[0], ind_peaks[nid] = ind_peaks[nid], ind_peaks[0]
#         # raise ValueError("Unexpected scenario, ind_max: %d, ind_peak1: %d"%(ind_max, ind_peaks[0]))
    
#     if mode == 1: # largest
#         ind_peaks = detect_2nd_largest(ind_peaks, c, tol_c=tol_c, tol_t=tol_t, srate=srate)
#     elif mode == 2: # first peak
#         ind_peaks = ind_peaks[:2]
#     elif mode == 3:
#         idp_large = detect_2nd_largest(ind_peaks, c, tol_c=tol_c, tol_t=tol_t, srate=srate)
#         idp_1st = ind_peaks[:2]
#         return idp_large, idp_1st
#     else:
#         raise ValueError("Wrong mode typped, choose 0 (largest) or 1 (first)")    

#     return ind_peaks
    

# def detect_2nd_largest(ind_p, c, tol_c=0.2, tol_t=1e-2, srate=2000):
#     dn_left = (ind_p[1:] - ind_p[0])
#     cp_left = c[ind_p[1:]]
#     # sort_ind_p = [ind_p[0]]
    
#     # find the largest peak
#     ind_max = np.argmax(cp_left)
#     dn_max, cp_max = dn_left[ind_max], cp_left[ind_max]
    
#     if ind_max == 0:
#         return [ind_p[0], dn_max + ind_p[0]]
    
#     for n in range(len(dn_left)):
#         if n == ind_max:
#             continue
#         a = dn_max // dn_left[n]
#         b = dn_max - a * dn_left[n]
        
#         flag = True
#         flag = flag and (b/srate < tol_t*a) # condition 1
#         flag = flag and np.abs(cp_max - cp_left[n])/np.abs(cp_max) < tol_c
        
#         if flag:
#             ind_max = n
#             dn_max, cp_max = dn_left[n], cp_left[n]
    
#     return [ind_p[0], dn_max + ind_p[0]]

"""
