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


def get_stfft(x, t, fs, mbin_t=0.1, wbin_t=1, f_range=None, buf_size=100):

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
    
    if f_range is not None:
        idf = (fpsd >= f_range[0]) & (fpsd <= f_range[1])
        psd = psd[idf, :]
        fpsd = fpsd[idf]
    tpsd = ind / fs
    
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
    tlag = np.arange(-max_lag, max_lag+1/srate, 1/srate)
    
    # get normalization block
    # nc = len(cc)
    # num_use = np.zeros(nc)
    # num_use[:nc//2+1] = len(yn) - np.arange(0, nc//2+1)[::-1]
    # num_use[nc//2:] = len(yn) - np.arange(0, nc//2+1)
    
    num_use = len(yn)
    cc = cc/num_use
    
    return cc, tlag


# find peak
def find_corr_peaks(c, prominence=0.05):
    """
    Find peak of cc
    ind_peaks = find_corr_peaks(cc, prominence=0.05)
    ind_peaks[i] = The i th largest peak
    """
    from scipy.signal import find_peaks
    
    ind_peaks, _ = find_peaks(c, prominence=prominence)
    n0 = len(c)//2
    dn = np.abs(ind_peaks - n0)
    ind = np.argsort(dn)

    # ind_peaks = ind_peaks[ind[:3]]

    # use ~10 points
    ind_peaks = ind_peaks[ind[:10]]

    # align peak according to peak amp
    peak_amp = c[ind_peaks]
    
    ind_a = np.argsort(peak_amp)[::-1]
    ind_peaks = ind_peaks[ind_a[:3]]

    return ind_peaks


# find peak
def find_corr_peaks_old(c, prominence=0.05):
    """
    Find peak of cc
    ind_peaks = find_corr_peaks(cc, prominence=0.05)
    ind_peaks[i] = The i th largest peak
    """
    from scipy.signal import find_peaks
    
    ind_peaks, _ = find_peaks(c, prominence=prominence)
    n0 = len(c)//2
    dn = np.abs(ind_peaks - n0)
    ind = np.argsort(dn)

    return ind_peaks[ind[:3]]
