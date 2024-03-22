import numpy as np
import sys

sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import hhsignal
import pickle as pkl
from tqdm import tqdm
from multiprocessing import Pool
from numba import jit


fs = int(2000)
tmax = 10.5
teq  = 0.5
wbin_t = 1
mbin_t = 0.01
frange = (2/wbin_t, 100)
nsample = 10
len_f = int((frange[1] - frange[0]) * wbin_t)

_ncore = 4

# load sobj
sobj = None


def main(fdir="./data"):
    global sobj
    
    sobj = hhtools.SummaryLoader(fdir)
    num_itr = sobj.num_controls[1]
    id_set = [[i, j] for i in range(1, 9) for j in range(num_itr)]
    
    p = Pool(_ncore)
    
    corr_sets = []
    with tqdm(total=len(id_set)) as pbar:
        for n, res in enumerate(p.imap(get_corr_map, id_set)):
            corr_sets.append(res[0])
            pbar.update()

    fpsd = res[1]
    
    corr_maps = np.zeros([8, 4, len_f, len_f])
    for n in range(len(id_set)):
        cid, _ = id_set[n]
        corr_maps[cid-1] += corr_sets[n] / num_itr
        
    
    with open("./corr_maps_all.pkl", "wb") as fp:
        pkl.dump({"corr_maps": corr_maps, "fpsd": fpsd,
                  "attrs":{
                      "wbin_t": wbin_t, "mbin_t": mbin_t, "frange": frange
                  }}, fp)
    print("Printed to corr_maps_all.pkl")
    

def get_corr_map(id_set):
    
    cid, nitr = id_set[0], id_set[1]
    corr_maps = np.zeros([4, len_f, len_f])
    detail = sobj.load_detail(cid-1, nitr)
    
    psd_set = []
    for n in range(3):
        psd, fpsd, _ = hhsignal.get_stfft(detail["vlfp"][n], detail["ts"], fs, mbin_t=mbin_t, wbin_t=wbin_t, frange=frange)
        psd_set.append(psd)

    for n in range(nsample):
        n0 = int((np.random.rand() * (tmax - teq - 2*wbin_t) + teq) / mbin_t)
        n1 = n0 + int(wbin_t/mbin_t)
        
        # auto-correlation
        corr_maps[0] += psd_auto_corr(psd_set[0][:, n0:n1]) / nsample
        corr_maps[1] += psd_auto_corr(psd_set[1][:, n0:n1]) / nsample
        corr_maps[2] += psd_auto_corr(psd_set[2][:, n0:n1]) / nsample
        
        # cross-correlation
        corr_maps[3] += psd_cross_corr(psd_set[1][:, n0:n1], psd_set[2][:, n0:n1]) / nsample
                    
    return corr_maps, fpsd


@jit(nopython=True)
def psd_auto_corr(psd):
    len_f = len(psd)
    psd_corr = np.zeros((len_f, len_f))
    for nf1 in range(len(psd)):
        y1 = psd[nf1]
        for nf2 in range(nf1, len(psd)):
            y2 = psd[nf2]
            psd_corr[nf1, nf2] = corr(y1, y2)    
    return psd_corr


@jit(nopython=True)
def psd_cross_corr(psd1, psd2):
    len_f = len(psd1)
    psd_corr = np.zeros((len_f, len_f))
    for nf1 in range(len(psd1)):
        y1 = psd1[nf1]
        for nf2 in range(len(psd1)):
            y2 = psd2[nf2]
            psd_corr[nf1, nf2] = corr(y1, y2)    
    return psd_corr


@jit(nopython=True)
def corr(x, y):
    sx = np.std(x)
    sy = np.std(y)
    if sx == 0 or sy == 0:
        return 0
    
    _x = x - np.average(x)
    _y = y - np.average(y)
    return np.sum(_x * _y) / len(_x) / sx / sy
    
    
if __name__ == "__main__":
    main()