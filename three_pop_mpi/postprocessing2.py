# Post process after run main.c: This code calculate AC/CC and FFT
# upgraded version of 'postprocessing.py'
import numpy as np
import sys
from tqdm import tqdm
import xarray as xa
import pandas as pd

# Add custom modules
sys.path.append("/home/jungyoung/Project/hh_neuralnet/include/")
import hhtools
import hhsignal
from multiprocessing import Pool
import pickle as pkl
import argparse

srate = 2000
teq = 0.5
mbin_t = 0.05
wbin_t = 1 # s
prominence = 0.05
num_itr = 1

seed = 200


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help="prefix of your simulated data", required=True)
    parser.add_argument("--fout", help="output file name", required=True)
    parser.add_argument("-n", help="the # of iteration", type=int)
    return parser
    

def pick_sample_data(data):
    tmax = data["ts"][-1]
    t0 = np.random.rand() * (tmax-teq-wbin_t-0.1) + teq 
    n0 = int(t0 * srate)
    n1 = n0 + int(wbin_t * srate)

    vlfp = [data["vlfp"][0][n0:n1],
            data["vlfp"][1][n0:n1],
            data["vlfp"][2][n0:n1]]
    return vlfp, t0


def get_ac2_peak(x, prominence=0.01):
    # Need to return 2nd peak lag, mag
    ac, tlag = hhsignal.get_correlation(x, x, srate, max_lag=0.2)
    idp_1st, idp_large = hhsignal.detect_peak(ac, prominence=prominence, mode=3)
    return ac[idp_large[1]], tlag[idp_large[1]], ac[idp_1st[1]], tlag[idp_1st[1]]


def get_cc_peak(x, y, prominence=0.01):
    # Need to return 1nd peak lag, mag
    cc, tlag = hhsignal.get_correlation(x, y, srate, max_lag=0.2)
    idp = hhsignal.detect_peak(cc, prominence=prominence, mode=0)
    return cc[idp[0]], tlag[idp[0]]


def get_pwr(vlfp_t, ts, f_targets):
    # f_targets: (4,)
    psd, fpsd, tpsd = hhsignal.get_stfft(vlfp_t, ts, srate, mbin_t=mbin_t, wbin_t=wbin_t, frange=(3, 100))
    pwrs = np.zeros_like(f_targets)
    for n, ft in enumerate(f_targets):
        nf = np.argmin(np.abs(fpsd - ft))
        pwrs[n] = pwrs[nf]
    return pwrs


def slice_t2(data2, t, trange):
    idt = (t >= trange[0]) & (t < trange[1])
    return data2[:, idt]


def find_nn_ind(x, xtarget):
    return np.argmin(np.abs(x - xtarget))


def extract_single_result(data):
    # nid: simulation number
    keys = ("ac2p_large", "ac2p_1st", "tlag_large", "tlag_1st", "cc1p", "tlag_cc", "pwr_1st", "pwr_large")
    
    res = xa.DataArray(np.zeros([len(keys), 2, 2]), 
                       dims=("data", "pop_type", "momentum"),
                       coords={"data": list(keys),
                               "pop_type": [0, 1],
                               "momentum": ["1st", "2nd"]})
    
    psd, fpsd, tpsd = hhsignal.get_stfft(data["vlfp"][0], data["ts"], 2000,
                                         mbin_t=mbin_t, wbin_t=wbin_t, frange=(3, 100))

    for n in range(num_itr):
        vlfp, tr = pick_sample_data(data)
        
        psd1 = np.average(slice_t2(psd, tpsd, tr), axis=1)
        for tp in range(2):
            vals = get_ac2_peak(vlfp[tp+1], prominence=prominence)
            for k, v in zip(("ac2p_large", "tlag_large", "ac2p_1st", "tlag_1st"), vals):
                res.loc[k, tp, "1st"] += v
                res.loc[k, tp, "1st"] += v**2

            nf_1st   = find_nn_ind(fpsd, 1/res.loc["tlag_1st",   tp, "1st", n])
            nf_large = find_nn_ind(fpsd, 1/res.loc["tlag_large", tp, "1st", n])
            res.loc["pwr_1st",   tp, "1st"] = psd1[nf_1st]
            res.loc["pwr_1st",   tp, "2nd"] = psd1[nf_1st]**2
            res.loc["pwr_large", tp, "1st"] = psd1[nf_large]
            res.loc["pwr_large", tp, "1st"] = psd1[nf_large]**2
            
        c = res.loc[("cc1p", "tlag_cc"), tp, "1st", n] = get_cc_peak(vlfp[1], vlfp[2], prominence=prominence)
        res.loc["cc1p_large", tp, "1st"] += c
        res.loc["cc1p_large", tp, "2nd"] += c**2
    
    return res, data["job_id"]


def extract_result(summary_obj, ncore=50):
    
    args = []
    for nt in range(summary_obj.num_total):
        data = summary_obj.load_detail(nt)
        data["job_id"] = nt
        del(data["step_spk"])
        args.append(data)
        
    # with Pool(ncore=ncore) as p:
        # outs = p.map(extract_single_result, args)
    
    outs = []
    for n in range(summary_obj.num_total):
        outs.append(extract_single_result(args[n]))
    id_sort = np.argsort([o[1] for o in outs])
    res_set = xa.concat([outs[n][0] for n in id_sort.astype(int)], dim="id")
    
    nt = summary_obj.num_controls[-1]
    
    buf = []
    for n in range(summary_obj.num_total // nt):
        buf.append(res_set.isel(id=slice(n*nt, (n+1)*nt)).sum("id"))
    res_avg = xa.concat(buf, dim=np.arange(summary_obj.num_total // nt))
    
    res_avg = res_avg.expand_dims(dim={"type": ["mean", "std"]}, axis=-1)
    res_avg.loc[{"type": "mean"}] = res_avg.loc[{"momentum": "1st"}] / (nt * num_itr)
    res_avg.loc[{"type": "std"}]  = res_avg.loc[{"momentum": "2nd"}] / (nt * num_itr) - res_avg.loc[{"type": "mean"}]
    res_avg.loc[{"type": "std"}]  = np.sqrt(res_avg.loc[{"type": "std"}])
    res_avg = res_avg.drop("momentum")
    
    return res_avg    
    

def main():
    pass