# Post process after run main.c: This code calculate AC/CC and FFT
# upgraded version of 'postprocessing.py'
import numpy as np
import sys
from tqdm import tqdm
import xarray as xr
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
_ncore = 10

keys_dyna = ("ac2p_large", "tlag_large", "ac2p_1st", "tlag_1st",
             "pwr_large_ft", "tlag_large_ft", "pwr_1st_ft", "tlag_1st_ft",
             "cc1p", "tlag_cc", "leading_ratio", "leading_ratio(abs)", "dphi")

keys_order = ("chi", "cv", "frs_m")


def main(prefix=None, fout=None, nitr=5, ncore=50, seed=200):
    global num_itr, _ncore
    num_itr = nitr
    _ncore = ncore
    
    np.random.seed(seed)
    summary_obj = hhtools.SummaryLoader(prefix)
    summary_obj.summary["chi"][summary_obj.summary["chi"] > 1] = np.nan
    df_res = extract_result(summary_obj)
    df_res.attrs = {"srate": srate, "teq": teq, "mbin_t": mbin_t, "wbin_t": wbin_t,
                    "prominence": prominence, "num_itr": num_itr, "prefix": prefix,
                    "seed": seed, "date": read_date()}
    df_res.to_netcdf(fout)
    print("Calculation done, exported to %s"%(fout))


def read_date():
    from datetime import datetime
    
    date_now = datetime.now()
    return "%d-%d-%d"%(date_now.year, date_now.month, date_now.day)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help="prefix of your simulated data", required=True)
    parser.add_argument("--fout", help="output file name", default="./dynamic_orders.nc")
    parser.add_argument("--nitr", help="the # of iteration", type=int)
    parser.add_argument("--ncore", help="the # of cores to use", type=int)
    parser.add_argument("--seed", help="seed used to calcualte", default=200, type=int)
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
    
    idp3 = idp[np.argsort(np.abs(tlag[idp]))[:3]] # use first 3 points
    if np.abs(idp3[0]) < 1e-5:
        idp3 = idp3[:2]
    
    npeak = idp3[np.argmax(cc[idp3])]
    
    return cc[npeak], tlag[npeak]


def find_peak2(sig):
    from scipy.signal import find_peaks
    idp = find_peaks(sig)[0]
    id_sort = np.argsort(sig[idp])[::-1]
    idp = idp[id_sort]
    
    n0 = idp[0]
    n = 1
    while np.abs(n0 - idp[n]) < 5:
        n += 1
    
    n1 = idp[n]
    if sig[idp[n]] < sig[n0]/3:
        n1 = n0

    return n0, n1


def get_fft_peak(x):
    ac, tlag = hhsignal.get_correlation(x, x, srate, max_lag=0.2)
    yf, freq = hhsignal.get_fft(ac, srate, frange=(1, 120))
    n0, n1 = find_peak2(yf)
    
    return yf[n0], 1/freq[n0], yf[n1], 1/freq[n1]


def extract_single_result(sample):
    # nid: simulation number
    data = sample.load()
    res = np.zeros([len(keys_dyna), 3, 2]) # key, pop_type, momentum (1/2)
    # psd, fpsd, tpsd = hhsignal.get_stfft(data["vlfp"][0], data["ts"], 2000,
    #                                      mbin_t=mbin_t, wbin_t=wbin_t, frange=(3, 100))
    
    tau_ac_peaks = np.zeros([num_itr, 2]) # two subpopulations (F: ac1, S: ac_large)
    tau_cc_peaks = np.zeros(num_itr)
    num_lead = 0
    
    for nid in range(num_itr):
        vlfp, t0 = pick_sample_data(data)
        
        for tp in range(3):
            vals_ac2 = get_ac2_peak(vlfp[tp], prominence=prominence) # detect peak based on auto-correlation
            vals_ft = get_fft_peak(vlfp[tp]) # detect peak based on FT
            
            for n in range(4): # of vals_ac2, vals_ft
                res[n, tp, 0] += vals_ac2[n]
                res[n, tp, 1] += vals_ac2[n]**2
                res[n+4, tp, 0] += vals_ft[n]
                res[n+4, tp, 1] += vals_ft[n]**2
            
            if tp > 0:
                tau_ac_peaks[nid, tp-1] = vals_ft[1]
        
        cc1p, cc_tlag = get_cc_peak(vlfp[1], vlfp[2], prominence=prominence)
        res[8, 1, 0] += cc1p
        res[8, 1, 1] += cc1p**2
        res[9, 1, 0] += cc_tlag # tlag
        res[9, 1, 1] += cc_tlag**2
        
        tau_cc_peaks[nid] = cc_tlag
        if abs(cc_tlag) > 1e-3:
            num_lead += 1
            
    res /= num_itr
        
    # calculate lead-lag ratio
    pos_lead = num_lead / num_itr
    if (pos_lead > 0.1):
        res[10, 1, 0] = (np.sum(tau_cc_peaks > 0) - np.sum(tau_cc_peaks < 0)) / num_itr
        res[11, 1, 0] = num_lead / num_itr
        
        exp_sum = 0
        for nid in range(num_itr):
            if tau_cc_peaks[nid] <= 0:
                # if tau_cc < 0: 'fast subpop' lead
                T = tau_ac_peaks[nid, 0]
                tau_cc = -tau_cc_peaks[nid]
            else:
                # if tau_cc > 0: 'slow subpop' lead
                T = tau_ac_peaks[nid, 1]
                tau_cc = tau_cc_peaks[nid]
            
            dphi = 2*np.pi*tau_cc % T
            # print("[%4d] dphi: %.3f, cc_lag: %.3f, T: %.2f"%(data["job_id"], dphi, tau_cc, T))
            exp_sum += np.exp(1j * dphi)
        
        dphi_avg = np.angle(exp_sum)
        res[12, 1, 0] = dphi_avg
        
        exp_avg = exp_sum / num_itr
        r2 = np.real(exp_avg * np.conj(exp_avg))
        res[12, 1, 1] = 1 - r2
    
    # print("dphi_avg: %.4f, dphi_var: %.4f"%(res[12, 1, 0], res[12, 1, 1]))
    
    # copy data to subpop
    res[8:, 2, :] = res[8:, 1, :]
    
    nid = data["nid"]
    del data
    
    return nid, res


def construct_args(summary_obj, desc=None):
    
    global _load_single
    def _load_single(nt):
        data = summary_obj.load_detail(nt, load_now=False)
        return nt, data

    dataset = parrun(_load_single, np.arange(summary_obj.num_total, dtype=int), desc=desc)

    # dataset = []
    
    # p = Pool(_ncore)
    # for n in tqdm(range(summary_obj.num_total)):
        # _, data = p.imap(_load_single, )
    
    # p.close()
    # p.join()
    
    # for n in tqdm(range(summary_obj.num_total)):
    #     _, data = _load_single(n)
    #     dataset.append(data)
    return dataset


def parrun(func, args, desc=None):
    outs = []
    p = Pool(_ncore)
    with tqdm(total=len(args), desc=desc) as pbar:
        if _ncore == 1:
            for res in args:
                outs.append(func(res))
                pbar.update() 
        else:
            for n, res in enumerate(p.imap(func, args)):
                outs.append(res)
                pbar.update()
            
    id_sort = np.argsort([o[0] for o in outs])
    res = [outs[i][1] for i in id_sort]
    p.close()
    p.join()
    
    return res


def align_order(summary_obj):
    res_order = np.zeros([len(keys_order), 3, 2] + list(summary_obj.num_controls[:-1]))
    
    for i, key in enumerate(keys_order):
        for nf, func in enumerate((np.nanmean, np.nanstd)):
            tmp = func(summary_obj.summary[key], axis=4) # na, nb, nc, nw, pop_type
            for tp in range(3):
                res_order[i, tp, nf] = tmp[:, :, :, :, tp]
    
    return res_order


def extract_result(summary_obj) -> xr.DataArray:
    
    dataset = construct_args(summary_obj, desc="Load Dataset...")
    res_set = np.stack(parrun(extract_single_result, dataset, desc="Calculating...."))
    
    nt = summary_obj.num_controls[-1]
    nums = summary_obj.num_total // nt
    
    res_dyna = np.zeros([len(keys_dyna), 3, 2, nums]) # data, pop_type (or from), mean / std
    # calculate leading_ratio
    
    for n in range(nums):
        res_dyna[:,:,:,n] = np.average(res_set[n*nt:(n+1)*nt,:,:,:], axis=0)
    res_dyna[:-1,:,1,:] = res_dyna[:-1,:,1,:] - res_dyna[:-1,:,0,:]**2 # data, pop_type (or from), mean / var, iter
    
    shape = list(res_dyna.shape)
    shape = shape[:-1] + list(summary_obj.num_controls[:-1])
    res_dyna = np.reshape(res_dyna, shape)
    
    res_order = align_order(summary_obj)
    
    res_avg = np.concatenate((res_order, res_dyna))
    keys = list(keys_order) + list(keys_dyna)
    
    df = xr.DataArray(res_avg,
                      dims=("key", "pop", "type", "alpha", "beta", "rank", "w"),
                      coords={"key": list(keys),
                              "pop": ["T", "F", "S"],
                              "type": ["mean", "var"],
                              "alpha": summary_obj.controls["alpha_set"],
                              "beta": summary_obj.controls["beta_set"],
                              "rank": summary_obj.controls["rank_set"],
                              "w": summary_obj.controls["p_ratio_set"]}
                      )
    
    return df


if __name__ == "__main__":
    main(**vars(build_arg_parser().parse_args()))