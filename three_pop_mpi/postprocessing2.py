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

keys = ("ac2p_large", "tlag_large", "ac2p_1st", "tlag_1st", "pwr_1st", "pwr_large", "cc1p", "tlag_cc")

seed = 200


def main(prefix=None, fout="./dynamic_orders.pkl", nitr=5, ncore=50):
    global num_itr
    num_itr = nitr
    
    summary_obj = hhtools.SummaryLoader(prefix, load_only_control=True)
    df_res = extract_result(summary_obj, ncore=ncore)
    df_res.to_netcdf(fout)
    print("Calculation done, exported to %s"%(fout))


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help="prefix of your simulated data", required=True)
    parser.add_argument("--fout", help="output file name", required=True)
    parser.add_argument("--nitr", help="the # of iteration", type=int)
    parser.add_argument("--ncore", help="the # of cores to use", type=int)
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
    res = np.zeros([len(keys), 2, 2]) # key, pop_type, momentum (1/2)
    psd, fpsd, tpsd = hhsignal.get_stfft(data["vlfp"][0], data["ts"], 2000,
                                         mbin_t=mbin_t, wbin_t=wbin_t, frange=(3, 100))

    for _ in range(num_itr):
        vlfp, t0 = pick_sample_data(data)
        tr = [t0, t0 + wbin_t*srate]
        
        psd1 = np.average(slice_t2(psd, tpsd, tr), axis=1)
        for tp in range(2):
            vals = get_ac2_peak(vlfp[tp+1], prominence=prominence)
            for n, v in enumerate(vals):
                res[n, tp, 0] += v
                res[n, tp, 1] += v**2
            
            nf_large = find_nn_ind(fpsd, 1/vals[1])
            nf_1st   = find_nn_ind(fpsd, 1/vals[3])
            res[4, tp, 0] += psd1[nf_1st]
            res[4, tp, 1] += psd1[nf_1st]**2
            res[5, tp, 0] += psd1[nf_large]
            res[5, tp, 1] += psd1[nf_large]**2
        
        cc1p, cc_tlag = get_cc_peak(vlfp[1], vlfp[2], prominence=prominence)
        res[6, 0, 0] += cc1p
        res[6, 0, 1] += cc1p**2
        res[7, 0, 0] += cc_tlag
        res[7, 0, 1] += cc_tlag**2
    
    res /= num_itr
    return data["job_id"], res


def extract_result(summary_obj, ncore=50) ->xr.DataArray:
    
    args = []
    for nt in range(summary_obj.num_total):
        data = summary_obj.load_detail(nt)
        data["job_id"] = nt
        del(data["step_spk"])
        args.append(data)
        
    with Pool(ncore) as p:
        outs = p.map(extract_single_result, args)
    
    id_sort = np.argsort([o[0] for o in outs])
    res_set = np.stack([outs[n][1] for n in id_sort.astype(int)])
    
    nt = summary_obj.num_controls[-1]
    nums = summary_obj.num_total // nt
    
    res_avg = np.zeros([nums, len(keys), 2, 2]) # data, pop_type (or from), mean / std
    for n in range(nums):
        res_avg[n,:,:,:] = np.average(res_set[n*nt:(n+1)*nt,:,:,:], axis=0)
    res_avg[:,:,:,1] = np.sqrt(res_avg[:,:,:,1] - res_avg[:,:,:,0]**2)
    res_avg = np.swapaxes(res_avg, 0, 1)
    res_avg = np.swapaxes(res_avg, 1, 3) # data, pop_type (or from), mean / std, iter
    
    shape = list(res_avg.shape)
    shape = shape[:-1] + list(summary_obj.num_controls[:-1])
    res_avg = np.reshape(res_avg, shape)
    
    df = xr.DataArray(res_avg,
                      dims=("key", "pop", "type", "alpha", "beta", "rank", "w"),
                      coords={"key": list(keys),
                              "pop": ["F", "S"],
                              "type": ["mean", "std"],
                              "alpha": summary_obj.controls["alpha_set"],
                              "beta": summary_obj.controls["beta_set"],
                              "rank": summary_obj.controls["rank_set"],
                              "w": summary_obj.controls["p_ratio_set"]}
                      )
    
    return df


if __name__ == "__main__":
    main(**vars(build_arg_parser().parse_args()))