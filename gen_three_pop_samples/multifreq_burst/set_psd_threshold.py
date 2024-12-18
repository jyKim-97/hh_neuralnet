import numpy as np
import os
from argparse import ArgumentParser
from tqdm import trange
import pickle as pkl
from datetime import datetime

import sys
sys.path.append("/home/jungyoung/Project/hh_neuralnet/include")
import hhtools
import hhsignal
import parrun
from functools import partial


# hard fix parameters
fs = 2000
mbin_t = 0.01
wbin_t = 1
flim = (10, 100)

def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--method", default="absolute", choices=("absolute", "absolute_c", "relateive"))
    parser.add_argument("--fdir_data", required=True)
    parser.add_argument("--fout", required=True)
    parser.add_argument("--std_ratio", default=3.29, type=float)
    return parser


def load_cluster_info(fdir_data):
    cinfo = {"cluster_id": [], "cluster_order": [], "nrank": []}
    with open(os.path.join(fdir_data, "picked_cluster.txt"), "r") as fp:
        fp.readline()
        line = fp.readline()
        while line:
            buf = line.split(",")
            cinfo["cluster_id"].append(int(buf[0]))
            cinfo["cluster_order"].append(int(buf[1]))
            cinfo["nrank"].append(int(buf[2]))
            line = fp.readline()
        
    return cinfo


def _get_psd_m12(fs, mbin_t, wbin_t, flim, vlfp_set, ts):
        
        psd_m = np.zeros(2)
        psd_m2 = np.zeros(2)
        for i in range(2):
            psd, fpsd, tpsd = hhsignal.get_stfft(vlfp_set[i], ts, fs,
                                                    mbin_t=mbin_t, wbin_t=wbin_t, frange=flim)
            psd_m[i]  = np.average(psd)
            psd_m2[i] = np.average(psd**2)
            
        return psd_m, psd_m2


def get_attrib_psd(summary_obj):
    
    def _cut_signal(vlfp_set, t_sig):
        idt = t_sig >= 0.5 # t_sig (s)
        return [v[idt] for v in vlfp_set], t_sig[idt]
    
    # explore if psd_m exist
    fname = os.path.join(summary_obj.fdir, "./psd_attrib.pkl")
    if os.path.exists(fname):
        with open(fname, "rb") as fp:
            buf = pkl.load(fp)
        return buf["psd_m"], buf["psd_m2"]

    # get mean & std for each case
    psd_m = np.zeros([summary_obj.num_total, 2])
    psd_m2 = np.zeros([summary_obj.num_total, 2])
    
    global _arg_func
    _arg_func = partial(_get_psd_m12, fs, mbin_t, wbin_t, flim)

    for n in trange(summary_obj.num_total, desc="put..."):
        detail_data = summary_obj.load_detail(n)
        vlfp_set, ts = _cut_signal(detail_data["vlfp"][1:], detail_data["ts"])
        psd_m[n, :], psd_m2[n, :] = _arg_func(vlfp_set, ts)
        
    with open(fname, "wb") as fp:
        pkl.dump({"psd_m": psd_m, "psd_m2": psd_m2}, fp)
    
    return psd_m, psd_m2


def get_th_abs(psd_m, psd_m2, cluster_info):
    N = psd_m.shape[0]
    th_m = np.zeros([N, 2])
    th_s = np.zeros([N, 2])

    nrank_set = np.array(cluster_info["nrank"]).astype(int)
    for nrank in range(3): # NOTE: take care with different rank size
        is_r = nrank_set == nrank
        m1 = np.average(psd_m[is_r, :],  axis=0)
        m2 = np.average(psd_m2[is_r, :], axis=0)
        s = np.sqrt(m2 - m1**2)

        th_m[is_r, :] = m1
        th_s[is_r, :] = s

    return th_m, th_s


def get_th_abs_cluster(psd_m, psd_m2, cluster_info):
    N = psd_m.shape[0]
    th_m = np.zeros([N, 2])
    th_s = np.zeros([N, 2])

    clsuter_id_set = np.unique(cluster_info["cluster_id"])
    for nc in clsuter_id_set:
        is_c = cluster_info["cluster_id"] == nc
        m1 = np.average(psd_m[is_c, :],  axis=0)
        m2 = np.average(psd_m2[is_c, :], axis=0)
        s = np.sqrt(m2 - m1**2)

        th_m[is_c, :] = m1
        th_s[is_c, :] = s

    return th_m, th_s


def get_th_rel(psd_m, psd_m2):
    th_m = psd_m
    th_s = np.sqrt(psd_m2 - psd_m**2)
    return th_m, th_s



def main(std_ratio=0, fdir_data=None, fout=None, method=None):
    cluster_info = load_cluster_info(fdir_data)
    summary_obj = hhtools.SummaryLoader(fdir_data)

    # Check validity
    if summary_obj.num_total != len(cluster_info["cluster_id"]):
        raise ValueError("# of data & cluster id do not match: %d, %d"%(summary_obj.num_total, len(cluster_info["cluster_id"])))
    
    psd_m, psd_m2 = get_attrib_psd(summary_obj)

    # get threshold
    if method == "absolute":
        th_m, th_s = get_th_abs(psd_m, psd_m2, cluster_info)
    elif method == "absolute_c":
        th_m, th_s = get_th_abs_cluster(psd_m, psd_m2, cluster_info)
    elif method == "relative":
        th_m, th_s = get_th_rel(psd_m, psd_m2)
    th_psd = th_m + std_ratio * th_s

    # export
    now = datetime.now()
    th_out = {"th_psd": th_psd,
              "th_m": th_m, "th_s": th_s, "std_ratio": std_ratio,
              "method": method, "fdir_data": fdir_data,
              "psd_params": {"mbin_t": mbin_t, "wbin_t": wbin_t, "flim": flim},
              "updated": "%d-%d-%d"%(now.year, now.month, now.day)}
    
    print("Export result to %s"%(fout))
    with open(fout, "wb") as fp:
        pkl.dump(th_out, fp)
    

if __name__ == "__main__":
    main(**vars(build_argument_parser().parse_args()))
