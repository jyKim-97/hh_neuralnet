import numpy as np
import burst_tools as bt
import pickle as pkl
from tqdm import trange
from argparse import ArgumentParser
from multiprocessing import Pool
import os
from numba import jit
from dataclasses import dataclass
from typing import Tuple

import sys
sys.path.append("/home/jungyoung/Project/hh_neuralnet/include")
import hhtools
import hhsignal

# hard fix parameters
fs = 2000
mbin_t = -1
wbin_t = -1
flim = (0, 0)
std_min = -1
std_max = -1
std_step = -1
nmin_width = -1
arange = (0, 0)
da = -1


@dataclass
class mfoConfig:
    mbin_t: float = 0.1
    wbin_t: float = 1
    flim: Tuple[float, float] = (10, 100)
    std_min: float = 3.3
    std_max: float = 8
    std_step: float = 0.1
    nmin_width: float = -1
    arange: Tuple[float, float] = (0.1, 2.1)
    da: float = 0.4
    fdir: str = ""


def config_params(config: mfoConfig):
    for k, v in vars(config).items():
        globals()[k] = v


def check_params(key):
    print(globals()[key])


def main(fname_th=None, 
         std_min=3.3, std_max=8, std_step=0.1, nmin_width=-1,
         arange=(0.1, 2.1), da=0.4, ncore=20):

    th_psd, fdir_data = load_psd_threshold(fname_th)
    # configuration
    nmin_width = int(0.05/th_psd["psd_params"]["mbin_t"]) if nmin_width == -1 else nmin_width
    config = mfoConfig(fdir=fdir_data,
                       std_min=std_min, std_max=std_max, std_step=std_step,
                       nmin_width=nmin_width,
                       arange=arange, da=da,
                       **th_psd["psd_params"])
    config_params(config)

    summary_obj = hhtools.SummaryLoader(fdir_data)

    pcorr_maps = []
    bcorr_maps = []
    num_detected = []
    for cid in trange(1, summary_obj.num_controls[0]+1):
        pmaps, bmaps, amp_edges, nd = get_corr_map_cluster(summary_obj, cid, th_psd, ncore=ncore)
        pcorr_maps.append(pmaps)
        bcorr_maps.append(bmaps)
        num_detected.append(nd)
    
    with open("./corr_maps.pkl", "wb") as fp:
        pkl.dump({
            "pcorr_maps": pcorr_maps,
            "bcorr_maps": bcorr_maps,
            "num_detected": num_detected,
            "amp_edges": amp_edges
        }, fp)


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--fname_th", required=True)
    parser.add_argument("--fout", required=True)
    parser.add_argument("--std_min", default=3.3, type=float)
    parser.add_argument("--std_max", default=8, type=float)
    parser.add_argument("--std_step", default=0.1, type=float)
    parser.add_argument("--nmin_width", default=-1, type=int)
    return parser


def load_psd_threshold(fname):
    with open(fname, "rb") as fp:
        th_obj = pkl.load(fp)
    return th_obj, th_obj["fdir_data"]


def _get_corr(xset):
    # xset: (2,?,?)
    N = np.shape(xset)[1]
    corr_mats = np.zeros([N, N, 4])
    flag_empty = [np.all(xset[i] == 0) for i in range(2)]
    for tp in [0,1,3]:
        i, j = tp//2, tp%2
        if any([flag_empty[i], flag_empty[j]]): continue
        corr_mats[:, :, tp] = (xset[i] @ xset[j].T) / np.shape(xset[0])[1]
    corr_mats[:,:,2] = corr_mats[:,:,1].copy().T
    return corr_mats


def _get_psd_corr(psd_set):
    norm_psd_set = [(x - np.average(x, axis=1)[:, np.newaxis]) / np.std(x, axis=1)[:, np.newaxis] for x in psd_set]
    return _get_corr(norm_psd_set)


def _get_burst_corr(burst_maps):
    bsets = [(b > 1).astype(int) for b in burst_maps]
    p_burst_corr = _get_corr(bsets)
    # get indep prob
    p_indep = [np.sum(b, axis=1)[:, np.newaxis]/b.shape[1] for b in bsets]
    p_indep_map = _get_corr(p_indep)
    return p_burst_corr - p_indep_map


def _get_psd_single(fname):
    vlfp_t, fs = hhtools.load_vlfp(fname)
    ts = np.arange(len(vlfp_t[0])) / fs
    idt = (ts >= 0.5)
    psd_set = [[], []]
    psd_set[0], _, _ = hhsignal.get_stfft(vlfp_t[1][idt], ts[idt], fs, mbin_t=mbin_t, wbin_t=wbin_t, frange=flim)
    psd_set[1], fpsd, tpsd = hhsignal.get_stfft(vlfp_t[2][idt], ts[idt], fs, mbin_t=mbin_t, wbin_t=wbin_t, frange=flim)
    return psd_set, fpsd, tpsd


def _get_bmap(psd_set, fpsd, th_psd_m=(0, 0), th_psd_s=(0, 0)):

    bmaps = []
    burst_fs = []
    burst_ranges = []
    burst_amps = []
    

    for tp in range(2):
        bmap = bt.find_blob_filtration(psd_set[tp], th_psd_m[tp], th_psd_s[tp],
                                            std_min=std_min, std_max=std_max, std_step=std_step,
                                            nmin_width=nmin_width)
        burst_f, burst_range, burst_amp = bt.extract_burst_attrib(psd_set[tp], fpsd, bmap)

        bmaps.append(bmap)
        burst_fs.append(burst_f)
        burst_ranges.append(burst_range)
        burst_amps.append(burst_amp)
    
    return bmaps, burst_fs, burst_ranges, burst_amps


def _get_bcorr_with_amp(bmaps, burst_amps, arange):
    bmaps_c = np.zeros_like(bmaps)
    for tp in range(2):
        rm_sets = np.where((burst_amps[tp] < arange[0]) | (burst_amps[tp] >= arange[1]))[0]
        bmaps_c[tp] = _clean_burst(bmaps[tp], rm_sets)
    return _get_burst_corr(bmaps_c)


def _clean_burst(burst_map, target_ids):
    bmap2 = burst_map.copy()
    for nt in target_ids:
        bmap2[burst_map == nt] = 0
    return bmap2


def get_corr_map(psd_set, fpsd, 
                 th_psd_m=(0, 0), th_psd_s=(0, 0)):

    psd_corr_map = _get_psd_corr(psd_set)

    bmaps, _, _, burst_amps = _get_bmap(psd_set, fpsd, th_psd_m, th_psd_s)
    edges = np.arange(arange[0], arange[1]+da/2, da)
    if len(edges) == 1:
        edges = arange
    
    b_corr_map = []
    for na in range(len(edges)-1):
        b_corr_map.append(_get_bcorr_with_amp(bmaps, burst_amps, edges[na:na+2]))

    b_corr_map = np.array(b_corr_map) 

    return psd_corr_map, b_corr_map, edges, burst_amps


def get_corr_map_cluster(summary_obj, cid, th_buf, ncore=10):

    if cid < 1:
        raise ValueError("Invalid cluster Id")

    global _get_cmap_cluster
    def _get_cmap_cluster(args):

        job_id = args[0]
        fname = args[1][0]
        th_psd_m = args[1][1]
        th_psd_s = args[1][2]

        psd_set, fpsd, tpsd = _get_psd_single(fname)
        return job_id, get_corr_map(psd_set, fpsd, th_psd_m, th_psd_s)
    
    def _run_parallel(ncore):
        args_job = zip(np.arange(len(args)), args)
        with Pool(ncore) as p:
            outs = p.map(_get_cmap_cluster, args_job)

        id_sort = [o[0] for o in outs] # sort with job_id
        return [outs[i][1] for i in id_sort]
    
    num_itr = summary_obj.num_controls[1]

    args = []
    for n in range(num_itr):
        nid = summary_obj.get_id(cid-1, n)
        fname = os.path.join(summary_obj.fdir, "id%06d_lfp.dat"%(nid))
        args.append((fname, th_buf["th_m"][nid, :], th_buf["th_s"][nid, :]))
    
    res = _run_parallel(ncore)

    psd_corr_map = np.average([r[0] for r in res], axis=0)
    b_corr_map   = np.average([r[1] for r in res], axis=0)
    amp_edges = res[0][2]

    num_detected = np.sum([[len(r[3][0]), len(r[3][1])] for r in res], axis=0)
    print(num_detected)
    return psd_corr_map, b_corr_map, amp_edges, num_detected
    

