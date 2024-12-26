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

sys.path.append("/home/jungyoung/Project/hh_neuralnet/extract_osc_motif")
import utils

import tetools as tt
from frites.core import mi_nd_gg


# tag = "_mslow2"
tag = ""


num_process = 4
srate = 2000
fdir_summary = "/home/jungyoung/Project/hh_neuralnet/gen_three_pop_samples_repr/data"+tag

chunk_size = 100
nchunks = 400
# nchunks = 600


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", help="cluster id", required=True, type=int)
    parser.add_argument("--wid", help="word id", required=True, type=int)
    # parser.add_argument("--nsamples", default=1000, type=int)
    parser.add_argument("--nsample", default=100, type=int)
    parser.add_argument("--target", default="mua", type=str, choices=("lfp", "mua"))
    parser.add_argument("--tlag_max", default=40, type=float)
    parser.add_argument("--tlag_min", default=1, type=float)
    parser.add_argument("--tlag_step", default=0.5, type=float)
    parser.add_argument("--fout", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main(cid=5, wid=10, nsample=10,
         tlag_max=40, tlag_min=1, tlag_step=1,
         target="lfp", fout=None, seed=42):
    
    tw = 1.2
    tadd = 0.3
    
    # load oscmotif
    summary_obj = hhtools.SummaryLoader(fdir_summary, load_only_control=True)

    # collect chunks
    nadd = int(tadd * srate)
    v_set = utils.collect_chunk(cid, wid,
                                summary_obj=summary_obj,
                                target=target,
                                srate=srate, 
                                nequal_len=int(tw * srate), nadd=nadd,
                                norm=True, filt_range=None, verbose=True)
    
    nlag_max = int(tlag_max * srate / 1000)
    nlag_min = int(tlag_min * srate / 1000)
    nlag_step = int(tlag_step * srate / 1000)
        
    params = dict(nchunks=nchunks, chunk_size=chunk_size,
                  nmax_delay=nlag_max, nmin_delay=nlag_min, nstep_delay=nlag_step)
    
    mi_true, tlag = compute_ais_true(v_set, nsample, nadd, params)
    
    if fout is None:
        fout = "ais_%d%02d.pkl"%(cid, wid)
    
    with open(fout, "wb") as fp:
        
        info = {"cid": cid, "wid": wid, "nsample": nsample,
                "tw": tw, "tadd": tadd,
                "fdir": fdir_summary,
                "seed": seed, "sample": ('F', 'S')}
        info.update(params)
        
        pkl.dump({
            "info": info,
            "tlag": tlag,
            "mi": mi_true
        }, fp)
        
        
def compute_ais_true(v_set, ntrue, nadd, params):
    seed_set = np.random.randint(10, int(1e8), ntrue)
    f = partial(_compute_ais_true, v_set=v_set, nadd=nadd, **params)
    te_results = utils.par_func(f, seed_set, num_process, desc="AIS_true")
    return unzip_te_result(te_results)
        
        
def _compute_ais_true(seed, v_set=None, nadd=None,
                     nchunks=0, chunk_size=0, **config):
    
    np.random.seed(seed)
    v_set_sub = tt.sample_true(v_set,
                               nadd=nadd,
                               nchunks=nchunks, chunk_size=chunk_size,
                               nmax_delay=config["nmax_delay"])
    
    mi, nlag = compute_ais(v_set_sub, **config)
    tlag = nlag/srate*1e3
    
    return mi, tlag


def compute_ais(v_set, nmax_delay=80, nmin_delay=1, nstep_delay=1):
    
    nlag_set = np.arange(nmin_delay, nmax_delay, nstep_delay)
    
    mi_sample = []
    for nt in range(2):
        xset, yset = [], []
        for nl in nlag_set:
            x = v_set[:,nt,:-nmax_delay]
            n1 = -nmax_delay+nl if nl != nmax_delay else v_set.shape[-1]
            y = v_set[:,nt,nl:n1]
            xset.append(x)
            yset.append(y)
        
        xset = np.stack(xset)
        yset = np.stack(yset)

        mi = mi_nd_gg(xset, yset, demeaned=False, biascorrect=True)
        mi_sample.append(mi.mean(axis=1))
    
    return np.array(mi_sample), nlag_set


def unzip_te_result(te_results):
    tlag = te_results[0][1]
    te_set = np.array([te[0] for te in te_results])
    return te_set, tlag


if __name__ == "__main__":
    main(**vars(build_args().parse_args()))