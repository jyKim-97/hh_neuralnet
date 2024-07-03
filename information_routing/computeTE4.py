"""
Thiu version uses Gaussian coupula method with frites
"""

import numpy as np
import utils
import argparse

import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import pickle as pkl
from functools import partial

import tetools as tt


num_process = 4
srate = 2000
fdir_summary = "/home/jungyoung/Project/hh_neuralnet/gen_three_pop_samples_repr/data"

chunk_size = 100
nchunks = 200


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", help="cluster id", required=True, type=int)
    parser.add_argument("--wid", help="word id", required=True, type=int)
    # parser.add_argument("--nsamples", default=1000, type=int)
    parser.add_argument("--ntrue", default=100, type=int)
    parser.add_argument("--nsurr", default=1000, type=int)
    parser.add_argument("--method", default="naive", type=str, choices=("naive", "spo", "mit", "2d", "full"))
    parser.add_argument("--target", default="lfp", type=str, choices=("lfp", "mua"))
    parser.add_argument("--nhist", default=1, type=int)
    parser.add_argument("--tlag_max", default=40, type=float)
    parser.add_argument("--tlag_min", default=1, type=float)
    parser.add_argument("--tlag_step", default=0.5, type=float)
    parser.add_argument("--fout", default=None, type=str)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main(cid=5, wid=10, ntrue=10, nsurr=1000, nhist=1, 
         tlag_max=40, tlag_min=1, tlag_step=1,
         method="naive", target="lfp", fout=None, seed=42):
    
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
    
    # compute TE
    if method in ("naive", "spo", "mit"):
        fte = tt.compute_te
        params = dict(nchunks=nchunks, chunk_size=chunk_size,
                      nmax_delay=nlag_max, nmin_delay=nlag_min, nstep_delay=nlag_step,
                      method=method,
                      nrel_points=list(-np.arange(nhist)))
    elif method == "2d":
        fte = tt.compute_te_2d
        params = dict(nchunks=nchunks, chunk_size=chunk_size,
                      nmax_delay=nlag_max, nmin_delay=nlag_min, nstep_delay=nlag_step)
    elif method == "full":
        fte = tt.compute_te_full
        params = dict(nchunks=nchunks, chunk_size=chunk_size,
                      nmax_delay=nlag_max, nmin_delay=nlag_min, nstep_delay=nlag_step)
    
    
    te_true, tlag = compute_te_true(fte, v_set, ntrue, nadd, params)
    te_surr, tlag = compute_te_surr(fte, v_set, nsurr, nadd, params)
    
    if fout is None:
        fout = "te_%d%02d.pkl"%(cid, wid)
    
    with open(fout, "wb") as fp:
        
        info = {"cid": cid, "wid": wid, "ntrue": ntrue, "nsurr": nsurr,
                "tw": tw, "tadd": tadd,
                "seed": seed, "dir": ("0->1", "1->0")}
        info.update(params)
        
        pkl.dump({
            "info": info,
            "tlag": tlag,
            "te": te_true,
            "te_surr": te_surr
        }, fp)
        
        
def compute_te_true(fte, v_set, ntrue, nadd, params):
    seed_set = np.random.randint(10, int(1e8), ntrue)
    f = partial(_compute_te_true, fte=fte, v_set=v_set, nadd=nadd, **params)
    te_results = utils.par_func(f, seed_set, num_process, desc="TE_true")
    return unzip_te_result(te_results)


def compute_te_surr(fte, v_set, nsurr, nadd, params):
    seed_set = np.random.randint(10, int(1e8), nsurr)
    f = partial(_compute_te_surr, fte=fte, v_set=v_set, nadd=nadd, **params)
    te_results = utils.par_func(f, seed_set, 1, desc="TE_surr")
    return unzip_te_result(te_results)
        
        
def _compute_te_true(seed, fte=None, v_set=None, nadd=None,
                     nchunks=0, chunk_size=0, **config):
    
    np.random.seed(seed)
    v_set_sub = tt.sample_true(v_set,
                               nadd=nadd,
                               nchunks=nchunks, chunk_size=chunk_size,
                               nmax_delay=config["nmax_delay"])
    
    te, nlag = fte(v_set_sub, **config)
    tlag = nlag/srate*1e3
    
    return te, tlag


def _compute_te_surr(seed, fte=None, v_set=None, nadd=None, 
                     nchunks=0, chunk_size=0, **config):
    
    np.random.seed(seed)        
    v_set_sub = tt.sample_surrogate(v_set,
                               nadd=nadd,
                               nchunks=nchunks, chunk_size=chunk_size,
                               nmax_delay=config["nmax_delay"],
                               warp_range=(0.8, 1.2))
    
    te, nlag = fte(v_set_sub, **config)
    tlag = nlag/srate*1e3
    
    return te, tlag


def unzip_te_result(te_results):
    tlag = te_results[0][1]
    te_set = np.array([te[0] for te in te_results])
    return te_set, tlag


if __name__ == "__main__":
    main(**vars(build_args().parse_args()))