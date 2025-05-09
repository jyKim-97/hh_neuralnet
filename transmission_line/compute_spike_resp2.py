import numpy as np

import os
import sys
import argparse
import pickle as pkl
import xarray as xr
from numba import njit

from tqdm import tqdm
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhsignal
import hhtools

import utils

tmax = 10.5
dt = 0.01 * 1e-3
num_trans = 40
td_unique = np.arange(0, 31, 2)

nmax_sel = 5


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir", required=True, type=str)
    parser.add_argument("--t0", default=-1, type=float)
    parser.add_argument("--t1", default=40, type=float)
    parser.add_argument("--tstep", default=0.5, type=float)
    parser.add_argument("--fout", default="tmp.nc", type=str)
    parser.add_argument("--nc", default=0, type=int)
    parser.add_argument("--nw_set", default="[]", type=str)
    
    return parser


def main(fdir=None, t0=0, t1=40, tstep=0.1, fout=None, nc=0, nw_set=None):
    # check prameter
    if os.path.exists(fout):
        raise ValueError("File already exist")
    
    print("File will be saved as %s"%(fout))
    summary_obj = hhtools.SummaryLoader(fdir, load_only_control=True)
    winfo = utils.load_pickle(os.path.join(fdir, "osc_motif/motif_info_%d.pkl"%(nc)))["winfo"]
    tspk_edges = np.arange(t0, t1+tstep/2, tstep)
    
    nspk_edges = (tspk_edges / dt).astype(int)
    nw_set = [int(x) for x in nw_set[1:-1].split(",")]
    
    prob_resp2 = configure_spk_prob(summary_obj, winfo, nc, nw_set, nspk_edges)
    
    # configure xr datarray
    
    t = (tspk_edges[1:] + tspk_edges[:-1])/2
    dr = xr.DataArray(
        prob_resp2, 
        coords={"nw": nw_set,
                "td": td_unique,
                "type": [0, 1],
                "t1": t,
                "t2": t,
                "spk1": [0, 1],
                "spk2": [0, 1]},
        attrs=dict(
            fdir=summary_obj.fdir,
            nc=nc
        )
    )

    dr.to_netcdf(fout)
    print("Printed to %s"%(fout))
    

    
def configure_spk_prob(summary_obj, winfo, nc, nw_set, nspk_edges):
    
    max_trial = summary_obj.num_controls[0] * summary_obj.num_controls[2]
    prob_resp2 = np.zeros((len(nw_set), len(td_unique), 2, len(nspk_edges)-1, len(nspk_edges)-1, 2, 2), dtype=np.int32)
    
    for nt in tqdm(range(max_trial), ncols=100):
    # configure osc_idx
        osc_idx = np.zeros(int(tmax/dt), dtype=np.int8) - 1
        for i, nw in enumerate(nw_set):        
            nd_tmp = nt // summary_obj.num_controls[2]
            nt_tmp = nt %  summary_obj.num_controls[2]
            for nt_w, tl in winfo[nd_tmp][nw]:
                if nt_w != nt_tmp: continue
                t0 = tl[0]# - ta
                t1 = tl[1]# + ta
                n0 = int(t0/dt)
                n1 = int(t1/dt)
                assert np.all(osc_idx[n0:n1] == -1)
                osc_idx[n0:n1] = i
        
        # compute response
        detail = utils.load_detail(summary_obj, nc, nt, load_ntk=True)
        for n in range(num_trans):
            idt = detail["sel_tr"][0][n]
            idr = detail["sel_tr"][1][n]
            td  = detail["sel_td"][n]
            nd  = np.where(td_unique == td)[0]
            ntp = n // (num_trans//2)
            
            adj_out = detail["adj_out"][idr]
            nstep_t = detail["step_spk"][idt]
            nstep_r = detail["step_spk"][idr]
            
            adj_out_sub = [npost for npost in adj_out if npost//1000 == idr//1000]
            adj_out_sub = np.random.choice(adj_out_sub, size=min(len(adj_out_sub), nmax_sel), replace=False)
            
            for nstep in nstep_t:
                nw = osc_idx[nstep]        
                if nw == -1: continue
                
                nspk_r, _ = utils.align_spike_single(nstep_r, nstep, nspk_edges, nsearch_start=0)
                
                for npost in adj_out_sub:
                    # if npost//1000 != idr//1000: continue
                    
                    nstep_r2 = detail["step_spk"][npost]
                    nspk_r2, _  = utils.align_spike_single(nstep_r2, nstep, nspk_edges, nsearch_start=0)
                    pmat = construct_prob_mat(nspk_r, nspk_r2)
                    
                    prob_resp2[nw, nd, ntp] += pmat.astype(np.int32)
    
    return prob_resp2


@njit
def construct_prob_mat(nspk_r1, nspk_r2):
    T = len(nspk_r1)
    pmat = np.zeros((T,T,2,2))
    for i in range(T):
        for j in range(T):
            pmat[i, j, nspk_r1[i], nspk_r2[j]] += 1
            
    return pmat


if __name__=="__main__":
    main(**vars(build_parser().parse_args()))