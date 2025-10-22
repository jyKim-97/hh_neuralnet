import numpy as np
import os
# import subprocess
# import xarray as xa
from tqdm import tqdm
from collections import OrderedDict
import argparse

import pickle as pkl


""" Default params """
a_set = [9.4, 4.5]
b_set = [3, 2.5]
c = 0.01
d_set = [14142.14, 15450.58]
plim = [[0.051, 0.234],
        [0.028, 0.105]]

# TODO: Currently, the delay is fixed in the entire populations. 
# But, ideally, this have to be heterogeneous within the population 

# TODO: Also, the wratio should also be differ depending on the structures, 
# because each landmark has different connection probabilities, which cannot compromise the losed connection for receiver neuron.
# This would be different depending on their cluster ID
# you may need to consider the firing rate distribution comparison

"""
This code is for computing delayed spike transmission for the populations with single frequency 
nums: [Ne, Ni]
    Nt is the number of tranmitting neurons. The first Nt neurons are belonged to Fast population,
    and the last Nt neurons in Slow population. The number of receiving neurons (listner) Nr == Nt.
    The range is
        (    0,        Ne+Ni): Fast pop
        ( Ne+Ni, 2*(Ne+Ni)=N): Slow pop
    Thus, the number of entire cells in population equals 2*(Ne+Ni) + 4*Nt
"""

nums = [800, 200]
# target_cid = [4, 7]
# target_cid = [1, 2, 3, 4, 5, 6, 7]
target_cid = []
# ratio_set = [5, 10, 50]
ratio = 10
# tdelay_set = [5, 10, 20, 30]
tdelay_set = np.arange(0, 31, 2)
# tdelay_set = [10, 20, 30]

# fname_repr_info = "../dynamics_clustering/data/cluster_repr_points.pkl"
fname_repr_info = "../dynamics_clustering/data/cluster_repr_points"


def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir", required=True, type=str)
    parser.add_argument("--ntrials", default=10, type=int)
    parser.add_argument("--wratio", default=10, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--ptype", choices=("mfast", "mslow"), required=True, type=str)
    return parser


def main(fdir=None, wratio=10, ntrials=10, seed=42, ptype="mfast"):
    # export template params
    
    global ratio
    ratio = wratio
    
    global fname_repr_info
    fname_repr_info += "_%s.pkl"%(ptype)
    
    global a_set, b_set, d_set, plim
    nid = 0 if ptype == "mfast" else 1
    a_set[1-nid] = a_set[nid]
    b_set[1-nid] = b_set[nid]
    d_set[1-nid] = d_set[nid]
    plim[1-nid]  = plim[nid]
    
    export_template_params(fdir, ntrials, seed)


def export_template_params(fdir_origin, ntrials, seed):
    with open(fname_repr_info, "rb") as fp:
        repr_info = pkl.load(fp)
    print("cluster ID: ", repr_info["cluster_id"])
    
    global target_cid
    if len(target_cid) == 0:
        target_cid = np.arange(1, (repr_info["cluster_id"] > 0).sum() + 1)
        print("Target cid: ", target_cid)
    
    np.random.seed(seed)
    params_tot, id_tot = collect_params(repr_info, ntrials)
    
    # write params
    write_control_origin(target_cid, ntrials, fdir_origin)
    write_params_origin(fdir_origin, params_tot)
    write_selected_points(fdir_origin, seed, id_tot)


def collect_params(repr_info, ntrials):
    id_tot = []
    params_tot = []
    
    for td in tdelay_set:
    # for r in ratio_set:
        for cid in target_cid:
            nc = cid - 1
            
            # seed_set = np.random.randint(low=0, high=100000, size=nsamples_for_each)
            id_sub = [repr_info["repr_idx"][nc]] * ntrials
            param = repr_info["repr_params"][nc]
            
            params_sub = []
            for _ in range(ntrials):
                seed = np.random.randint(low=0, high=100000)
                params_sub.append([seed] + convert_ind2params(*param) + [ratio, td])
            
            id_tot.extend(id_sub)
            params_tot.extend(params_sub)
            
    return params_tot, id_tot

        
def convert_ind2params(alpha, beta, rank, wx):
    # inter-population projection
    w_slow, w_fast = 1, 1
    if wx > 0:
        w_fast = 1 - wx
    else:
        w_slow = 1 + wx

    pe_fs = [plim[i][0]*(1-rank) + plim[i][1]*rank for i in range(2)]
    pi_fs = [b_set[i] * pe_fs[i] for i in range(2)]
    
    we_fs = [c * np.sqrt(0.01) / np.sqrt(pe) for pe in pe_fs]
    wi_fs = [a_set[n] * we_fs[n] for n in range(2)]
    
    param_sub = [
        we_fs[0], wi_fs[0], we_fs[1], wi_fs[1],
        pe_fs[0], pe_fs[0], w_fast*alpha*pe_fs[0], w_fast*alpha*pe_fs[0],
        pi_fs[0], pi_fs[0], w_fast*beta*pi_fs[0],  w_fast*beta*pi_fs[0],
        w_slow*alpha*pe_fs[1], w_slow*alpha*pe_fs[1], pe_fs[1],  pe_fs[1],
        w_slow*beta*pi_fs[1],  w_slow*beta*pi_fs[1],  pi_fs[1],  pi_fs[1], 
        d_set[0]*np.sqrt(pe_fs[0]),
        d_set[1]*np.sqrt(pe_fs[1])
    ]

    return param_sub


def write_control_origin(cluster_id_set, nsamples_for_each, fdir_out):
    fname = os.path.join(fdir_out, "control_params.txt")
    if os.path.exists(fname):
        raise ValueError("file %s already exists"%(fname))
    
    with open(fname, "w") as fp:
        fp.write("%d,%d,%d,\ntdelay_set:"%(len(tdelay_set), len(cluster_id_set), nsamples_for_each))
        for td in tdelay_set:
            fp.write("%f,"%(td))
        fp.write("\n")
        
        fp.write("cluster_id:")
        for n in cluster_id_set:
            fp.write("%.1f,"%(n))
        fp.write("\n")

        
def write_selected_points(fdir_out, seed, id_selected):
    fname = os.path.join(fdir_out, "selected_points.txt")
    with open(fname, "w") as fp:
        fp.write("Cluster ID, *index, init_seed=%d\n"%(seed))
        
        for idc in id_selected:
            for n in idc:
                fp.write("%d,"%(n))
            fp.write("\n")
            
            
def write_params_origin(fdir_out, params):
    fname_out = os.path.join(fdir_out, "params_to_run.txt")
    print("Write parametes to %s"%(fname_out))
    with open(fname_out, "w") as fp:
        fp.write("%d\n"%(len(params)))
        for pset in params:
            for i, p in enumerate(pset):
                if i == 0:
                    fp.write("%ld,"%(p))
                else:
                    fp.write("%f,"%(p))
            fp.write("\n")
        
    
# def convert_template_params(params, ratio, poisson_fr=0.9):
    
#     keys = ("EF", "IF", "ES", "IS", "RF", "RS")
    
#     seed = params[0]
#     wE = [params[1], params[3]]
#     wI = [params[2], params[4]]
#     pE = [params[5], params[7],  params[13], params[15]]
#     pI = [params[9], params[11], params[17], params[19]]
#     nu = params[21:23]
    
#     prob = dict(
#         EF=(pE[0], pE[0], pE[1], pE[1], pE[0], pE[1]),
#         IF=(pI[0], pI[0], pI[1], pI[1], pI[0], pI[1]),
#         ES=(pE[2], pE[2], pE[3], pE[3], pE[2], pE[3]),
#         IS=(pI[2], pI[2], pI[3], pI[3], pI[2], pI[3]),
#         RF=(    0,     0,     0,     0,     0,     0),
#         RS=(    0,     0,     0,     0,     0,     0)
#     )
    
#     weight = dict(
#         EF=(wE[0], wE[0], wE[0], wE[0], wE[0], wE[0]),
#         IF=(wI[0], wI[0], wI[0], wI[0], wI[0], wI[0]),
#         ES=(wE[1], wE[1], wE[1], wE[1], wE[1], wE[1]),
#         IS=(wI[1], wI[1], wI[1], wI[1], wI[1], wI[1]),
#         RF=(    0,     0,     0,     0,     0,     0),
#         RS=(    0,     0,     0,     0,     0,     0)
#     )
    
#     weight_tr = [ratio, ratio]
    
#     params = [seed] + nums[:]
#     for k in keys:
#         params.extend(prob[k])
#     for k in keys:
#         params.extend(weight[k])
    
#     params.extend(weight_tr)
#     params.append(poisson_fr)
#     params.extend(nu)
    
#     return params
 

def read_controls(fdir_origin):
    with open(os.path.join(fdir_origin, "control_params.txt"), "r") as fp:
        line = fp.readline()
        num, num_itr = [int(x) for x in line.split(",")[:-1]]
        
        line = fp.readline()
        cid_set = [int(float(x)) for x in line.split(":")[1].split(",")[:-1]]
        
    return num, num_itr, cid_set
            
            
if __name__=="__main__":
    main(**vars(build_arg_parse().parse_args()))
