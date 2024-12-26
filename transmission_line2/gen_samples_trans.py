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

"""
nums: [Ne, Ni, Nt]
    Nt is the number of tranmitting neurons. The first Nt neurons are belonged to Fast population,
    and the last Nt neurons in Slow population. The number of receiving neurons (listner) Nr == Nt.
    The range is
        (    0,        Ne+Ni): Fast pop
        ( Ne+Ni, 2*(Ne+Ni)=N): Slow pop
        (     N,        N+Nr): Receiver in Fast pop
        (  N+Nr,      N+2*Nr): Receiver in Slow pop
        (N+2*Nr,      N+3*Nr): Transmitter to Reciever in Fast pop (one-to-one match)
        (N+3*Nr,      N+4*Nr): Transmitter to Reciever in Slow pop (one-to-one match)
    Thus, the number of entire cells in population equals 2*(Ne+Ni) + 4*Nt
"""

nums = [800, 200, 200]
# ratio_set = [0.005, 0.01, 0.05]
# ratio_set = [0.008, 0.02, 0.1]
ratio_set = [0.008]
target_cid = [4, 7]

fname_repr_info = "../dynamics_clustering/data/cluster_repr_points.pkl"


def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir_origin", required=True, type=str)
    parser.add_argument("--fdir_trans", required=True, type=str)
    parser.add_argument("--ntrials", default=10, type=int)
    # parser.add_argument("--nrepeat", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--trans_only", default=False, action="store_true")
    parser.add_argument("--origin_only", default=False, action="store_true")
    parser.add_argument("--num_trans", default=200, type=int)
    return parser


def main(fdir_origin=None, fdir_trans=None, ntrials=10, num_trans=200, seed=42, trans_only=False, origin_only=False):
    # export template params
    if not trans_only:
        export_template_params(fdir_origin, ntrials, seed)
    else:
        ntrials_origin = read_num_trial(fdir_origin)
        if ntrials_origin != ntrials:
            print("ntrials in origin is %d, change ntrials from %d to %d"%(ntrials_origin, ntrials, ntrials_origin))
            ntrials = ntrials_origin
    # export trans params
    nums[2] = num_trans
    if not origin_only:
        export_trans_params(fdir_origin, fdir_trans, ntrials)
    

def export_trans_params(fdir_origin, fdir_trans, ntrials):
    origin_params_set = read_template_samples(fdir_origin)
    params = []
    for r in ratio_set:        
        for p in origin_params_set:
            params.append(convert_template_params(p, r))
    
    controls = OrderedDict(ratio_set=ratio_set, cluster_id=target_cid)
    write_control_trans(fdir_trans, controls, ntrials)
    write_params_trans(fdir_trans, params)


def export_template_params(fdir_origin, ntrials, seed):
    with open(fname_repr_info, "rb") as fp:
        repr_info = pkl.load(fp)
    print("cluster ID: ", repr_info["cluster_id"])
    
    np.random.seed(seed)
    params_tot, id_tot = collect_params(repr_info, ntrials)
    
    # write params
    write_control_origin(target_cid, ntrials, fdir_origin)
    write_params_origin(fdir_origin, params_tot)
    write_selected_points(fdir_origin, seed, id_tot)
    
    
def read_num_trial(fdir_origin):
    with open(os.path.join(fdir_origin, "control_params.txt"), "r") as fp:
        return  int(fp.readline().split(",")[-2])


def collect_params(repr_info, ntrials):
    id_tot = []
    params_tot = []
    
    for cid in target_cid:
        nc = cid - 1
        
        # seed_set = np.random.randint(low=0, high=100000, size=nsamples_for_each)
        id_sub = [repr_info["repr_idx"][nc]] * ntrials
        param = repr_info["repr_params"][nc]
        
        params_sub = []
        for _ in range(ntrials):
            seed = np.random.randint(low=0, high=100000)
            params_sub.append([seed] + convert_ind2params(*param))
        
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
        fp.write("%d,%d,\ncluster_id:"%(len(cluster_id_set), nsamples_for_each))
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
        
    
def convert_template_params(params, ratio, poisson_fr=0.9):
    
    keys = ("EF", "IF", "ES", "IS", "RF", "RS")
    
    seed = params[0]
    wE = [params[1], params[3]]
    wI = [params[2], params[4]]
    pE = [params[5], params[7],  params[13], params[15]]
    pI = [params[9], params[11], params[17], params[19]]
    nu = params[21:23]
    
    prob = dict(
        EF=(pE[0], pE[0], pE[1], pE[1], pE[0], pE[1]),
        IF=(pI[0], pI[0], pI[1], pI[1], pI[0], pI[1]),
        ES=(pE[2], pE[2], pE[3], pE[3], pE[2], pE[3]),
        IS=(pI[2], pI[2], pI[3], pI[3], pI[2], pI[3]),
        RF=(    0,     0,     0,     0,     0,     0),
        RS=(    0,     0,     0,     0,     0,     0)
    )
    
    weight = dict(
        EF=(wE[0], wE[0], wE[0], wE[0], wE[0], wE[0]),
        IF=(wI[0], wI[0], wI[0], wI[0], wI[0], wI[0]),
        ES=(wE[1], wE[1], wE[1], wE[1], wE[1], wE[1]),
        IS=(wI[1], wI[1], wI[1], wI[1], wI[1], wI[1]),
        RF=(    0,     0,     0,     0,     0,     0),
        RS=(    0,     0,     0,     0,     0,     0)
    )
    
    weight_tr = [ratio, ratio]
    
    params = [seed] + nums[:]
    for k in keys:
        params.extend(prob[k])
    for k in keys:
        params.extend(weight[k])
    
    params.extend(weight_tr)
    params.append(poisson_fr)
    params.extend(nu)
    
    return params
    

def read_template_samples(fdir_origin):
    
    params_set = []
    with open(os.path.join(fdir_origin, "params_to_run.txt"), "r") as fp:
        line = fp.readline()
        line = fp.readline()
        while line:
            params_set.append([])
            for n, x in enumerate(line.split(",")[:-1]):
                if n == 0:
                    params_set[-1].append(int(x))
                else:
                    params_set[-1].append(float(x))

            line = fp.readline()
    
    return params_set        


def read_controls(fdir_origin):
    with open(os.path.join(fdir_origin, "control_params.txt"), "r") as fp:
        line = fp.readline()
        num, num_itr = [int(x) for x in line.split(",")[:-1]]
        
        line = fp.readline()
        cid_set = [int(float(x)) for x in line.split(":")[1].split(",")[:-1]]
        
    return num, num_itr, cid_set
        
        
def write_control_trans(fdir_out, controls, num_itr):
    keys = list(controls.keys())
    
    fname = os.path.join(fdir_out, "control_params.txt")
    if os.path.exists(fname):
        raise ValueError("file %s already exists"%(fname))
    
    with open(fname, "w") as fp:
        for k in keys:
            fp.write("%d,"%(len(controls[k])))
        fp.write("%d,\n"%(num_itr))
        
        for k in keys:
            fp.write("%s:"%(k))
            for val in controls[k]:
                fp.write("%f,"%(val))
            fp.write("\n")
            
    
def write_params_trans(fdir_out, params):
    fname_out = os.path.join(fdir_out, "params_to_run.txt")
    print("Write parametes to %s"%(fname_out))
    with open(fname_out, "w") as fp:
        fp.write("%d\n"%(len(params)))
        for pset in params:
            for n, p in enumerate(pset):
                if n < 4:
                    fp.write("%ld,"%(p))
                else:
                    fp.write("%f,"%(p))
            fp.write("\n")
            
            
if __name__=="__main__":
    main(**vars(build_arg_parse().parse_args()))