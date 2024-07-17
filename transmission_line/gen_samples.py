import numpy as np
import os
# import subprocess
import xarray as xa
from tqdm import tqdm
from collections import OrderedDict
import argparse


"""
Input: .nc dataset file

- pbest: 

Output: ./params_to_run.txt and ./selected_cluster_info

"""

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdir", required=True)
    return parser


# repr_points = [[0, 0, 2, 0]]

# multi-frequency oscillation
repr_points = ((4,  1,  0, 1),
               (6, 10,  0,  5),
               (9,  7,  0, 1),
               (8, 4, 1, 8),
               (10,  9,  2, 1),
               (6, 9, 1, 4),
               (5, 12,  2,  4),
               (6, 2, 2, 8))

# mono-frequency oscillation
# repr_points = ((6, 1, 0, 3),
#                (2, 10, 0, 3),
#                (11, 10, 0, 1),
#                (4, 2, 1, 2),
#                (5, 12, 1, 1),
#                (6, 11, 2, 3),
#                (4, 2, 2, 3),
#                (11, 11, 2, 0))

# intra-population projection
a_set = [9.4, 4.5]
b_set = [3, 2.5]
c = 0.01
d_set = [14142.14, 15450.58]
plim = [[0.051, 0.234],
        [0.028, 0.105]]

# mono-frequency population
# a_set = [9.4, 9.4]
# b_set = [3, 3]
# c = 0.01
# d_set = [14142.14, 14142.14]
# plim = [[0.051, 0.234],
#         [0.051, 0.234]]


alpha_set = []
beta_set = []
echelon_set = []
w_set = []


def main(fname_cinfo=None, nsamples_for_each=100, fdir="./data"):
    """
    Input
    - pbest: 상위 몇퍼센트 데이터?
    """
    
    if fname_cinfo is None:
        fname_cinfo = "../dynamics_clustering/data/cluster_id_sub.nc"

    cid_dataset = xa.load_dataset(fname_cinfo)
    read_param_range(cid_dataset)
    
    # select data set and export parameters
    cid_set = [4, 5, 8]
    # ratio_set = [1, 0.75, 0.5, 0.25, 0.1]
    # ratio_set = [0.1, 0.05, 0.01, 0]
    ratio_set = [0, 0.25, 0.5, 0.75, 1]
    num_itr = 10
    
    params_tot = []
    for cid in cid_set:
        loc = repr_points[int(cid-1)]
        for ratio in ratio_set:
            params = set_params(loc, ratio=ratio)
            params_tot.extend([params]*num_itr)
    
    # cid = 8
    # loc = repr_points[int(cid-1)]
    # ratio = 2
    # num_itr = 1
    
    # ratio_set = [ratio]
    # cid_set = [cid]
    # params_tot = [set_params(loc, ratio=ratio)]
    
    controls = OrderedDict(cluster_id=cid_set,
                           ratio_set=ratio_set)
    
    write_params(fdir, params_tot)
    write_control_params(fdir, controls, num_itr)
    # write_control_params(fdir, [8], 2)
    
    
def read_param_range(cid_dataset):
    global alpha_set, beta_set, echelon_set, w_set
    
    alpha_set   = cid_dataset.coords["alpha"].data
    beta_set    = cid_dataset.coords["beta"].data
    echelon_set = cid_dataset.coords["rank"].data
    w_set       = cid_dataset.coords["w"].data
    
    
# TODO: NEED TO DISCUSS about inter-inhibitory connection in transmission line
def set_params(locs, ratio=1):
    alpha = alpha_set[locs[0]]
    beta = beta_set[locs[1]]
    echelon = echelon_set[locs[2]]
    w_fast, w_slow = transform_w(w_set[locs[3]])
    
    # intra-population connection
    pE = [pl[0]*(1-echelon) + pl[1]*echelon for pl in plim]
    pI = [b * p for b, p in zip(b_set, pE)]
    
    wE = [c * np.sqrt(0.01) / np.sqrt(p) for p in pE]
    wI = [a * w for a, w in zip(a_set, wE)]
    
    '''
    Types
    EF, IF, TF, RF, ES, IS, TS, RS (T: transmisson, R: receiver)
    '''
    keys = ("EF", "IF", "TF", "RF", "ES", "IS", "TS", "RS")
    
    # number of cells
    NE, NI = 800, 200
    NTR = 20
    
    nums = dict(EF=NE, IF=NI, TF=NTR, RF=NTR,
                ES=NE, IS=NI, TS=NTR, RS=NTR)
    # nums = dict(EF=800, IF=200, TF=1, RF=1,
    #             ES=800, IS=200, TS=1, RS=1)
    
    wa = (w_fast*alpha, w_slow*alpha)
    wb = (w_fast*beta,  w_slow*beta)
    
    # adjust connection probability
    npop = NE
    ntot = NE + NTR*2
    
    pe_local, pe_ts = [], []
    for p in pE:
        p0 = np.round(p*npop/ntot, 3)
        n0 = p0 * ntot
        dn = p * npop - n0
        
        pe_local.append(p0)
        pe_ts.append(np.round(p0+dn/NTR, 4))
    
    # projection probability (region A -> region B)
    prob = dict(
        EF=(pe_local[0], pe_local[0], pe_local[0], pe_local[0], wa[0]*pE[0], wa[0]*pE[0], wa[0]*pE[0],           0),
        IF=(      pI[0],       pI[0],       pI[0],       pI[0], wb[0]*pI[0], wb[0]*pI[0], wb[0]*pI[0], wb[0]*pI[0]),
        TF=(   pe_ts[0],    pe_ts[0],           0,           0,           0,           0,           0,          -1),
        RF=(   pe_ts[0],    pe_ts[0],           0,           0,           0,           0,           0,           0),
        ES=(wa[1]*pE[1], wa[1]*pE[1], wa[1]*pE[1],           0, pe_local[1], pe_local[1], pe_local[1], pe_local[1]),
        IS=(wb[1]*pI[1], wb[1]*pI[1], wb[1]*pI[1], wb[1]*pI[1],       pI[1],       pI[1],       pI[1],       pI[1]),
        TS=(          0,           0,           0,          -1,    pe_ts[1],    pe_ts[1],           0,           0),
        RS=(          0,           0,           0,           0,    pe_ts[1],    pe_ts[1],           0,           0)
    )
    
    # prob = dict(
    #     EF=(      pE[0],       pE[0],       pE[0],       pE[0], wa[0]*pE[0], wa[0]*pE[0], wa[0]*pE[0],           0),
    #     IF=(      pI[0],       pI[0],       pI[0],       pI[0], wb[0]*pI[0], wb[0]*pI[0], wb[0]*pI[0], wb[0]*pI[0]),
    #     TF=(   pe_ts[0],    pe_ts[0],           0,           0,           0,           0,           0,          -1),
    #     RF=(   pe_ts[0],    pe_ts[0],           0,           0,           0,           0,           0,           0),
    #     ES=(wa[1]*pE[1], wa[1]*pE[1], wa[1]*pE[1],           0,       pE[1],       pE[1],       pE[1],       pE[1]),
    #     IS=(wb[1]*pI[1], wb[1]*pI[1], wb[1]*pI[1], wb[1]*pI[1],       pI[1],       pI[1],       pI[1],       pI[1]),
    #     TS=(          0,           0,           0,          -1,    pe_ts[1],    pe_ts[1],           0,           0),
    #     RS=(          0,           0,           0,           0,    pe_ts[1],    pe_ts[1],           0,           0)
    # )
    
    # connection strength (out)
    # wTR = [10*w for w in wE]
    wTR = [ratio*p*a*NE for p, a in zip(pE, wa)]
    
    weight = dict(
        EF=(wE[0], wE[0], wE[0],  wE[0], wE[0], wE[0], wE[0],      0),
        IF=(wI[0], wI[0], wI[0],  wI[0], wI[0], wI[0], wI[0],  wI[0]),
        TF=(wE[0], wE[0],     0,      0,     0,     0,     0, wTR[0]),
        RF=(wE[0], wE[0],     0,      0,     0,     0,     0,      0),
        ES=(wE[1], wE[1], wE[1],      0, wE[1], wE[1], wE[1],  wE[1]),
        IS=(wI[1], wI[1], wI[1],  wI[1], wI[1], wI[1], wI[1],  wI[1]),
        TS=(    0,     0,     0, wTR[1], wE[1], wE[1],     0,      0),
        RS=(    0,     0,     0,      0, wE[1], wE[1],     0,      0),
    )
    
    nu = [d*np.sqrt(p) for d, p in zip(d_set, pE)]
    
    # convert
    params = [nums[k] for k in keys]
    for k in keys:
        params.extend(prob[k])
    for k in keys:
        params.extend(weight[k])
    params.extend(nu)
    
    return params
    

def transform_w(w):
    if w > 0:
        w_fast = 1 - w
        w_slow = 1
    else:
        w_fast = 1
        w_slow = 1 - w
    return w_fast, w_slow
    

def write_selected_points(fdir_out, seed, id_selected):
    fname = os.path.join(fdir_out, "selected_points.txt")
    with open(fname, "w") as fp:
        fp.write("Cluster ID, *index, init_seed=%d\n"%(seed))
        
        for idc in id_selected:
            for n in idc:
                fp.write("%d,"%(n))
            fp.write("\n")
                    
    
def write_params(fdir_out, params):
    fname_out = os.path.join(fdir_out, "params_to_run.txt")
    print("Write parametes to %s"%(fname_out))
    with open(fname_out, "w") as fp:
        fp.write("%d\n"%(len(params)))
        for pset in params:
            for p in pset:
                fp.write("%f,"%(p))
            fp.write("\n")
            
            
def write_control_params(fdir_out, controls, num_itr):
    keys = list(controls.keys())
    
    fname = os.path.join(fdir_out, "control_params.txt")
    with open(fname, "w") as fp:
        for k in keys:
            fp.write("%d,"%(len(controls[k])))
        fp.write("%d,\n"%(num_itr))
        
        for k in keys:
            fp.write("%s:"%(k))
            for val in controls[k]:
                fp.write("%f,"%(val))
            fp.write("\n")


# def write_control_params(fdir_out, kset, nsamples_for_each):
#     fname = os.path.join(fdir_out, "control_params.txt")
#     with open(fname, "w") as fp:
#         fp.write("%d,%d,\ncluster_id:"%(len(kset), nsamples_for_each))
#         for k in kset:
#             fp.write("%.1f,"%(k))
#         fp.write("\n")


if __name__ == "__main__":
    # main(fname_cinfo="../dynamics_clustering/data/cluster_id_sub.nc",
    #      nsamples_for_each=100,
    #      fdir_out="./data_weak")
    
    main(**vars(build_parser().parse_args()))
    
    # main(fname_cinfo="../dynamics_clustering/data/cluster_id_sub.nc",
    #      pbest=10, nsamples_for_each=500,
    #      fdir_out="./data2", init_seed=42)
    
    # # main(fname_cinfo="../dynamics_clustering/data/cluster_id_mfast.nc",
    # #      pbest=10, nsamples_for_each=300,
    # #      fdir_out="./data_mfast", init_seed=2000)
