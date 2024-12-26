import numpy as np
import os
# import subprocess
import xarray as xa
from tqdm import tqdm
import pickle as pkl


"""
Input: .nc dataset file

- pbest: 

Output: ./params_to_run.txt and ./selected_cluster_info

"""

# repr_points = [[0, 0, 2, 0]]

# multi-frequency oscillation
# repr_points = ((4,  1,  0, 1),  # 1
#                (6, 10,  0,  5), # 2
#                (9,  7,  0, 1),  # 3
#                (8, 4, 1, 8),    # 4
#                (10,  9,  2, 1), # 5
#                (6, 9, 1, 4),    # 6
#                (5, 12,  2,  4), # 7
#                (6, 2, 2, 8))    # 8

# mono-frequency oscillation
# FAST
# repr_points = ((6, 1, 0, 3),
#                (2, 10, 0, 3),
#                (11, 10, 0, 1),
#                (4, 2, 1, 2),
#                (5, 12, 1, 1),
#                (6, 11, 2, 3),
#                (4, 2, 2, 3),
#                (11, 11, 2, 0))
# SLOW
# repr_points = ((2, 4, 0, 2),
#                (10, 10, 0, 2),
#                (10,  7, 1, 1),
#                (12,  3, 1, 1),
#                ( 3,  7, 1, 2),
#                ( 5,  5, 2, 3),
#                (11, 12, 2, 3))

## NEW version (240903)
# Multi-freq oscillations
# repr_points = ((0,  0, 0,  0), # for the comparison
#                (9,  7, 0, 12),
#                (7, 11, 0,  4),
#                (9, 10, 2, 13),
#                (8,  4, 1,  7),
#                (6,  7, 2,  4),
#                (6, 12, 2,  4),
#                (4,  2, 2,  7))

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

# a_set = [4.5, 4.5]
# b_set = [2.5, 2.5]
# c = 0.01
# d_set = [15450.58, 15450.58]
# plim = [[0.028, 0.105],
#         [0.028, 0.105]]


def main(fname_repr_info=None, nsamples_for_each=200, fdir_out="./data", init_seed=42):
    np.random.seed(init_seed)
    
    with open(fname_repr_info, "rb") as fp:
        repr_info = pkl.load(fp)
    print("cluster ID: ", repr_info["cluster_id"])
    
    # select data set and export parameters
    id_tot = []
    params_tot = []
    for nc in range(len(repr_info["cluster_id"])):
        id_sub = [repr_info["repr_idx"][nc]] * nsamples_for_each
        param = repr_info["repr_params"][nc]
        params_sub = [cvt_ind2params_sub(*param)] * nsamples_for_each
        # params_sub = [repr_info["repr_params"][nc]] * nsamples_for_each
        
        id_tot.extend(id_sub)
        params_tot.extend(params_sub)
    
    # write parameters
    write_control_params(repr_info["cluster_id"], nsamples_for_each, fdir_out)
    write_params(fdir_out, params_tot)
    
    # write sample point info
    write_selected_points(fdir_out, init_seed, id_tot)
    




# def main(fname_cinfo=None, nsamples_for_each=100,
#          fdir_out="./data", init_seed=None):

#     np.random.seed(init_seed)
    
#     cid_dataset = xa.load_dataset(fname_cinfo)
#     assert len(repr_points)-1 == len(cid_dataset.attrs["id_set"])
    
    
#     # select data set and export parameters
#     id_tot = []
#     params_tot = []
#     for nc in len(repr_points):
        
        
    
#     # for cid in tqdm(cid_dataset.attrs["id_set"]):
        
#         # use repr
#         # nc = int(cid-1)
#         # print(np.array(repr_points[nc]))
#         # a, b, c, d = repr_points[nc]
        
#         locs_ = [repr_points[nc] for _ in range(nsamples_for_each)]
#         params = cvt_ind2params(locs_, cid_dataset.coords)
#         params_tot.extend(params)
        
#         for _ in range(nsamples_for_each):
#             id_tot.extend([[cid] + list(repr_points[nc])])
#             # params_tot.extend(params)
    
#     # write parameters
#     write_control_params(len(cid_dataset.attrs["id_set"]), nsamples_for_each, fdir_out)
#     write_params(fdir_out, params_tot)
    
#     # write sample point info
#     write_selected_points(fdir_out, init_seed, id_tot)
    
    
    # run
    # if run_c:
    #     cmd = "mpirun -np %d ./main.out -n %d -t %d --fparam %s --fdir_out %s > exec.txt"%(mpi_num_core, len(params), tmax, fname_out, fdir_out)
    #     print("execute command: %s"%(cmd))
    #     res = subprocess.run(cmd, shell=True)    
        
# def pick_cluster_points(fname=None, nbest=10, nsamples_for_each=100):
#     cluster_info = load_best_cluster_points(fname)
#     K = len(cluster_info["loc_points"])

#     # NOTE: need to modify the file includes the exact value/parameters of each points
#     # This is the temporal section: hard fix parameters to reconstruct parameters
    
#     selected_id = [] # 
#     samples = [] # [cluster id, ...]
#     for n in range(K):
#         sval = cluster_info["sval_points"][n][:nbest+1]
#         p = (np.array(sval) - sval[-1]) # prob of last one will become 0
#         p /= np.sum(p)

#         nid_select = np.random.choice(nbest+1, size=nsamples_for_each, p=p)
#         # ADD rank id
#         nrank_set =  [cluster_info["loc_points"][n][i][2] for i in nid_select]
#         selected_id.extend([[n+1, nid, nrank_set[i]] for i, nid in enumerate(nid_select)])
#         samples.extend([cluster_info["loc_points"][n][i] for i in nid_select])

#     params = cvt_ind2params(samples)

#     return params, selected_id, K

def pick_cluster_points(cid_dataset, target_id, pbest=10, nsamples=0):
    cluster_id = cid_dataset.cluster_id.data.flatten()
    
    is_target = cluster_id == target_id
    target_id = np.where(is_target)[0]
    
    sval_target = cid_dataset.sval.data.flatten()[is_target]
    sth = np.percentile(sval_target, 100-pbest)
    
    is_upper  = sval_target > sth
    sval_best = sval_target[is_upper]
    target_id_best = target_id[is_upper]
    
    # random selection
    p = np.exp(sval_best) / np.sum(np.exp(sval_best))
    id_selected = np.random.choice(len(sval_best), size=nsamples, p=p, replace=True)

    pick_id_f = target_id_best[id_selected]
    
    # convert to index number
    # sz = list(cid_dataset.coords.dims.values())
    sz = cid_dataset.cluster_id.shape
    pick_id = [np.unravel_index(n, sz) for n in pick_id_f]
    
    return pick_id


def write_selected_points(fdir_out, seed, id_selected):
    fname = os.path.join(fdir_out, "selected_points.txt")
    with open(fname, "w") as fp:
        fp.write("Cluster ID, *index, init_seed=%d\n"%(seed))
        
        for idc in id_selected:
            for n in idc:
                fp.write("%d,"%(n))
            fp.write("\n")
            

# def load_best_cluster_points(fname=None):
#     import pickle as pkl

#     if fname is None:
#         fname = "../three_pop_mpi/clustering/data/cluster_repr_points_rank3.pkl"
    
#     if not os.path.exists(fname):
#         raise ValueError("file %s is not exist"%(fname))
    
#     with open(fname, "rb") as fp:
#         return pkl.load(fp) # key: loc_points, sval_points


# def write_cluster_info(cluster_info, fdir_out, seed):
#     fname = os.path.join(fdir_out, "picked_cluster.txt")
#     print("Cluster info written to %s"%(fname))
#     with open(fname, "w") as fp:
#         # Write rank id
#         fp.write("cluster_id(starts from 1), nth best point, nrank, init_seed=%d\n"%(seed))
#         for cid in cluster_info:
#             fp.write("%d,%d,%d,\n"%(cid[0], cid[1], cid[2]))

    
def write_params(fdir_out, params):
    fname_out = os.path.join(fdir_out, "params_to_run.txt")
    print("Write parametes to %s"%(fname_out))
    with open(fname_out, "w") as fp:
        fp.write("%d\n"%(len(params)))
        for pset in params:
            for p in pset:
                fp.write("%f,"%(p))
            fp.write("\n")


# def write_control_params(K, nsamples_for_each, fdir_out):
#     fname = os.path.join(fdir_out, "control_params.txt")
#     with open(fname, "w") as fp:
#         fp.write("%d,%d,\ncluster_id:"%(K, nsamples_for_each))
#         for n in range(K):
#             fp.write("%.1f,"%(n))
#         fp.write("\n")

def write_control_params(cluster_id_set, nsamples_for_each, fdir_out):
    fname = os.path.join(fdir_out, "control_params.txt")
    with open(fname, "w") as fp:
        fp.write("%d,%d,\ncluster_id:"%(len(cluster_id_set), nsamples_for_each))
        for n in cluster_id_set:
            fp.write("%.1f,"%(n))
        fp.write("\n")
        

def cvt_ind2params_sub(alpha, beta, rank, wx):
    # NOTE: This is the temporal function that hard fix parameters to reconstruct parameters

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
        

# def cvt_ind2params(sample_inds, coords):
#     # NOTE: This is the temporal function that hard fix parameters to reconstruct parameters

#     # inter-population projection
#     alpha_set = coords["alpha"].data #np.linspace(0, 2, 15)
#     beta_set  = coords["beta"].data # np.linspace(0, 1, 15)
#     rank_set  = coords["rank"].data # [0, 0.5, 1]
#     w_set = coords["w"].data # np.linspace(0.1, 1, 7)

#     params = []
#     for sample_id in sample_inds: # nrow: alpha, ncol: beta, nrank, nw
#         # NOTE check parameter order in 'main.c'
#         nrow, ncol, nrank, nw = sample_id

#         alpha = alpha_set[nrow]
#         beta  = beta_set[ncol]
#         rank  = rank_set[nrank]
        
#         wx = w_set[nw]
#         w_slow, w_fast = 1, 1
#         if wx > 0:
#             w_fast = 1 - wx
#         else:
#             w_slow = 1 + wx

#         pe_fs = [plim[i][0]*(1-rank) + plim[i][1]*rank for i in range(2)]
#         pi_fs = [b_set[i] * pe_fs[i] for i in range(2)]
        
#         we_fs = [c * np.sqrt(0.01) / np.sqrt(pe) for pe in pe_fs]
#         wi_fs = [a_set[n] * we_fs[n] for n in range(2)]
        
#         param_sub = [
#             we_fs[0], wi_fs[0], we_fs[1], wi_fs[1],
#             pe_fs[0], pe_fs[0], w_fast*alpha*pe_fs[0], w_fast*alpha*pe_fs[0],
#             pi_fs[0], pi_fs[0], w_fast*beta*pi_fs[0],  w_fast*beta*pi_fs[0],
#             w_slow*alpha*pe_fs[1], w_slow*alpha*pe_fs[1], pe_fs[1],  pe_fs[1],
#             w_slow*beta*pi_fs[1],  w_slow*beta*pi_fs[1],  pi_fs[1],  pi_fs[1], 
#             d_set[0]*np.sqrt(pe_fs[0]),
#             d_set[1]*np.sqrt(pe_fs[1])
#         ]
        
#         params.append(param_sub)

#     return params


if __name__ == "__main__":
    
    # main(fname_cinfo="../dynamics_clustering/data/cluster_id_sub.nc",
    #      pbest=10, nsamples_for_each=300,
    #      fdir_out="./data", init_seed=2000)
    # main(fname_cinfo="../dynamics_clustering/data/cluster_id_sub.nc",
    #      pbest=10, nsamples_for_each=500,
    #      fdir_out="./data2", init_seed=42)
    # main(fname_cinfo="../dynamics_clustering/data/cluster_id_mfast.nc",
    #      pbest=10, nsamples_for_each=600,
    #      fdir_out="./data_mfast", init_seed=2000)
    # main(fname_cinfo="../dynamics_clustering/data/cluster_id_mfast.nc",
    #      pbest=10, nsamples_for_each=600,
    #      fdir_out="./data_mfast2", init_seed=42)
    # main(fname_cinfo="../dynamics_clustering/data/cluster_id_sub_mslow.nc",
    #      pbest=10, nsamples_for_each=600,
    #      fdir_out="./data_mslow2", init_seed=2000)    
    # main(fname_cinfo="../dynamics_clustering/data/cluster_id_sub_mslow.nc",
    #      pbest=10, nsamples_for_each=600,
    #      fdir_out="./data_mslow2", init_seed=42)
    
    # main(fname_cinfo="../dynamics_clustering/data/cluster_id_sub.nc",
    #      nsamples_for_each=200,
    #      fdir_out="./data_multi", init_seed=42)
    
    # Directly import sample ID
    main(fname_repr_info="../dynamics_clustering/data/cluster_repr_points.pkl",
         nsamples_for_each=600,
         fdir_out="./data_multi2", init_seed=2000) # 42 (200), 2000 (600)