import numpy as np
import os
import subprocess


def main(fname_cinfo=None, nbest=10, nsamples_for_each=100, run_c=False,
         fname_out="./params_to_run.txt", fdir_out="./data"):
    # get params
    params, cluster_info = pick_cluster_points(fname=fname_cinfo, nbest=nbest,
                                               nsamples_for_each=nsamples_for_each)

    # export
    print("Selected %d samples"%(len(params)))
    write_params(params, fname_out)
    write_cluster_info(cluster_info, fdir_out)

    # run
    if run_c:
        cmd = "mpirun -np 110 ./main.out -n %d --fparam %s --fdir_out %s > exec.txt"%(len(params), fname_out, fdir_out)
        print("execute command: %s"%(cmd))
        res = subprocess.run(cmd, shell=True)


def load_best_cluster_points(fname=None):
    import pickle as pkl

    if fname is None:
        fname = "../three_pop_mpi/clustering/data/cluster_repr_points_rank3.pkl"
    
    if not os.path.exists(fname):
        raise ValueError("file %s is not exist"%(fname))
    
    with open(fname, "rb") as fp:
        return pkl.load(fp) # key: loc_points, sval_points


def write_cluster_info(cluster_info, fdir_out):
    with open(os.path.join(fdir_out, "picked_cluster.txt"), "w") as fp:
        fp.write("cluster_id, nth best point")
        for cid in cluster_info:
            fp.write("%d,%d\n"%(cid[0], cid[1]))

    
def write_params(params, fname_out="./params_to_run.txt"):
    print("Write parametes to %s"%(fname_out))
    with open(fname_out, "w") as fp:
        for pset in params:
            for p in pset:
                fp.write("%f,"%(p))
            fp.write("\n")


def pick_cluster_points(fname=None, nbest=10, nsamples_for_each=100):
    cluster_info = load_best_cluster_points(fname)
    K = len(cluster_info["loc_points"])

    # NOTE: need to modify the file includes the exact value/parameters of each points
    # This is the temporal section: hard fix parameters to reconstruct parameters
    
    selected_id = [] # 
    samples = [] # [cluster id, ...]
    for n in range(K):
        sval = cluster_info["sval_points"][n][:nbest+1]
        p = (np.array(sval) - sval[-1]) # prob of last one will become 0
        p /= np.sum(p)

        nid_select = np.random.choice(nbest+1, size=nsamples_for_each, p=p)
        selected_id.extend([[n+1, i] for i in nid_select])
        samples.extend([cluster_info["loc_points"][n][i] for i in nid_select])

    params = cvt_ind2params(samples)

    return params, selected_id
        

def cvt_ind2params(sample_inds):
    # NOTE: This is the temporal function that hard fix parameters to reconstruct parameters

    # inter-population projection
    alpha_set = np.linspace(0, 2, 15)
    beta_set  = np.linspace(0, 1, 15)
    rank_set  = [0, 0.5, 1]
    w_set = np.linspace(0.1, 1, 7)

    # intra-population projection
    a_set = [9.4, 4.5]
    b_set = [3, 2.5]
    c = 0.01
    d_set = [14142.14, 15450.58]
    plim = [[0.051, 0.234],
            [0.028, 0.105]]

    params = []
    for sample_id in sample_inds: # nrow: alpha, ncol: beta, nrank, nw
        # NOTE check parameter order in 'main.c'
        nrow, ncol, nrank, nw = sample_id

        alpha = alpha_set[nrow]
        beta  = beta_set[ncol]
        rank  = rank_set[nrank]
        w     = w_set[nw]

        pe_fs = [plim[i][0]*rank + plim[i][1]*(1-rank) for i in range(2)]
        pi_fs = [b_set[i] * pe_fs[i] for i in range(2)]
        
        we_fs = [c * np.sqrt(0.01) / np.sqrt(pe) for pe in pe_fs]
        wi_fs = [c * np.sqrt(0.01) / np.sqrt(pi) for pi in pi_fs]
        
        param_sub = [
            we_fs[0], wi_fs[0], we_fs[1], wi_fs[1],
            pe_fs[0], pe_fs[0], w*alpha*pe_fs[0], w*alpha*pe_fs[0],
            pi_fs[0], pi_fs[0], w*beta*pi_fs[0],  w*beta*pi_fs[0],
            pe_fs[1], pe_fs[1], alpha*pe_fs[1],   alpha*pe_fs[1],
            pe_fs[1], pe_fs[1], beta*pe_fs[1],    beta*pe_fs[1],
            d_set[0]*np.sqrt(pe_fs[0]),
            d_set[1]*np.sqrt(pe_fs[1])
        ]
        
        params.append(param_sub)
    return params


if __name__ == "__main__":
    seed = 1000
    np.random.seed(seed)
    main(fname_cinfo="../three_pop_mpi/clustering/data/cluster_repr_points_rank3.pkl",
        nbest=10, nsamples_for_each=100, run_c=True,
        fname_out="./params_to_run.txt", fdir_out="./data")
    