import numpy as np
import pickle as pkl
from tqdm import tqdm

import argparse
import utils
import hhclustering as hc
from multiprocessing import Pool

_ncore = 4

def load_data(fdata):    
    with open(fdata, "rb") as fp:
        return pkl.load(fp)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fdata", help="simulated data", required=True)
    parser.add_argument("--fout", help="output file name", default="./data/kmeans_result.pkl")
    parser.add_argument("--nitr", help="the # of iteration", type=int, default=10)
    parser.add_argument("--seed", help="seed used to calcualte", default=200, type=int)
    return parser


def parrun(func, args, desc=None):
    
    outs = []
    p = Pool(_ncore)
    with tqdm(total=len(args), desc=desc) as pbar:
        if _ncore == 1:
            for res in args:
                outs.append(func(res))
                pbar.update() 
        else:
            for n, res in enumerate(p.imap(func, args)):
                outs.append(res)
                pbar.update()
            
    id_sort = np.argsort([o[0] for o in outs])
    res = [outs[i][1:] for i in id_sort]
    p.close()
    p.join()
    
    return res


def run_kmeans(args):
    kobj, _, scoeff = hc.kmeans_specific_seed(args["nc"], args["data"].copy(), args["seed"])
    cid = kobj.predict(args["data"].T)
        
    return args["job_id"], cid, kobj.inertia_, scoeff

def main(fdata=None, fout="./kmeans_out.pkl", seed=100, nitr=100):
    # get seed
    np.random.seed(seed)
    seeds = np.random.randint(low=1, high=10000, size=nitr).astype(int)

    # get cluster 
    pdata = load_data(fdata)
    num_clusters = np.arange(2, 20)
    
    args = []
    num = len(num_clusters) * nitr
    for nid in range(num):
        n = nid // nitr
        nc = num_clusters[n]
        seed = seeds[nid % nitr]
        
        args.append(dict(job_id=nid, data=pdata["data"], seed=seed, nc=nc))
        
    res = parrun(run_kmeans, args, desc="run_kmeans")
    
    # reshape
    cid_sets = np.reshape([r[0] for r in res], (len(num_clusters), nitr, -1))
    kcoeff_sets = np.reshape([r[1] for r in res], (len(num_clusters), nitr))
    scoeff_sets = np.reshape([r[2] for r in res], (len(num_clusters), nitr))
    
    # save
    print("save to %s"%(fout))
    with open(fout, "wb") as fp:
        pkl.dump({"cid_sets": cid_sets,
                  "kcoeff_sets": kcoeff_sets,
                  "scoeff_sets": scoeff_sets,
                  "seeds": seeds,
                  "date": utils.get_date_string()},
                  fp)


if __name__ == "__main__":
    main(**vars(build_arg_parser().parse_args()))
