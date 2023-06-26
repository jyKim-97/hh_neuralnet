import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

curr_dir = "~/jungyoung/Project/hh_neuralnet/three_pop_mpi"

import hhclustering as hc

import sys
sys.path.append("../../include/")
import hhtools

def load_obj():
    obj = hhtools.SummaryLoader("../asym_link_full/")
    # correction
    obj.summary["chi"][:,:,:,:,24:48,0] = np.nan
    return obj


def export_sample_figs(obj, cluster_id, silhouette_vals, col_names, nshow=3):
    import os
    fdir = "./sample_figs"

    k0 = min(cluster_id)
    k1 = max(cluster_id)
    targets = []
    for target_cid in range(k0, k1+1):
        targets.append([])
        for case in ("best", "intermediate", "worst"):
            tags = hc.show_sample_cases(obj, target_cid, cluster_id, silhouette_vals,
                                col_names, case=case, nshow=nshow, save=True, fdir=fdir)
            targets[-1].append(tags)

    import pickle as pkl
    with open(os.path.join(fdir, "tags.pkl"), "wb") as fp:
        pkl.dump({"targets": targets}, fp)
        

def main():
    obj = load_obj()

    # load data
    with open("./data/rcluster.pkl", "rb") as fp:
        tmp = pkl.load(fp)

    rcluster_id = tmp["rcluster_id"]
    rsval = tmp["rsval"]

    # load data
    with open("./data/purified_data.pkl", "rb") as fp:
        buf = pkl.load(fp)
    col_names = buf["col_names"]

    export_sample_figs(obj, rcluster_id, rsval, col_names, nshow=3)


if __name__ == "__main__":
    main()
