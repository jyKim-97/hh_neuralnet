import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from argparse import ArgumentParser

curr_dir = "~/jungyoung/Project/hh_neuralnet/three_pop_mpi"

import hhclustering as hc

import sys
sys.path.append("../../include/")
import hhtools


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--fdir_data", required=True,
                        help="diretory of the full simulation dataset")
    parser.add_argument("--fcluster", required=True)
    parser.add_argument("--fpdata", required=True)
    parser.add_argument("--fdir_out", required=True)
    parser.add_argument("--nshow", default=3, type=int)
    return parser


def load_obj(fdir_data):
    obj = hhtools.SummaryLoader(fdir_data)
    # correction
    # obj.summary["chi"][:,:,:,:,24:48,0] = np.nan
    obj.summary["chi"][obj.summary["chi"] > 1] = np.nan
    return obj


def export_sample_figs(obj, fdir_out, cluster_id, silhouette_vals, col_names, nshow=3):
    import os

    k0 = min(cluster_id)
    k1 = max(cluster_id)
    targets = []
    for target_cid in range(k0, k1+1):
        targets.append([])
        for case in ("best", "intermediate", "worst"):
            tags = hc.show_sample_cases(obj, target_cid, cluster_id, silhouette_vals,
                                col_names, case=case, nshow=nshow, save=True, fdir=fdir_out)
            targets[-1].append(tags)

    import pickle as pkl
    with open(os.path.join(fdir_out, "tags.pkl"), "wb") as fp:
        pkl.dump({"targets": targets}, fp)
        

def main(fdir_data=None, fcluster=None, fpdata=None, fdir_out=None, nshow=3):
    obj = load_obj(fdir_data)

    # load data
    with open(fcluster, "rb") as fp:
        tmp = pkl.load(fp)

    rcluster_id = tmp["rcluster_id"]
    rsval = tmp["rsval"]

    # load data
    with open(fpdata, "rb") as fp:
        buf = pkl.load(fp)
    col_names = buf["col_names"]

    export_sample_figs(obj, fdir_out, rcluster_id, rsval, col_names, nshow=nshow)


if __name__ == "__main__":
    main(**vars(build_argument_parser().parse_args()))
