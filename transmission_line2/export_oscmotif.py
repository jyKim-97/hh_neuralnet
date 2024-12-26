import numpy as np
import os
import pickle as pkl
import argparse
from tqdm import trange
import matplotlib.pyplot as plt

"""
Export oscillation motif for transmission line
"""

import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
sys.path.append("../extract_osc_motif")
import oscdetector as od

fdir_root = "/home/jungyoung/Project/hh_neuralnet/"
# summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data", load_only_control=True)

def build_arg_parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--tag", required=True, type=str, choices=("", "_mslow", "_mfast"))
    parser.add_argument("--fdir", default=None, type=str, help="directory for summary_obj")
    parser.add_argument("--cid", required=False, type=int, default=0)
    parser.add_argument("--wbin_t", default=0.5, type=float)
    parser.add_argument("--mbin_t", default=0.01, type=float)
    parser.add_argument("--th", default=75, type=float)
    parser.add_argument("--reverse", default=False, type=bool)
    # parser.add_argument("--fdir", default=None, type=str, help="directory for summary_obj")
    # parser.add_argument("--famp", default=None, type=str, help="fname for amplitude range (.pkl)")
    parser.add_argument("--fout", default=None, type=str, help="output file name (.pkl)")
    parser.add_argument("--nc", default=-1, type=int, help="Use if actual cluster id and index is different")
    parser.add_argument("--compute_avg_spec", action="store_true", help="Export spectrum for selected cluster ID")
    parser.add_argument("--prefix_spec", default=None, help="Export directory")
    return parser


summary_obj = None
num_itr_max = -1

FLAG_SPEC = False
PREFIX = False

# tag = "_mfast"
# tag = ""
# summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data%s"%(tag), load_only_control=True)

srate = 2000

def read_current_time():
    from datetime import datetime
    now = datetime.now()
    return "%d%02d%02d"%(now.year, now.month, now.day)


def load_amp_range(famp):
    with open(famp, "rb") as fp:
        data = pkl.load(fp)
        
    print("amp_range_set is updated in %s"%(data["last-updated"]))
    return data["amp_range_set"]


def main(fdir=None, cid=None, wbin_t=None, mbin_t=None, 
         th=75, reverse=False, fout=None, nc=-1,
         compute_avg_spec=False, prefix_spec=None):
    global summary_obj, num_itr_max
    
    # fdir_summary = os.path.join(fdir_root, "gen_three_pop_samples_repr/data%s"%(tag))
    # fname_amp = os.path.join(fdir_root, "extract_osc_motif/data/osc_motif%s/amp_range_set.pkl"%(tag))
    fname_amp = os.path.join(fdir_root, "extract_osc_motif/data/osc_motif/amp_range_set.pkl")
    
    # print("tag: %s"%(tag))
    print("fdir_summary: %s"%(fdir))
    print("fname_amp: %s"%(fname_amp))
    
    amp_range_set = load_amp_range(fname_amp)
    summary_obj = hhtools.SummaryLoader(fdir)
    # summary_obj = hhtools.SummaryLoader(fdir_summary)
    num_itr_max = summary_obj.num_controls[1]
    
    if compute_avg_spec:
        if prefix_spec is None:
            raise ValueError("The argument 'prefix_spec' is required")
    else:
        if prefix_spec is not None:
            raise ValueError("'compute_avg_spec' mode is not selected")
        
    global FLAG_SPEC, PREFIX
    FLAG_SPEC = compute_avg_spec
    PREFIX = prefix_spec
    
    if cid == 0: # run for all motif
        os.mkdir(os.path.join(fdir, "osc_motif"))
        if FLAG_SPEC:
            os.mkdir(os.path.join(fdir, "osc_motif/figs"))
        for nc in range(summary_obj.num_controls[0]):
            cid = summary_obj.controls["cluster_id"][nc]
            cid = int(cid)
            fout = os.path.join(fdir, "osc_motif/motif_info_%d.pkl"%(cid))
            PREFIX = os.path.join(fdir, "osc_motif/figs/spec#%d"%(cid))
            # fout = "./postdata/osc_motif/motif_info_%d.pkl"%(cid)
            # PREFIX = "./figs/motif_spec/spec#%d"%(cid)
            # FLAG_SPEC = "./figs/motif_spec/spec#%d"%(cid)
            # print(FLAG_SPEC)
            save_motif(fout, nc, amp_range_set[cid-1], wbin_t, mbin_t, th, reverse)
    else:
        if fout is None:
            # fout = "./data/osc_motif/motif_info_%d.pkl"%(cid)
            fout = "./postdata/osc_motif/motif_info_%d.pkl"%(cid)
        
        if nc == -1: nc = cid-1
        save_motif(fout, nc, amp_range_set[cid-1], wbin_t, mbin_t, th, reverse)    
    

def save_motif(fout, nc, amp_range, wbin_t, mbin_t, th, reverse):
    winfo = collect_osc_motif(nc, amp_range, wbin_t, mbin_t, th, reverse=reverse)
    
    # save information
    # if reverse:
    #     fname = "./data/osc_motif%s/motif_info_%d(low).pkl"%(cid)
    # else:
    #     fname = "./data/osc_motif%s/motif_info_%d.pkl"%(tag, cid)
    
    print("Saved into %s"%(fout))
    with open(fout, "wb") as fp:
        pkl.dump({"winfo": winfo, 
                  "metainfo": {"amp_range": amp_range,
                               "wbin_t": wbin_t,
                               "mbin_t": mbin_t,
                               "th": th,
                               "reverse": reverse,
                               "last-updated": read_current_time()}
                  }, fp)


def collect_osc_motif(nc, amp_range, wbin_t, mbin_t, th, reverse=False):
    
    if FLAG_SPEC:
        psd_motif = [[0, 0] for _ in range(16)]
        num_psd = np.zeros(16)
    
    winfo = [[] for _ in range(16)]
    for i in trange(num_itr_max, desc="detecting oscillation motifs"):
        detail_data = summary_obj.load_detail(nc, i)
        
        # detect oscillatory motif
        psd_set, fpsd, tpsd = od.compute_stfft_all(detail_data, mbin_t=mbin_t, wbin_t=wbin_t)
        words = od.compute_osc_bit(psd_set[1:], fpsd, tpsd, amp_range, q=th, min_len=2, cat_th=2, reversed=reverse)
        osc_motif = od.get_motif_boundary(words, tpsd)
        
        # add information
        for motif in osc_motif:
            nw = motif["id"]
            tl = motif["range"]
            winfo[nw].append((i, tl))
            
        # add spec
        if FLAG_SPEC:
            psd_set = np.array(psd_set)
            mpsd = psd_set.mean(axis=2)
            for motif in osc_motif:
                nw = motif["id"]
                tl = motif["range"]
                
                if tl[1]-tl[0] < mbin_t: continue
                nl = [int((t-tpsd[0])/mbin_t) for t in tl]
                assert nl[1] > nl[0]
                psd_motif[nw][0] += (psd_set[1,:,nl[0]:nl[1]] - mpsd[1,:,None]).sum(axis=1)
                psd_motif[nw][1] += (psd_set[2,:,nl[0]:nl[1]] - mpsd[2,:,None]).sum(axis=1)
                num_psd[nw] += nl[1] - nl[0]
                
    if FLAG_SPEC:
            # psd_motif = np.array(psd_motif) / num_psd[:,None,None]
        for nw in range(16):
            if num_psd[nw] == 0: continue
            
            fig = plt.figure(figsize=(4, 3))
            # print(psd_motif[nw][0].shape)
            plt.plot(fpsd, psd_motif[nw][0]/num_psd[nw], c="r", lw=1.5)
            plt.plot(fpsd, psd_motif[nw][1]/num_psd[nw], c="b", lw=1.5)
            fout = PREFIX + "_%02d.png"%(nw)
            plt.savefig(fout, bbox_inches="tight")
            plt.close(fig)
            
    return winfo
    

if __name__ == "__main__":
    main(**vars(build_arg_parse().parse_args()))
