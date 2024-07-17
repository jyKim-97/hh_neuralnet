import numpy as np
import pickle as pkl
import oscdetector as od
import argparse
from tqdm import trange

import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
# summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data", load_only_control=True)

def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", required=False, type=int, default=0)
    parser.add_argument("--wbin_t", default=0.5, type=float)
    parser.add_argument("--mbin_t", default=0.01, type=float)
    parser.add_argument("--th", default=75, type=float)
    parser.add_argument("--reverse", default=False, type=bool)
    return parser


tag = "_mfast"
# tag = ""
summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data%s"%(tag), load_only_control=True)

num_itr_max = summary_obj.num_controls[1]
srate = 2000

def read_current_time():
    from datetime import datetime
    now = datetime.now()
    return "%d%02d%02d"%(now.year, now.month, now.day)


def load_amp_range():
    # with open('./data/amp_range_set.pkl', "rb") as fp:
    # with open("/home/jungyoung/Project/hh_neuralnet/information_routing/data/osc_motif/amp_range_set.pkl", "rb") as fp:
    with open("/home/jungyoung/Project/hh_neuralnet/information_routing/data/osc_motif%s/amp_range_set.pkl"%(tag), "rb") as fp:
        data = pkl.load(fp)
    print("amp_range_set is updated in %s"%(data["last-updated"]))
    return data["amp_range_set"]


def main(cid=None, wbin_t=None, mbin_t=None, th=None, reverse=False):
    amp_range_set = load_amp_range()
    
    if cid == 0:
        for n in range(summary_obj.num_controls[0]):
            save_motif(n+1, amp_range_set[n], wbin_t, mbin_t, th, reverse)
    else:
        save_motif(cid, amp_range_set[cid-1], wbin_t, mbin_t, th, reverse)    
    

def save_motif(cid, amp_range, wbin_t, mbin_t, th, reverse):
    winfo = collect_osc_motif(cid, amp_range, wbin_t, mbin_t, th, reverse=reverse)
    
    # save information
    if reverse:
        fname = "./data/osc_motif%s/motif_info_%d(low).pkl"%(cid)
    else:
        fname = "./data/osc_motif%s/motif_info_%d.pkl"%(tag, cid)
    print("Saved into %s"%(fname))
    with open(fname, "wb") as fp:
        pkl.dump({"winfo": winfo, 
                  "metainfo": {"amp_range": amp_range,
                               "wbin_t": wbin_t,
                               "mbin_t": mbin_t,
                               "th": th,
                               "reverse": reverse,
                               "last-updated": read_current_time()}
                  }, fp)


def collect_osc_motif(cid, amp_range, wbin_t, mbin_t, th, reverse=False):
    winfo = [[] for _ in range(16)]
    for i in trange(num_itr_max, desc="detecting oscillation motifs"):
        detail_data = summary_obj.load_detail(cid-1, i)
        
        # detect oscillatory motif
        psd_set, fpsd, tpsd = od.compute_stfft_all(detail_data, mbin_t=mbin_t, wbin_t=wbin_t)
        words = od.compute_osc_bit(psd_set[1:], fpsd, tpsd, amp_range, q=th, min_len=2, cat_th=2, reversed=reverse)
        osc_motif = od.get_motif_boundary(words, tpsd)
        
        # add information
        for motif in osc_motif:
            nw = motif["id"]
            tl = motif["range"]
            winfo[nw].append((i, tl))
            
    return winfo
    

if __name__ == "__main__":
    main(**vars(build_arg_parse().parse_args()))
