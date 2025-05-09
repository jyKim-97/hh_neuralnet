import numpy as np
import pickle as pkl

import argparse
from tqdm import trange

import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
sys.path.append("../extract_osc_motif")
import oscdetector as od

import matplotlib.pyplot as plt
# summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data", load_only_control=True)

def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", required=False, type=int, default=0)
    parser.add_argument("--wbin_t", default=0.5, type=float)
    parser.add_argument("--mbin_t", default=0.01, type=float)
    parser.add_argument("--th", default=75, type=float)
    # parser.add_argument("--reverse", default=False, type=bool)
    parser.add_argument("--fdir", default=None, type=str, help="directory for summary_obj")
    parser.add_argument("--famp", default=None, type=str, help="fname for amplitude range (.pkl)")
    parser.add_argument("--fout", default=None, type=str, help="output file name (.pkl)")
    parser.add_argument("--nc", default=-1, type=int, help="Use if actual cluster id and index is different")
    parser.add_argument("--compute_avg_spec", action="store_true", help="Export spectrum for selected cluster ID")
    parser.add_argument("--prefix_spec", default=None, help="Export directory")
    return parser


srate = 2000

FLAG_SPEC = False
PREFIX = False

import os

def main(cid=None, wbin_t=None, mbin_t=None, th=None,
         fdir=None, famp=None, fout=None, nc=-1,
         compute_avg_spec=False, prefix_spec=None):
    if fdir is None:
        fdir = "./simulation_data/data_new/"
    if famp is None:
        famp = "/home/jungyoung/Project/hh_neuralnet/extract_osc_motif/data/osc_motif/amp_range_set.pkl"
    if fout is None:
        fout = os.path.join(fdir, "./osc_motif/motif_info_%d.pkl"%(nc))
        
    if compute_avg_spec:
        if prefix_spec is None:
            raise ValueError("The argument 'prefix_spec' is required")
    else:
        if prefix_spec is not None:
            raise ValueError("'compute_avg_spec' mode is not selected")
    
    global FLAG_SPEC, PREFIX
    FLAG_SPEC = compute_avg_spec
    PREFIX = prefix_spec
    
    # load amplitude_range
    amp_range_set = load_amp_range(famp)
    
    # read summary_obj
    summary_obj = hhtools.SummaryLoader(fdir)
    
    if cid == 0: # run for all classes
        cid_set = summary_obj.controls["cluster_id"]
        for nc, cid in enumerate(cid_set):
            cid = int(cid)
            amp_range = amp_range_set[cid-1]
            fout = os.path.join(fdir, "./osc_motif/motif_info_%d.pkl"%(nc))
            save_motif(fout, summary_obj, nc, amp_range, wbin_t, mbin_t, th)    
            
        # for n in range(summary_obj.num_controls[0]):
        #     fout = "./data/osc_motif/motif_info_%d.pkl"%(n+1)
        #     save_motif(fout, n+1, amp_range_set[n], wbin_t, mbin_t, th, reverse)
    else:
        if nc == -1: nc = cid-1
        amp_range = amp_range_set[cid-1]
        save_motif(fout, summary_obj, nc, amp_range, wbin_t, mbin_t, th)    
    
    


def load_amp_range(famp):
    if famp is None:
        famp = "/home/jungyoung/Project/hh_neuralnet/information_routing/data/osc_motif/amp_range_set.pkl"
    with open(famp, "rb") as fp:
        data = pkl.load(fp)
    print("amp_range_set is updated in %s"%(data["last-updated"]))
    return data["amp_range_set"]


def read_current_time():
    from datetime import datetime
    now = datetime.now()
    return "%d%02d%02d"%(now.year, now.month, now.day)


def save_motif(fout, summary_obj, nc, amp_range, wbin_t, mbin_t, th):
    winfo_set = collect_osc_motif(summary_obj, nc, amp_range, wbin_t, mbin_t, th)
    with open(fout, "wb") as fp:
        pkl.dump({"winfo": winfo_set, 
                  "metainfo": {"amp_range": amp_range,
                               "wbin_t": wbin_t,
                               "mbin_t": mbin_t,
                               "th": th,
                               "fdir": summary_obj.fdir,
                               "last-updated": read_current_time()}
                  }, fp)


def collect_osc_motif(summary_obj, nc, amp_range, wbin_t, mbin_t, th):
    winfo_set = []
    for nr in trange(summary_obj.num_controls[0]):
        winfo_set.append([[] for _ in range(16)])
        
        if FLAG_SPEC:
            psd_motif = [[0, 0] for _ in range(16)]
            num_psd = np.zeros(16)
        
        for i in range(summary_obj.num_controls[2]):
            detail_data = summary_obj.load_detail(nr, nc, i)
            
            # detect oscillatory motif
            psd_set, fpsd, tpsd = od.compute_stfft_all(detail_data, mbin_t=mbin_t, wbin_t=wbin_t)
            words = od.compute_osc_bit(psd_set[1:], fpsd, tpsd, amp_range, q=th, min_len=2, cat_th=2)
            osc_motif = od.get_motif_boundary(words, tpsd)
            
            # add information
            for motif in osc_motif:
                nw = motif["id"]
                tl = motif["range"]
                if tl[1]-tl[0] < mbin_t: continue
                
                winfo_set[nr][nw].append((i, tl))
                
            # save
            if FLAG_SPEC:
                psd_set = np.array(psd_set)
                mpsd = psd_set.mean(axis=2)
                for motif in osc_motif:
                    nw = motif["id"]
                    tl = motif["range"]
                    
                    if tl[1]-tl[0] < mbin_t: continue
                    nl = [int((t-tpsd[0])/mbin_t) for t in tl]
                    assert nl[1] > nl[0]
                    # num = nl[1] - nl[0]
                    psd_motif[nw][0] += (psd_set[1,:,nl[0]:nl[1]] - mpsd[1,:,None]).sum(axis=1)
                    psd_motif[nw][1] += (psd_set[2,:,nl[0]:nl[1]] - mpsd[2,:,None]).sum(axis=1)
                    
                    # psd_motif[nw][0] += (psd_set[1,:,:] - mpsd[1,:,None]).mean(axis=1)
                    # psd_motif[nw][1] += (psd_set[2,:,:] - mpsd[2,:,None]).mean(axis=1)
                    num_psd[nw] += nl[1] - nl[0]
        
        if FLAG_SPEC:
            # psd_motif = np.array(psd_motif) / num_psd[:,None,None]
            for nw in range(16):
                if num_psd[nw] == 0: continue
                
                fig = plt.figure(figsize=(4, 3))
                # print(psd_motif[nw][0].shape)
                plt.plot(fpsd, psd_motif[nw][0]/num_psd[nw], c="r", lw=1.5)
                plt.plot(fpsd, psd_motif[nw][1]/num_psd[nw], c="b", lw=1.5)
                
                # plt.plot(fpsd, mpsd[1], c="r", lw=1.5)
                # plt.plot(fpsd, mpsd[2], c="b", lw=1.5)
                fout = PREFIX + "_%d%02d.png"%(nr, nw)
                plt.savefig(fout, bbox_inches="tight")
                plt.close(fig)
                
        # print(len(winfo_set[-1][2]), num_psd[2])
        # print(winfo_set[0][2])
    
    return winfo_set
                    
            
    

# def save_motif(fout, nc, amp_range, wbin_t, mbin_t, th, reverse):
#     winfo = collect_osc_motif(nc, amp_range, wbin_t, mbin_t, th, reverse=reverse)
    
#     # save information
#     # if reverse:
#     #     fname = "./data/osc_motif%s/motif_info_%d(low).pkl"%(cid)
#     # else:
#     #     fname = "./data/osc_motif%s/motif_info_%d.pkl"%(tag, cid)
    
#     print("Saved into %s"%(fout))
#     with open(fout, "wb") as fp:
#         pkl.dump({"winfo": winfo, 
#                   "metainfo": {"amp_range": amp_range,
#                                "wbin_t": wbin_t,
#                                "mbin_t": mbin_t,
#                                "th": th,
#                                "reverse": reverse,
#                                "last-updated": read_current_time()}
#                   }, fp)


# def collect_osc_motif(nc, amp_range, wbin_t, mbin_t, th, reverse=False):
#     # winfo = [[] for _ in range(16)]
#     winfo_set = []
#     for nr in trange(summary_obj.num_controls[1]):
#         winfo_set.append([[] for _ in range(16)])
#         for i in range(summary_obj.num_controls[2]):
#             detail_data = summary_obj.load_detail(nc, nr, i)
        
#             # detect oscillatory motif
#             psd_set, fpsd, tpsd = od.compute_stfft_all(detail_data, mbin_t=mbin_t, wbin_t=wbin_t)
#             words = od.compute_osc_bit(psd_set[1:], fpsd, tpsd, amp_range, q=th, min_len=2, cat_th=2, reversed=reverse)
#             osc_motif = od.get_motif_boundary(words, tpsd)
        
#             # add information
#             for motif in osc_motif:
#                 nw = motif["id"]
#                 tl = motif["range"]
#                 winfo_set[nr][nw].append((i, tl))
            
#     return winfo_set
    

if __name__ == "__main__":
    main(**vars(build_arg_parse().parse_args()))
