import numpy as np
import sys
import pickle as pkl
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import hhsignal
from tqdm import trange

import oscdetector as od
import hhinfo


srate = 2000
vrange = [-70, -50]
wbin_t, mbin_t = 0.5, 0.01

key_names = ("fpop", "spop")
amp_range_set = [
    dict(fpop=[[]], spop=[[]]), # 1
    dict(fpop=[[]], spop=[[]]), # 2
    dict(fpop=[[20, 30], [40, 50]],
         spop=[[20, 30]]), # 3
    dict(fpop=[[30, 40], [50, 70]],
         spop=[[30, 40]]), # 4
    dict(fpop=[[], [60, 70]],
         spop=[[25, 35], [60, 70]]), # 5 -> tuned
    dict(fpop=[[50, 70]], 
         spop=[[20, 40], [50, 70]]), # 6
    dict(fpop=[[60, 70]],
         spop=[[60, 70]]), # 7
    dict(fpop=[[30, 40], [60, 70]],
         spop=[[30, 40]]), # 8 -> tuned
]

summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data", load_only_control=True)
# summary_obj = hhtools.SummaryLoader("~/Project/hh_neuralnet/gen_three_pop_samples_repr/data", load_only_control=True)


def main(cid=5, word_id=4, srate_d=2000):
    
     tlag_max = 30 # ms
     bin_size_set = np.arange(6, 152, 5)
     
     # collect osc motif
     osc_motif = collect_osc_motif(cid, word_id)
      
     # compute te
     nlag_max = int(tlag_max * srate_d / 1000)
     te_result = []
     for nbin in bin_size_set:
          te_result.append(compute_te(osc_motif, nlag_max, srate_d, nbin))
          
     with open("./data/te_validation_%d.pkl"%(srate_d), "wb") as fp:
          pkl.dump(dict(
               metainfo=dict(cid=cid, word_id=word_id, srate_d=srate_d, tlag_max=tlag_max),
               tlag=np.arange(nlag_max)/srate_d*1e3,
               bin_size=bin_size_set,
               te=np.swapaxes(np.array(te_result), 0, 2),
               desc="(Ncase, F->S/S->F, Nbin, Nlag)"
          ), fp)


def digitize_v(v, vrange, nbin):
    v[v < vrange[0]] = vrange[0]
    v[v > vrange[1]] = vrange[1]
    
    de = (vrange[1] - vrange[0])/(nbin-1)
    e = np.arange(vrange[0]-de/2, vrange[1]+de, de)
    
    return np.digitize(v, e)


def compute_te(osc_motif, nlag_max, srate_d, nbin):
    
    te_result = np.zeros((2, len(osc_motif), nlag_max))
    
    vf, vs = None, None
    prev_info = (-1, -1)
    for n in trange(len(osc_motif), desc="srate: %d, nbin: %d"%(srate_d, nbin)):
        info = osc_motif[n][0]
        if prev_info != info:
            prev_info = info
            detail_data = summary_obj.load_detail(info[0], info[1])
            
            # downsample
            vf, vs = detail_data["vlfp"][1:]
            vf_d = digitize_v(hhsignal.downsample(vf, srate, srate_d), vrange, nbin)
            vs_d = digitize_v(hhsignal.downsample(vs, srate, srate_d), vrange, nbin)
            
        nr = osc_motif[n][1]
        if nr[1] - nr[0] < nlag_max:
            continue
        
        x = vf_d[nr[0]:nr[1]]
        y = vs_d[nr[0]:nr[1]]
        
        te_result[0][n], _ = hhinfo.compute_te(x, y, nbin, nlag_max) # fpop -> spop
        te_result[1][n], _ = hhinfo.compute_te(y, x, nbin, nlag_max) # spop -> fpop
    
    return te_result
    

def collect_osc_motif(cid, word_id):
     osc_motif = []
     amp_range = amp_range_set[cid-1]
     for i in trange(200, desc="collect oscillation motif"):
         detail_data = summary_obj.load_detail(cid-1, i)
         psd_set, fpsd, tpsd = od.compute_stfft_all(detail_data, mbin_t=mbin_t, wbin_t=wbin_t)
         words = od.compute_osc_bit(psd_set[1:], fpsd, tpsd, amp_range, q=80, min_len=2, cat_th=2)
         osc_motif_sub = od.get_motif_boundary(words)
        
         for n in range(len(osc_motif_sub)):
            if osc_motif_sub[n]["id"] == word_id:
               osc_motif.append([(cid-1, i), osc_motif_sub[n]["range"]])
               
     return osc_motif
                

import argparse


def build_arg_parse():
     parser = argparse.ArgumentParser()
     parser.add_argument("--cid", help="cluster id", required=True, type=int)
     parser.add_argument("--word_id", help="word id", required=True, type=int)
     parser.add_argument("--srate_d", help="sampling rate", required=True, type=int)
     return parser
     



if __name__ == "__main__":
#     pass
     main(**vars(build_arg_parse().parse_args()))

     # main(5, 4, 1000)