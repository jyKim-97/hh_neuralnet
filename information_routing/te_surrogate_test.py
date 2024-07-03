import numpy as np
import sys
import pickle as pkl
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import hhsignal
from tqdm import trange

import oscdetector as od
import hhinfo


def load_amp_range():
    with open('./data/amp_range_set.pkl', "rb") as fp:
        data = pkl.load(fp)
    print("amp_range_set is updated in %s"%(data["last-updated"]))
    return data["amp_range_set"]

# load information
amp_range_set = load_amp_range()
summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data", load_only_control=True)


srate = 2000
vrange = [-70, -50]
wbin_t, mbin_t = 0.5, 0.01



def main(cid, word_id):
    vf_seg, vs_seg = collect_voltage_segment(cid, word_id, 10)


def digitize_v(v, vrange, nbin):
    v[v < vrange[0]] = vrange[0]
    v[v > vrange[1]] = vrange[1]
    
    de = (vrange[1] - vrange[0])/(nbin-1)
    e = np.arange(vrange[0]-de/2, vrange[1]+de, de)
    
    return np.digitize(v, e)


def collect_voltage_segment(cid, word_id, min_len):
    amp_range = amp_range_set[cid-1]
    vf_seg, vs_seg = [], []
    for i in trange(200, desc="collect voltage segments"):
        # get digitized voltage
        detail_data = summary_obj.load_detail(cid-1, i)
        vf_d = digitize_v(vf, vrange, nbin)
        vs_d = digitize_v(vs, vrange, nbin)
        
        # detect oscillatory motif
        psd_set, fpsd, tpsd = od.compute_stfft_all(detail_data, mbin_t=mbin_t, wbin_t=wbin_t)
        words = od.compute_osc_bit(psd_set[1:], fpsd, tpsd, amp_range, q=80, min_len=2, cat_th=2)
        osc_motif = od.get_motif_boundary(words, np.arange(len(tpsd)))
        
        # select target motif
        for motif in osc_motif:
            if motif["id"] != word_id:
                continue
            
            nr = motif["range"]
            if nr[1] - nr[0] < min_len:
                continue
            
            vf_seg.append(vf_d[nr[0]:nr[1]])
            vs_seg.append(vs_d[nr[0]:nr[1]])
            
    return vf_seg, vs_seg
        
        
        
         

# def collect_osc_motif(cid, word_id):
#     osc_motif = []
#      amp_range = amp_range_set[cid-1]
#      for i in trange(200, desc="collect oscillation motif"):
#          detail_data = summary_obj.load_detail(cid-1, i)
#          psd_set, fpsd, tpsd = od.compute_stfft_all(detail_data, mbin_t=mbin_t, wbin_t=wbin_t)
#          words = od.compute_osc_bit(psd_set[1:], fpsd, tpsd, amp_range, q=80, min_len=2, cat_th=2)
#          osc_motif_sub = od.get_motif_boundary(words)
        
#          for n in range(len(osc_motif_sub)):
#             if osc_motif_sub[n]["id"] == word_id:
#                osc_motif.append([(cid-1, i), osc_motif_sub[n]["range"]])
               
#      return osc_motif


# def read_voltage(info):
#     detail_data = summary_obj.load_detail(info[0], info[1])
            
#     # downsample
#     vf, vs = detail_data["vlfp"][1:]
#     vf_d = digitize_v(hhsignal.downsample(vf, srate, srate_d), vrange, nbin)
#     vs_d = digitize_v(hhsignal.downsample(vs, srate, srate_d), vrange, nbin)

#     return vf_d, vs_d


# def compute_te(osc_motif, nlag_max, srate_d, nbin):
    
#     te_result = np.zeros((2, len(osc_motif), nlag_max))
    
#     vf, vs = None, None
#     prev_info = (-1, -1)
#     for n in trange(len(osc_motif), desc="srate: %d, nbin: %d"%(srate_d, nbin)):
#         info = osc_motif[n][0]
#         if prev_info != info:
#             prev_info = info
#             vf_d, vs_d = read_voltag(info)
            
#         nr = osc_motif[n][1]
#         if nr[1] - nr[0] < nlag_max:
#             continue
        
#         x = vf_d[nr[0]:nr[1]]
#         y = vs_d[nr[0]:nr[1]]
        
#         te_result[0][n], _ = hhinfo.compute_te(x, y, nbin, nlag_max) # fpop -> spop
#         te_result[1][n], _ = hhinfo.compute_te(y, x, nbin, nlag_max) # spop -> fpop
    
#     return te_result





if __name__=="__main__":
    main(8, 13)