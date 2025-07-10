"""Extract burst probs from selected landmark"""

import numpy as np
import os
import pickle as pkl
import argparse
from tqdm import tqdm

import sys
sys.path.append("../include")
import hhsignal
import hhtools
import burst.burst_tools as bt


# fdir = "./simulation_data/data"
fdir = "./data"
famp_range = "../extract_osc_motif/data/osc_motif/amp_range_set.pkl"
wbin_t = 0.5
mbin_t = 0.01
max_std_ratio = 5
min_std_ratio = 1.96 # 1.645
amp_range = None


def build_parser():
    parser = argparse.ArgumentParser(description="Extract burst when two populations are disconnected")
    parser.add_argument("--fout", type=str, default="./tmp.pkl", help="File name to save the data")
    parser.add_argument("--cid", type=int, required=True, help="Regime ID (integer)")
    return parser


def set_psd_th(psd_avg, fpsd, ntp):
    assert ntp in (1, 2)
    pop = "fpop" if ntp == 1 else "spop"
    th_m = psd_avg.copy()
    ar = amp_range[pop]
    
    # set leftcase
    n0, n1 = -1, -1
    if len(ar[0]) == 0: # empty
        nl = np.where(fpsd < ar[1][0])[0][-1] - 1
        th_m[:nl] = th_m[nl]*1.2
    else:
        nl = np.where(fpsd < ar[0][0])[0][-1] - 1
        th_m[:nl] = th_m[nl]*1.2
        n0 = np.where(fpsd > ar[0][1])[0][0] + 1
    
    if len(ar[1]) == 0:
        nr = np.where(fpsd > ar[0][1])[0][0] + 1
        th_m[nr:] = th_m[nr]*1.2
    else:
        nr = np.where(fpsd > ar[1][1])[0][0] + 1
        th_m[nr:] = th_m[nr]*1.2
        n1 = np.where(fpsd < ar[1][0])[0][-1] - 1
    
    if (n0 != -1) and (n1 != -1):
        th_m[n0:n1] = (th_m[n1]-th_m[n0])/(n1-n0)*np.arange(n1-n0) + th_m[n0]
        th_m[n0:n1] = th_m[n0:n1]*1.2
        
    return th_m


def main(fout=None, cid=None):
    global amp_range
    
    summary_obj = hhtools.SummaryLoader(fdir)
    with open(famp_range, "rb") as fp:
        amp_range = pkl.load(fp)["amp_range_set"][cid-1]
    
    burst_props = extract_probs(summary_obj, cid)
    burst_props = clean_bprops(burst_props)
    with open(fout, "wb") as fp:
        pkl.dump(dict(
            attrs=dict(fdir=fdir, cid=cid,
                       wbin_t=wbin_t, mbin_t=mbin_t, 
                       max_std_ratio=max_std_ratio, min_std_ratio=min_std_ratio,
                       source_dir=fdir),
            burst_props=burst_props
        ), fp)    


def extract_probs(summary_obj, cid):
    num_trial = summary_obj.num_controls[1]
    
    burst_props = [dict(
        burst_f=[],
        burst_len=[],
        burst_amp=[],
        burst_range=[],
        id_trial=[],
        tpsd=[]
        ) for _ in range(2)]
    
    for nt in tqdm(range(num_trial)):
    # for nt in range(50, 51):
        detail = summary_obj.load_detail(cid-1, nt)
        t = detail["ts"]
        for k in range(2):
            ntp = k+1
            # remove first 0.5 s signal
            # idt = t > 0.5
            # v = detail["vlfp"][ntp][idt]
            psd, fpsd, tpsd = hhsignal.get_stfft(detail["vlfp"][ntp], t, 2000, frange=(10, 90), wbin_t=wbin_t, mbin_t=mbin_t)
            
            idt = tpsd >= wbin_t
            psd  = psd[:,idt]
            tpsd = tpsd[idt]
            
            # get threshold
            th_m = set_psd_th(psd.mean(axis=1), fpsd, ntp)
            th_s = set_psd_th(psd.std(axis=1), fpsd, ntp)
            th_m = np.tile(th_m[:,np.newaxis], (1, psd.shape[1]))
            th_s = np.tile(th_s[:,np.newaxis], (1, psd.shape[1]))
            th_s = th_s.mean()
                
            bmap = bt.find_blob_filtration(psd, th_m, th_s,
                                           std_min=min_std_ratio, std_max=max_std_ratio, std_step=0.1, nmin_width=1)
            burst_f, burst_range, burst_amp = bt.extract_burst_attrib(psd, fpsd, bmap)
            
            # if nt == 50 and k == 1:
            #     print("Trial %d, pop %d"%(nt, ntp))
                
            #     import matplotlib.pyplot as plt
                
            #     plt.figure(figsize=(4, 3))
            #     plt.imshow(psd, extent=(tpsd[0], tpsd[-1], fpsd[0], fpsd[-1]), aspect="auto", cmap="jet", origin="lower", alpha=0.8, interpolation='none')
            #     plt.xlim([0.5, 3])
            #     plt.show()
            
            burst_props[k]["burst_f"].extend(burst_f)
            burst_props[k]["burst_len"].extend(burst_range[:, 1] - burst_range[:, 0])
            burst_props[k]["burst_amp"].extend(burst_amp)
            burst_props[k]["burst_range"].extend(burst_range)
            burst_props[k]["id_trial"].extend([nt]*len(burst_f))
            
        # if nt == 50:
        #     npop = 1
        #     for i in range(len(burst_props[npop]["burst_f"])):
        #         print("[%.2f, %.2f] (%.1f)"%(
        #             tpsd[int(burst_props[npop]["burst_range"][i][0])],
        #             tpsd[int(burst_props[npop]["burst_range"][i][1])],
        #             burst_props[npop]["burst_f"][i]
        #         ))
    
    for k in range(2):
        burst_props[k]["tpsd"] = tpsd
                
    return burst_props


def clean_bprops(bprops):
    bprops_new = bt.remove_short_burst(bprops)
    bprops_new = bt.concatenate_burst(bprops_new, amp_range)
    return bprops_new
                
if __name__ == "__main__":
    main(**vars(build_parser().parse_args()))