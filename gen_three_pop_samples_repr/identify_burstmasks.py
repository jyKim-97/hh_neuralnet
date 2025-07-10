"""
identify burst masks in the burst properties
Need to run after extract_burstprobs.py
"""

import os
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import argparse

import sys
sys.path.append("/home/jungyoung/Project/hh_neuralnet/include/")
import burst.burst_tools as bt

fdir_raw_signal = "/home/jungyoung/Project/hh_neuralnet/gen_three_pop_samples_repr/data"
famp_range = "/home/jungyoung/Project/hh_neuralnet/extract_osc_motif/data/osc_motif/amp_range_set.pkl"
bprops_dir = "/home/jungyoung/Project/hh_neuralnet/gen_three_pop_samples_repr/postdata/bprops"
date_str = "250710"


def build_parser():
    parser = argparse.ArgumentParser(description="Identify burst masks in the burst properties")
    parser.add_argument("--cid", type=int, required=True, help="Regime ID (integer)")
    parser.add_argument("--tmin_discon", type=float, default=0.1, help="Minimum time to consider as disconnected (in seconds)")
    parser.add_argument("--fout", type=str, default=None, help="File name to save the motif info")
    parser.add_argument("--export_spectrum", action='store_true', help="Export spectrum of the motif")
    parser.add_argument("--export_spectrum_dir", type=str, default=None, help="Directory to save the spectrum data")
    return parser


def main(cid=None, tmin_discon=0.1, fout=None,
         export_spectrum=False, export_spectrum_dir=None):
    
    amp_range = load_amprange(cid)
    bprops, meta = load_bprops(cid)
    bprops = preprocess_bprops(bprops, amp_range)
    
    # identify burst masks
    print("Identifying burst masks for cid=%d..."%(cid))
    motif_info = combine_bprops(bprops, tmin_discon=tmin_discon)
    metainfo = build_meta(meta, amp_range)
    print("Done identification")
    
    # save motif
    if fout is None: fout = "./motif_info_%d.pkl"%(cid)
    motif_dict = {"winfo": motif_info, "metainfo": metainfo}
    with open(fout, "wb") as fp:
        pkl.dump(motif_dict, fp)
    
    # get spectrum
    if export_spectrum:
        if export_spectrum_dir is None:
            raise ValueError("export_spectrum_dir must be specified if export_spectrum is True")
        save_spectrum(cid, motif_dict, export_spectrum_dir)


def load_bprops(cid):
    fname = os.path.join(bprops_dir, "bprops_%d.pkl"%(cid))
    with open(fname, "rb") as fp:
        data = pkl.load(fp)
    bprops = data["burst_props"]
    return bprops, data["attrs"]
    

def preprocess_bprops(bprops, amp_range):
    ramp_range = bt.resize_amp_range(amp_range)
    for npop in range(2):
        bt.identify_burst_fid(bprops[npop], ramp_range[npop])
        bprops[npop]["bf_range"] = ramp_range[npop]        
    return bprops


def load_amprange(cid):
    with open(famp_range, "rb") as fp:
        amp_range = pkl.load(fp)["amp_range_set"][cid-1]
    return amp_range


def find_consecutive_states(state_id):
    # find [start, end) with state
    N = len(state_id)
    segments = []
    start = 0
    while start < N:
        curr = state_id[start]
        end = start
        while end < N and state_id[end] == curr:
            end += 1
        segments.append((start, end, curr))
        start = end
    return segments


def fill_transient_included_states(state_id, nbin_discon):
    state_id = np.asarray(state_id)
    N = len(state_id)
    output = state_id.copy()

    segments = find_consecutive_states(state_id)

    i = 0
    while i < len(segments):
        group = []
        while i < len(segments):
            start, end, state = segments[i]
            seg_len = end - start
            if state == 0 or seg_len >= nbin_discon:
                break
            group.append((i, start, end, state))
            i += 1

        if group:
            # Check if we can merge into next (forward)
            last_idx = group[-1][0]
            if last_idx + 1 < len(segments):
                _, _, next_state = segments[last_idx + 1]
                if next_state != 0 and all((s | next_state) == next_state for _, _, _, s in group):
                    for _, start, end, _ in group:
                        output[start:end] = next_state
                    continue

            # Check if we can merge into previous (backward)
            first_idx = group[0][0]
            if first_idx - 1 >= 0:
                _, _, prev_state = segments[first_idx - 1]
                if prev_state != 0 and all((s | prev_state) == prev_state for _, _, _, s in group):
                    for _, start, end, _ in group:
                        output[start:end] = prev_state
                    continue

            # NEW: Try to merge internally if all short and compatible
            union_state = 0
            for _, _, _, s in group:
                union_state |= s
            if all((s | union_state) == union_state for _, _, _, s in group):
                for _, start, end, _ in group:
                    output[start:end] = union_state
                continue

        i += 1
    return output


def combine_bprops(bprops, num_cycle_reconnect=2, tmin_discon=0.1):
    
    if "burst_fid" not in bprops[0].keys():
        raise ValueError("burst_fid not found in bprops")
    
    if "bf_range" not in bprops[0].keys():
        raise ValueError("bf_range not found in bprops")
    
    N = len(bprops[0]["tpsd"])
    Npair = 4
    dt = bprops[0]["tpsd"][10] - bprops[0]["tpsd"][9]
    max_trials = max([max(bprops[i]["id_trial"]) for i in range(len(bprops))])
    motif_info = [[] for _ in range(16)] # 16 possible states
    
    for nt in range(max_trials+1):
        is_burst = np.zeros((Npair, N), dtype=bool)
        for npop in range(len(bprops)):
            for i in range(len(bprops[npop]["id_trial"])):
                if bprops[npop]["id_trial"][i] > nt:
                    break
                if bprops[npop]["id_trial"][i] == nt:
                    idf = bprops[npop]["burst_fid"][i]
                    nr = bprops[npop]["burst_range"][i]
                    is_burst[2*npop+idf, nr[0]:nr[1]] = True
            
        # connect consecutive states
        # num_recon = int(t_reconnect / ())
        for n in range(Npair):
            num_discon = 0
            buf_discon_start = -1
            npop, idf = divmod(n, 2)
            f0 = bprops[npop]["bf_range"][idf,:].mean()
            num_recon = int(num_cycle_reconnect/f0/dt)
            for i in range(N):
                if is_burst[n, i]:
                    if num_discon != 0:
                        if num_discon <= num_recon and buf_discon_start != -1:
                            is_burst[n, buf_discon_start:i] = True
                    num_discon = 0
                    buf_discon_start = -1
                else:
                    if num_discon == 0:
                        buf_discon_start = i                    
                    num_discon += 1
        
        # find partially overlapping states
        # is_cut = np.zeros(N, dtype=bool)
        # for i in range(Npair):
        #     for j in range(i+1, Npair):
        #         is_cut_ij = detect_partial_overlap_pair(is_burst[i].copy(), is_burst[j].copy(), p_partial=p_partial)
        #         is_cut = is_cut | is_cut_ij
                
        # extract state ID
        state_id = is_burst[0] + 2*is_burst[1] + 4*is_burst[2] + 8*is_burst[3]
        # state_id[is_cut] = 0
        
        # fill state id
        nbin_discon = int(tmin_discon / dt)
        filled_id = fill_transient_included_states(state_id, nbin_discon)

        # build state_info
        motif = find_consecutive_states(filled_id)
        for start, end, state in motif:
            if state == 0 and end - start < nbin_discon:
                continue
            
            motif_info[state].append((nt, [bprops[0]["tpsd"][start], bprops[0]["tpsd"][end-1]]))
        
    return motif_info


def build_meta(meta, amp_range):
    metainfo = dict(
        amp_range=amp_range, 
        wbin_t=meta["wbin_t"],
        mbin_t=meta["mbin_t"],
        reverse=False,
        )
    metainfo["last-updated"] = date_str
    return metainfo


def save_spectrum(cid, motif_dict, export_dir):
    import hhsignal
    import hhtools
    from tqdm import tqdm
    
    fs = 2000
    meta = motif_dict["metainfo"]
    
    nspec = np.zeros(16)
    spectrum_avg, spectrum_var = None, None
    
    summary_obj = hhtools.SummaryLoader(fdir_raw_signal)
    for nt in tqdm(range(summary_obj.num_controls[1])):
        detail_data = summary_obj.load_detail(cid-1, nt)
        
        # compute stfft
        psd_norm_set = []
        for i in range(1, 3):
            psd, fpsd, tpsd = hhsignal.get_stfft(detail_data["vlfp"][i], detail_data["ts"], fs, mbin_t=meta["mbin_t"], wbin_t=meta["wbin_t"])
            psd_norm_set.append(psd - psd.mean(axis=1, keepdims=True))
        psd_norm_set = np.array(psd_norm_set)
        
        if nt == 0:
            spectrum_avg = np.zeros((16, 2, len(fpsd)))
            spectrum_var = np.zeros((16, 2, len(fpsd)))
        
        for i in range(16): # 16 possible states    
            for n, tr in motif_dict["winfo"][i]:
                if n > nt:
                    break
                elif n < nt:
                    continue
                
                idt = (tpsd >= tr[0]) & (tpsd <= tr[1])
                p = psd_norm_set[:,:,idt]
                
                spectrum_avg[i] += p.sum(axis=2)
                spectrum_var[i] += (p**2).sum(axis=2)
                nspec[i] += np.sum(idt)
    
    nspec[nspec == 0] = 1
    spectrum_avg /= nspec[:, np.newaxis, np.newaxis]
    spectrum_var /= nspec[:, np.newaxis, np.newaxis]
    spectrum_var -= spectrum_avg**2
    
    idf = (fpsd >= 0) & (fpsd < 100)
    spectrum_avg = spectrum_avg[:, :, idf]
    spectrum_var = spectrum_var[:, :, idf]
    fpsd = fpsd[idf]
                
    # save spectrum
    with open(os.path.join(export_dir, "spectrum_%d.pkl"%(cid)), "wb") as fp:
        pkl.dump({
            "spectrum_avg": spectrum_avg,
            "spectrum_var": spectrum_var,
            "npoints": nspec,
            "fpsd": fpsd,
            "cid": cid,
            "date": meta["last-updated"]
        }, fp)
            
        
if __name__ == "__main__":
    main(**vars(build_parser().parse_args()))
    
    # main(
    #     cid=7,
    #     tmin_discon=0.1,
    #     fout="./postdata/mfop/motif_info_7.pkl",
    #     export_spectrum=True,
    #     export_spectrum_dir="./postdata/mfop/spec_summary"
    # )