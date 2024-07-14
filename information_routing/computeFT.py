import numpy as np
import sys
import pickle as pkl
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import hhsignal
from tqdm import trange, tqdm
import utils

# import oscdetector as od
import argparse

# from functools import partial
# from multiprocessing import Pool

# import computeTE2 as ct


# set const
_num_process = 4
# _teq = 1
srate = 2000
wbin_t = 0.5
mbin_t = 0.1

# tag = ""
tag = "_mfast"

summary_obj = hhtools.SummaryLoader("/home/jungyoung/Project/hh_neuralnet/gen_three_pop_samples_repr/data"+tag, load_only_control=True)


def build_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", help="cluster id", required=True, type=int)
    parser.add_argument("--wid", help="word id", required=True, type=int)
    parser.add_argument("--npoint", help="number of bins to discretize the voltage", default=1e5, type=int)
    parser.add_argument("--nboot", default=100, type=int)
    parser.add_argument("--fout", default=None)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def main(cid=5, wid=10, npoint=10000, nboot=100, fout=None, seed=42):
    np.random.seed(seed)
    
    # load oscillation motif informaiton
    winfo, update_date = utils.load_osc_motif(cid, wid, reverse=False)
    
    # collect voltage segments
    spec_set, fpsd = collect_ft_chunk(cid, winfo)
    npoint_spec = int(npoint / (srate * mbin_t))
    
    # sampling
    np.random.seed(seed)
    spec_boot = compute_spec_boot(spec_set, nboot, npoint_spec)
    spec_boot = np.swapaxes(spec_boot, 1, 2)
    
    if fout is None:
        fout = "./data/spec%s_%d%02d.pkl"%(tag, cid ,wid)
    
    print("Result is saved to %s"%(fout))
    with open(fout, "wb") as fp:
        pkl.dump({"info": {"cid": cid, "wid": wid, "wbin_t": wbin_t, "mbin_t": mbin_t,
                           "npoint": npoint, "nboot": nboot, "seed": seed},
                  "fpsd": fpsd, "spec_boot": spec_boot}, fp)    
    
    
def collect_ft_chunk(cid, winfo):
    
    pre_sos = hhsignal.get_sosfilter([1, 150], srate) # preprocessing
    
    spec_set = []
    nitr_prv = -1
    for i in trange(len(winfo), desc="load spec samples"):
        nitr = winfo[i][0]
        tl   = winfo[i][1]
        
        if nitr != nitr_prv: # load detail data
            detail_data = summary_obj.load_detail(cid-1, nitr)
            v1 = hhsignal.filt(detail_data["vlfp"][1], pre_sos)
            v2 = hhsignal.filt(detail_data["vlfp"][2], pre_sos)
            
            v1 = (v1 - v1.mean())/v1.std()
            v2 = (v2 - v2.mean())/v2.std()
            
            psd_set = [[], []]
            t = detail_data["ts"]
            # psd_set[0], fpsd, tpsd = hhsignal.get_stfft(v1, t, srate, mbin_t=mbin_t, wbin_t=wbin_t, frange=(2, 100))
            # psd_set[1], fpsd, tpsd = hhsignal.get_stfft(v2, t, srate, mbin_t=mbin_t, wbin_t=wbin_t, frange=(2, 100))
            psd_set, fpsd, tpsd = utils.compute_stfft_all([v1, v2], t, frange=(10, 100),
                                                          mbin_t=mbin_t, wbin_t=wbin_t, srate=srate)
            
            # normalize
            psd_set[0] = psd_set[0] - psd_set[0].mean(axis=1)[:, None]
            psd_set[1] = psd_set[1] - psd_set[1].mean(axis=1)[:, None]
            
            nitr_prv = nitr
        
        if (tl[0] < 1) or (tl[1] > tpsd[-1]-1):
            continue
            
        # collect voltage segments        
        nr = ((tl - tpsd[0]) / mbin_t).astype(int)
        
        psd_sample = np.array([psd_set[0][:, nr[0]:nr[1]], psd_set[1][:, nr[0]:nr[1]]])
        psd_sample = np.swapaxes(psd_sample, 0, 2)
        
        spec_set.extend(psd_sample)
        
        # spec_set.append([psd_set[0][:, nr[0]:nr[1]].mean(axis=1),
        #                psd_set[1][:, nr[0]:nr[1]].mean(axis=1)])
        
    return np.array(spec_set), fpsd


def compute_spec_boot(spec_set, nboot, npoint_spec):
    
    spec_boot = []
    for _ in trange(nboot):
        idx_sample = np.random.randint(0, len(spec_set), npoint_spec)
        spec_boot.append(spec_set[idx_sample].mean(axis=0))
    
    return spec_boot


if __name__ == "__main__":
    main(**vars(build_arg_parse().parse_args()))