import numpy as np
import burst_tools as bt
import pickle as pkl
from tqdm import trange
from argparse import ArgumentParser

import sys
sys.path.append("/home/jungyoung/Project/hh_neuralnet/include")
import hhtools
import hhsignal


# hard fix parameters
fs = 2000
mbin_t = 0.1
wbin_t = 1
flim = (10, 100)


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--fname_th", required=True)
    parser.add_argument("--fout", required=True)
    return parser


def load_psd_threshold(fname):
    with open(fname, "rb") as fp:
        th_obj = pkl.load(fp)
    return th_obj["th_psd"], th_obj["fdir_data"]


def extract_burst(fname_th):

    def cut_start(signal, t_sig):
        idt = t_sig >= 0.5 # t_sig (s)
        return signal[idt], t_sig[idt]

    th_psd, fdir_data = load_psd_threshold(fname_th)
    summary_obj = hhtools.SummaryLoader(fdir_data)

    burst_info = {"burst_f": [],
                  "burst_range": [],
                  "burst_amp": [],
                  "cluster_id": [],
                  "pop_type": []}

    for n in trange(summary_obj.num_total):
        detail_data = summary_obj.load_detail(n)
        n1 = n // summary_obj.num_controls[-1]
        nc = summary_obj.controls["cluster_id"][n1]

        for i in range(2):
            vlfp, t = cut_start(detail_data["vlfp"][i+1], detail_data["ts"])
            psd, fpsd, tpsd = hhsignal.get_stfft(vlfp, t, fs,
                                                 mbin_t=mbin_t, wbin_t=wbin_t, frange=flim)
            im_bin = psd >= th_psd[n, i]
            im_class = bt.find_blob(im_bin)
            burst_f, burst_range, burst_amp = bt.extract_burst_attrib(psd, fpsd, im_class)

            burst_info["burst_f"].append(burst_f)
            burst_info["burst_range"].append(tpsd[burst_range.astype(int)])
            burst_info["burst_amp"].append(burst_amp)
            burst_info["cluster_id"].append(nc)
            burst_info["pop_type"].append(i)

    return burst_info
    

def main(fname_th, fout):
    burst_info = extract_burst(fname_th)
    
    print("Export burst info to %s"%(fout))
    with open(fout, "wb") as fp:
        pkl.dump(burst_info, fp)


if __name__ == "__main__":
    main(**vars(build_argument_parser().parse_args()))
