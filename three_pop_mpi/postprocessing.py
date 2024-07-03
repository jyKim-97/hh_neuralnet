# Post process after run main.c: This code calculate AC/CC and FFT
import numpy as np
import sys
from tqdm import tqdm


# Add custom modules
sys.path.append("/home/jungyoung/Project/hh_neuralnet/include/")
import hhtools
import hhsignal

srate = 2000
teq = 0.5
wbin_t = 1 # s

seed = 20000
prominence = 0.05
num_itr = 1


def load_simualtion(prefix):
    obj_total = hhtools.SummaryLoader(prefix)
    print("Load %s Done"%(prefix))
    print("size of the array: ", obj_total.num_controls)
    print("Control params: ", obj_total.controls.keys())
    return obj_total


def postprocess_all(obj_total):
    # initialize storage list
    # keys = ["ac2_large_p", "ac2_lag", "cc_p", "cc_lag", "fpeak"]
    keys = ["ac2p_large", "ac2p_1st", "fpeak", "ac2t_large", "ac2t_1st", "cc1p_large", "cc1t_large"]

    # sz = list(np.shape(obj_total)) + [3]
    l = 1
    for s in obj_total.num_controls[:-1]:
        l *= s
    order_dyna = {k: {"mean": np.zeros(l*3), "std": np.zeros(l*3)} for k in keys}
    order_dyna["cc1p_large"]["mean"] = np.zeros(l)
    order_dyna["cc1p_large"]["std"] = np.zeros(l)
    order_dyna["cc1t_large"] = np.zeros(l * obj_total.num_controls[-1] * num_itr) # average X 
    order_dyna["prefix"] = obj_total.fdir

    num_total_itr = obj_total.num_controls[-1] * num_itr
    for nt in tqdm(range(obj_total.num_total)):
        data = obj_total.load_detail(nt)
        nt_ind = nt // obj_total.num_controls[-1]
        for n in range(num_itr):
            xsub, _ = pick_sample_data(data)
            for i in range(3):
                # get AC
                nid = 3*nt_ind + i
                ac2_large_p, tlag_large_p, ac2_1p, tlag_1p = get_ac2_peak(xsub[i], prominence=prominence)
                order_dyna["ac2p_large"]["mean"][nid]  += ac2_large_p
                order_dyna["ac2p_large"]["std"][nid]   += ac2_large_p*ac2_large_p
                order_dyna["ac2t_large"]["mean"][nid]  += tlag_large_p
                order_dyna["ac2t_large"]["std"][nid]   += tlag_large_p*tlag_large_p
                order_dyna["ac2p_1st"]["mean"][nid]    += ac2_1p
                order_dyna["ac2p_1st"]["std"][nid]     += ac2_1p*ac2_1p
                order_dyna["ac2t_1st"]["mean"][nid]    += tlag_1p
                order_dyna["ac2t_1st"]["std"][nid]     += tlag_1p*tlag_1p

                # get frequency
                fp = hhsignal.get_frequency_peak(xsub[i], fs=2000)
                order_dyna["fpeak"]["mean"][nid] += fp
                order_dyna["fpeak"]["std"][nid]  += fp*fp
            
            # get CC
            cc_p, tlag_p = get_cc_peak(xsub[1], xsub[2], prominence=prominence)
            order_dyna["cc1p_large"]["mean"][nt_ind]  += cc_p
            order_dyna["cc1p_large"]["std"][nt_ind]   += cc_p*cc_p
            order_dyna["cc1t_large"][nt*num_itr + n] = tlag_p

    shape = list(obj_total.num_controls[:-1]) + [3]
    for k in keys[:-1]: # run except cc1t_large
        order_dyna[k]["mean"] /= num_total_itr
        order_dyna[k]["std"]  /= num_total_itr
        order_dyna[k]["std"] = np.sqrt(order_dyna[k]["std"] - order_dyna[k]["mean"]**2)
    
        if k == "cc1p_large":
            order_dyna[k]["mean"] = np.reshape(order_dyna[k]["mean"], shape[:-1])
            order_dyna[k]["std"]  = np.reshape(order_dyna[k]["std"], shape[:-1])
        else:
            order_dyna[k]["mean"] = np.reshape(order_dyna[k]["mean"], shape)
            order_dyna[k]["std"]  = np.reshape(order_dyna[k]["std"], shape)

    shape_t = list(obj_total.num_controls)
    shape_t[-1] *= num_itr
    order_dyna["cc1t_large"] = np.reshape(order_dyna["cc1t_large"], shape_t)
    
    return order_dyna
    

def pick_sample_data(data):
    tmax = data["ts"][-1]
    t0 = np.random.rand() * (tmax-teq-wbin_t-0.1) + teq 
    n0 = int(t0 * srate)
    n1 = n0 + int(wbin_t * srate)

    vlfp = [data["vlfp"][0][n0:n1],
            data["vlfp"][1][n0:n1],
            data["vlfp"][2][n0:n1]]
    return vlfp, t0


def get_ac2_peak(x, prominence=0.01):
    # Need to return 2nd peak lag, mag
    ac, tlag = hhsignal.get_correlation(x, x, srate, max_lag=0.2)
    idp_1st, idp_large = hhsignal.detect_peak(ac, prominence=prominence, mode=3)
    return ac[idp_large[1]], tlag[idp_large[1]], ac[idp_1st[1]], tlag[idp_1st[1]]


def get_cc_peak(x, y, prominence=0.01):
    # Need to return 1nd peak lag, mag
    cc, tlag = hhsignal.get_correlation(x, y, srate, max_lag=0.2)
    idp = hhsignal.detect_peak(cc, prominence=prominence, mode=0)
    return cc[idp[0]], tlag[idp[0]]


if __name__=="__main__":
    import pickle as pkl
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help="prefix of your simulated data", required=True)
    parser.add_argument("--fout", help="output file name", required=True)
    parser.add_argument("-n", help="the # of iteration", type=int)

    args = parser.parse_args()

    prefix = args.prefix
    fout = args.fout
    num_itr = args.n

    np.random.seed(seed)

    # read data
    obj_total = load_simualtion(prefix)
    print("Print to %s"%(fout))

    # run 
    order_dyna = postprocess_all(obj_total)

    # save
    if ".pkl" not in fout:
        fout += ".pkl"
    with open(fout, "wb") as fp:
        pkl.dump(order_dyna, fp)

    print("Done")