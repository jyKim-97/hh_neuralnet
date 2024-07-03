import numpy as np
import sys
import pickle as pkl
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import hhsignal
from tqdm import trange

import oscdetector as od
import hhinfo

_teq = 0.5


srate = 2000
# vrange = [-70, -50]
vrange = [-3, 3]
wbin_t, mbin_t = 0.5, 0.01
tlag_max = 15
srate_d = 2000
nbin = 51

summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data")
    

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


def get_normv(detail_data):
    def _norm(x):
        return (x - x.mean())/x.std()
    
    vf, _ = hhsignal.get_eq_dynamics(detail_data["vlfp"][1], detail_data["ts"], _teq)
    vs, _ = hhsignal.get_eq_dynamics(detail_data["vlfp"][2], detail_data["ts"], _teq)
    
    return _norm(vf), _norm(vs)


def compute_te_on_motif(cid, amp_range, nlag_max, mbin_t=0.01, wbin_t=0.5, q=80, min_len=2, cat_th=2):
    # te = np.zeros((16, 2, nlag_max))
    h_motif = [[] for _ in range(16)]
    te_motif = [[] for _ in range(16)]
    len_motif = [[] for _ in range(16)]
    
    for i in trange(200, desc="cid: %d"%(cid)):
        detail_data = summary_obj.load_detail(cid-1, i)
        psd_set, fpsd, tpsd = od.compute_stfft_all(detail_data, mbin_t=mbin_t, wbin_t=wbin_t)
        
        # detect oscillation motif
        words = od.compute_osc_bit(psd_set[1:], fpsd, tpsd, amp_range, q=q, min_len=min_len, cat_th=cat_th)
        osc_motif = od.get_motif_boundary(words, tpsd)
        
        vf, vs = get_normv(detail_data)
        vf_d = digitize_v(downsample(vf, srate, srate_d), vrange, nbin)
        vs_d = digitize_v(downsample(vs, srate, srate_d), vrange, nbin)
        
        # vf, _ = hhsignal.get_eq_dynamics(detail_data["vlfp"][1], detail_data["ts"], _teq)
        # vs, _ = hhsignal.get_eq_dynamics(detail_data["vlfp"][2], detail_data["ts"], _teq)
        
        # add baseline period
        bd = od.get_boundary(words == 0)
        for i in range(len(bd)):
            osc_motif.append({"id": 0, "range": tpsd[bd[i]]})
        
        # compute TE on oscillation motif
        for n in range(len(osc_motif)):
            ido = osc_motif[n]["id"]
            nr = ((osc_motif[n]["range"] - _teq) * srate_d).astype(int)
            
            if nr[1] - nr[0] < nlag_max or nr[0] < 0:
                continue
            
            x = vf_d[nr[0]:nr[1]]
            y = vs_d[nr[0]:nr[1]]
            
            te_xy, hy = hhinfo.compute_te(x, y, nbin, nlag_max) # fpop -> spop
            te_yx, hx = hhinfo.compute_te(y, x, nbin, nlag_max) # fpop -> spop
            te_motif[ido].append([te_xy, te_yx])
            h_motif[ido].append([hy, hx])
            len_motif[ido].append(nr[1]-nr[0]) # number of sample points
    
    for ido in range(16):
        te_motif[ido] = np.array(te_motif[ido])
            
    return te_motif, h_motif, len_motif


def downsample(x, srate, srate_d):
    n = int(srate/srate_d)
    return x[::n]


def digitize_v(v, vrange, nbin):
    v[v < vrange[0]] = vrange[0]
    v[v > vrange[1]] = vrange[1]
    
    de = (vrange[1] - vrange[0])/(nbin-1)
    e = np.arange(vrange[0]-de/2, vrange[1]+de, de)
    
    return np.digitize(v, e)    


nlag_max = int(tlag_max * 1e-3 * srate_d)
te_motif8, h_motif8, len_motif8 = compute_te_on_motif(8, amp_range_set[7], nlag_max, q=75)

with open("./data/te_tmp_8.pkl", "wb") as fp:
    pkl.dump({"te": te_motif8,
              "h": h_motif8,
              "len": len_motif8}, fp)

# te_motif5, h_motif5, len_motif5 = compute_te_on_motif(5, amp_range_set[4], nlag_max, q=75)

# with open("./data/te_tmp_5.pkl", "wb") as fp:
#     pkl.dump({"te": te_motif5,
#               "h": h_motif5,
#               "len": len_motif5}, fp)
    
    # pkl.dump({"te": [te_motif5, te_motif8],
    #           "H": [h_motif5, h_motif8],
    #           "len": [len_motif5, len_motif8]}, fp)