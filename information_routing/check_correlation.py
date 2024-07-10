import numpy as np
import matplotlib.pyplot as plt

import visu
import utils

import sys
sys.path.append("../include")
import hhsignal
from tqdm import tqdm


tmax_lag = 0.04
srate = 2000

tp_labels = ("F - F", "F - S", "S - S")
cs = (visu.cs[0], "k", visu.cs[1])
yl = (-0.8, 1.2)

popt = dict(linestyle='--', alpha=0.5, linewidth=1)
topt = dict(va="center", ha="center", fontsize=8)

for cid, wid in utils.cw_pair:
    mua_chunk = utils.collect_chunk(cid, wid, target="mua", nadd=int(0.3*srate), nequal_len=int(1.2*srate))

    N = mua_chunk.shape[0]
    nmax_lag = int(2*tmax_lag*srate) + 1
    cout = np.zeros((N, 3, nmax_lag))
    for nid in tqdm(range(N)):
        is_in = ~np.isnan(mua_chunk[nid, 0])
        
        x = mua_chunk[nid, 0][is_in]
        y = mua_chunk[nid, 1][is_in]
        
        x = (x - x.mean())/x.std()
        y = (y - y.mean())/y.std()
        
        cout[nid,0,:], _    = hhsignal.get_correlation(x, x, srate, max_lag=tmax_lag)
        cout[nid,1,:], _    = hhsignal.get_correlation(x, y, srate, max_lag=tmax_lag)
        cout[nid,2,:], tlag = hhsignal.get_correlation(y, y, srate, max_lag=tmax_lag)

    tlag *= 1e3
    
    prefix = "./figs/correlation/corr_%d%02d"%(cid, wid)
    
    # Draw figure
    print("cid=%d, wid=%02d"%(cid, wid))
    fig = plt.figure(figsize=(4, 3), dpi=120)
    for n in range(3):
        visu.draw_with_err(tlag, cout[:,n,:], c=cs[n], label=tp_labels[n])
        
        y = cout[:,n,:].mean(axis=0)
        idp = hhsignal.detect_peak(y)
        
        if n%2 == 0: # auto-
            if len(idp) >= 2:
                tp = tlag[idp[2]]
                plt.vlines(tp, yl[0], yl[1], color=cs[n], **popt)
                print(f"tau: {tp:.1f} ms")
            else:
                print(f"tau:  ms")
            
        else: # cross-
            tp_set = tlag[idp[:3]]
            plt.vlines(tp_set, yl[0], yl[1], color=cs[n], **popt)
            
            print("tau: ", end="")
            for tp in tp_set:
                print(f"{tp:.1f}, ", end="")
            print("ms")
            
    plt.legend(fontsize=10, loc="upper left", edgecolor='none')
    plt.xlabel(r"$\tau$ (ms)", fontsize=14)
    plt.ylabel(r"$Cross-correlation$", fontsize=12)
    plt.yticks(np.arange(-0.5, 1.1, 0.5))
    plt.ylim(yl)
    
    fname = prefix + ".png"
    
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    
    # plt.show()