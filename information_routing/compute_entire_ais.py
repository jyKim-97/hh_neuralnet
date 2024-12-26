# import computeTE4 as ct
import computeAIS as ca

import os
import sys
sys.path.append("/home/jungyoung/Project/hh_neuralnet/extract_osc_motif")
import utils

nsample = 100
target = "mua"
seed = 42

# fdir = "./data/te_2d_newmotif_reverse"
fdir = "./data/ais_newmotif"

def main():
    for nc in range(10): # (added without connection 3)
        cid = nc + 1
        for wid in range(16):
            winfo, _ = utils.load_osc_motif(cid, wid, tag="")
            if len(winfo) < 100: continue

            fout = os.path.join(fdir, "ais_%d%02d.pkl"%(cid, wid))
            ca.main(cid=cid, wid=wid, nsample=nsample,
                    target=target, fout=fout,
                    tlag_min=0.5, tlag_step=0.5, tlag_max=40)
            
            print("Saved into %s"%(fout))
            
            
if __name__=="__main__":
    main()
