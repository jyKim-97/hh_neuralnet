import subprocess
import os
import argparse

cw_pair = [
    [1, 0], [1, 2], [1, 4], [1, 6],
    [2, 0], [2, 2], [2, 8], [2, 10],
    [3, 0], [3, 2], [3, 4], [3, 6],
    [4, 0], [4, 2], [4, 5], [4, 7],
    [5, 0], [5, 4], [5, 10], [5, 14],
    [6, 0], [6, 4], [6, 10], [6, 14],
    [7, 0], [7, 10],
    [8, 0], [8, 2], [8, 13], [8, 15]
]


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n0", type=int, default=0)
    parser.add_argument("--n1", type=int, default=1000)
    return parser

# wid_set = [
#     [0, 2, 4, 6],
#     [0, 2, 8, 10],
#     [0, 2, 4, 6],
#     [0, 2, 5, 7],
#     [0, 4, 10, 14],
#     [0, 4, 10, 14],
#     [0, 10],
#     [0, 2, 13, 15],
# ]

# NBIN=30
# NPOINT=500000
# TAG="./data/te_full2"
TAG="./data/te_2d_mua_2"
RUN="python computeTE4.py"
nhist = 1
# method = "full" # naive, 
# method = "full"
method = "2d"
target = "mua"

# TAG="./data/spec/"
# RUN="python computeFT.py"

ntrue = 100
nsurr = 1000

def main(n0=0, n1=0):
    n1 = min(n1, len(cw_pair))
    
    for cid, wid in cw_pair[n0:n1]:
        
        # com = f"{RUN} --cid={cid} --wid={wid} --ntrue={ntrue} --nsurr={nsurr} --method={method} --target={target} --tlag_min=1 --tlag_max=40 --tlag_step=1 --nhist={nhist} --fout={TAG}/te_{cid}{wid:02d}.pkl"
        com = f"{RUN} --cid={cid} --wid={wid} --ntrue={ntrue} --nsurr={nsurr} --method={method} --target={target} --nhist={nhist} --fout={TAG}/te_{cid}{wid:02d}.pkl"
        # com = f"{RUN} --cid={cid} --wid={wid} --npoint=100000 --fout={TAG}/spec_{cid}{wid:02d}.pkl"
        # com = f"{RUN} --cid={cid} --wid={wid} --nsamples=1000 --ntrue=100 --nsurr=100 --method={method} --nhist={nhist} --fout={TAG}/te_{cid}{wid:02d}%d%02d.pkl"
        # com = f"{RUN} --cid={cid} --wid={wid} --nbin={NBIN} --npoint={NPOINT} --fout={TAG}/te_%d%02d.pkl"%(cid, wid)
        print(com)
        os.system(com)


if __name__=="__main__":
    main(**vars(build_args().parse_args()))