import numpy as np
from numba import njit

@njit
def align_spike_single(step_r, step_n0, nspk_edges, nsearch_start=0):
    idr = nsearch_start
    dnspk = nspk_edges[1] - nspk_edges[0]
    num_spk_r = np.zeros(len(nspk_edges)-1, dtype=np.uint8)
    
    if (idr >= len(step_r)) or step_r[-1] < step_n0+nspk_edges[0] or step_r[0] > step_n0+nspk_edges[-1]:
        return num_spk_r, idr
    
    while step_r[idr] >= step_n0+nspk_edges[0]:
        idr -= 1
        if idr <= 0:
            idr = 0
            break

    while step_r[idr] < step_n0+nspk_edges[0]:
        idr += 1
        if idr == len(step_r):
            return num_spk_r, idr
    
    while step_r[idr] < step_n0+nspk_edges[-1]:
        dn = step_r[idr] - step_n0        
        if dn >= nspk_edges[-1]:
            break
        
        idn = (dn - nspk_edges[0])//dnspk
        num_spk_r[idn] = 1
        
        idr += 1
        if idr == len(step_r):
            break
    
    return num_spk_r, idr


@njit
def compute_spike_resp(nstep_spk_t, nstep_spk_r, nstep_edges):
    n_search = 0
    num_spk_set = np.zeros((len(nstep_spk_t), len(nstep_edges)-1), dtype=np.uint8)
    # num_spk_set = []
    for n, nstep in enumerate(nstep_spk_t):
        num_spk_set[n], n_search = align_spike_single(nstep_spk_r, nstep, nstep_edges, nsearch_start=n_search)
    
    return num_spk_set
    


def convert_spkvec(step_spk, dt, tmax, srate=2000): # seconds
    N = int(tmax * srate) + 1
    spkvec = np.zeros(N)
    for nt in step_spk:
        n = int(nt*dt*srate)
        spkvec[n] += 1
    return spkvec


# load detail info includeing TR and TD
def load_detail(summary_obj, nc, nt, load_ntk=False):
    assert nt < summary_obj.num_controls[0] * summary_obj.num_controls[2]
    
    def _read_tr_info(prefix):
        sel_tr = [[], []] # T, R, delay
        sel_td = []
        with open(prefix+"_trinfo.txt", "r") as fp:
            l = fp.readline() 
            l = fp.readline()
            while l:
                val = l[:-1].split(",")
                sel_tr[0].append(int(val[0]))
                sel_tr[1].append(int(val[1]))
                sel_td.append(int(val[2]))
                l = fp.readline()
        return sel_tr, sel_td
    
    i = nt // summary_obj.num_controls[2]
    j = nt % summary_obj.num_controls[2]
    
    detail = summary_obj.load_detail(i, nc, j)
    sel_tr, sel_td = _read_tr_info(detail["prefix"])
    detail['num_trans'] = len(sel_td)
    detail["sel_tr"] = sel_tr
    detail["sel_td"] = sel_td
    
    if load_ntk:
        detail["adj_out"] = read_adjout(detail["prefix"], len(detail["step_spk"]))
        
    return detail


def read_adjout(prefix, N=2000):
    fname = prefix + "_adj.txt"
    adj_out = [[] for _ in range(N)]
    with open(fname, "r") as fp:
        l = fp.readline()
        while l:        
            lsplit = l.split("<-")
            node_in  = int(lsplit[0])
            node_out = [int(n) for n in lsplit[1].split(",")[:-1]]
            
            l = fp.readline()
            if node_in == -1:
                continue
            
            for n in node_out:
                if n == -1: continue
                adj_out[n].append(node_in)
        
    for n in range(N):
        adj_out[n] = np.array(adj_out[n], dtype=np.uint32)
    
    return adj_out


import pickle as pkl

def load_pickle(fname):
    with open(fname, "rb") as fp:
        return pkl.load(fp)