import numpy as np
from tqdm import tqdm, trange
from multiprocessing import Pool
import pickle as pkl
from typing import List, Tuple
# import warnings
from functools import partial

import os
import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhsignal
import hhtools

tag = "_mfast"
default_dir = "/home/jungyoung/Project/hh_neuralnet/gen_three_pop_samples_repr/data"+tag
prefix_motif_dir = "/home/jungyoung/Project/hh_neuralnet/information_routing/data/osc_motif"+tag

# default_dir = "/home/jungyoung/Project/hh_neuralnet/gen_three_pop_samples_repr/data"
# prefix_motif_dir = "/home/jungyoung/Project/hh_neuralnet/information_routing/data/osc_motif/"

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


def set_default_dir(fdir):
    global default_dir
    default_dir = fdir


def par_func(f, arg_set, num_process, desc=""):
    N = len(arg_set)
    result_set = []
    if num_process == 1:
        for arg in tqdm(arg_set, desc=desc):
            result_set.append(f(arg))
    else:
        pool = Pool(processes=num_process)
        for result in tqdm(pool.imap(f, iterable=arg_set), total=N, desc=desc):
            result_set.append(result)
    
    return result_set


def load_osc_motif(cid, wid, reverse=False, verbose=False):
    # fname = "./data/osc_motif/motif_info%s_%d"%(tag, cid)
    fname = os.path.join(prefix_motif_dir, "motif_info_%d"%(cid))
    if reverse: fname = fname + "(low)"

    with open(fname+".pkl", "rb") as fp:
        osc_motif = pkl.load(fp)
    
    update_date = osc_motif["metainfo"]["last-updated"]
    print("Loaded oscillation motif information udpated in %s"%(update_date))
    
    winfo = osc_motif["winfo"][wid]
    if len(winfo) == 0:
        print("Word ID %2d does not exist in cluster%d"%(wid, cid))
        return
    elif verbose:
        print("%4d motifs are detected"%(len(winfo)), end=',')

    return osc_motif["winfo"][wid], update_date


def collect_chunk(cid: int, wid: int,
                  summary_obj=None, 
                  target="lfp", st_mua=1e-3, dt=0.01,
                  nequal_len: int=None, nadd: int=0, teq: Tuple=(.5, -.5),
                  norm=False, filt_range: Tuple=None, srate=2000,
                  verbose=True):
    
    """
    Collect data chunk
    
    nequal_len: stack v_set with the same length
    
    """
    
    if summary_obj is None:
        print("load default dataset in %s"%(default_dir))
        summary_obj = hhtools.SummaryLoader(default_dir, load_only_control=True)
        
    winfo, _ = load_osc_motif(cid, wid, reverse=False, verbose=verbose)

    # pre_sos = None if filt_range is None else hhsignal.get_sosfilter(filt_range, srate)
    
    assert target in ("lfp", "mua")
    if target == "lfp":
        _read_value = lambda detail_data: detail_data["vlfp"][1:]
    else: # mua
        _read_value = lambda detail_data: detail_data["mua"]
        
    if filt_range is not None:
        pre_sos = hhsignal.get_sosfilter(filt_range, srate)
        def _get_value(detail_data):
            x = _read_value(detail_data)    
            return hhsignal.filt(_read_value(x[0]), pre_sos), hhsignal.filt(_read_value(x[1]), pre_sos)
    else:
        _get_value = lambda detail_data: _read_value(detail_data)
            
            
    def _norm(x):
        if norm:
            return (x - x.mean())/x.std()
        else:
            return x

    # npoints = 0
    if nequal_len is not None:
        nequal_len += nadd
        
    chunk = []
    nitr_prv = -1
    
    if verbose:
        _range = partial(trange, desc="collect %s in %d%02d"%(target, cid, wid))
    else:
        _range = range
    
    for i in _range(len(winfo)):
        nitr = winfo[i][0]
        tl   = winfo[i][1]
        
        if nitr != nitr_prv: # load detail data
            detail_data = summary_obj.load_detail(cid-1, nitr)
            x1, x2 = _get_value(detail_data)
            nitr_prv = nitr
        
        if (tl[0] < teq[0]) or (tl[1] > detail_data["ts"][-1]+teq[1]):
            continue
            
        # collect voltage segments
        nr = ((tl - detail_data["ts"][0]) * srate).astype(int)
        nr[0] -= nadd
        
        if nr[0] < teq[0]*srate: continue
        if nr[1] > (detail_data["ts"][-1]+teq[1])*srate: continue
        
        if nequal_len is not None:
            x_sub = np.zeros((2, nequal_len)) * np.nan
            nmax = min(nequal_len, nr[1]-nr[0])
            x_sub[:, :nmax] = np.array([
                _norm(x1[nr[0]:nr[0]+nmax]),
                _norm(x2[nr[0]:nr[0]+nmax])
            ])
            
        else:
            x_sub = [_norm(x1[nr[0]:nr[1]]), _norm(x2[nr[0]:nr[1]])]

        chunk.append(x_sub)
        
    if nequal_len is not None:
        chunk = np.array(chunk)
        
    return chunk


def export_mua(detail, dt=0.01, st=0.001):
    mua = hhsignal.get_mua(detail, dt=dt, st=st)
    assert mua.shape[0] == 2
    assert mua.shape[1] == len(detail["vlfp"][0])
    data = np.concatenate([[dt], [st], mua[0], mua[1]]).astype(np.float32)
    prefix = detail["prefix"]
    fname = prefix + "_mua.dat"
    data.tofile(fname)
    
    
def compute_stfft_all(vs, ts, frange=(5, 100), mbin_t=0.01, wbin_t=0.5, srate=2000, t0=0.5):
    psd_set = []
    for v in vs:
        _v, _t = hhsignal.get_eq_dynamics(v, ts, t0)
        psd, fpsd, tpsd = hhsignal.get_stfft(_v, _t, srate, mbin_t=mbin_t, wbin_t=wbin_t, frange=frange)
        psd_set.append(psd)
    return psd_set, fpsd, tpsd


# def _downsample(tq, t, y):
#     yq = [
#         np.interp(tq, t, y[0]),
#         np.interp(tq, t, y[1])
#     ]
#     return np.array(yq)


# def get_mua(detail, dt=0.01, st=0.001):
#     from scipy.ndimage import gaussian_filter1d

#     tmax = detail["ts"][-1]
#     nmax = int((tmax+dt) * 1e3 / dt)
    
#     spk_array = np.zeros((2, nmax))
#     for n, n_spk in enumerate(detail["step_spk"]):
#         ntp = n // 1000
#         spk_array[ntp, n_spk] += 1
        
#     s = int(st * 1e3 / dt)
#     spk_array[0] = gaussian_filter1d(spk_array[0], s)
#     spk_array[1] = gaussian_filter1d(spk_array[1], s)
    
#     t = np.arange(nmax) * 1e-3 * dt
#     return _downsample(detail["ts"], t, spk_array)



# def collect_chunk(cid: int, wid: int,
#                   summary_obj=None, 
#                   nequal_len: int=None, nadd: int=0, teq: tuple=(.5, -.5),
#                   norm=False, filt_range: list|tuple=None, srate=2000,
#                   verbose=True):
    
#     """
#     Collect data chunk
    
#     nequal_len: stack v_set with the same length
    
#     """
    
#     if summary_obj is None:
#         print("load default dataset in %s"%(default_dir))
#         summary_obj = hhtools.SummaryLoader(default_dir, load_only_control=True)
    
#     winfo, _ = load_osc_motif(cid, wid, reverse=False, tag='', verbose=verbose)

    
#     pre_sos = None if filt_range is None else hhsignal.get_sosfilter(filt_range, srate)
    
#     def _norm(x):
#         if norm:
#             return (x - x.mean())/x.std()
#         else:
#             return x

#     # npoints = 0
#     nequal_len += nadd
        
#     v_seg = []
#     nitr_prv = -1
    
#     if verbose:
#         _range = trange
#     else:
#         _range = range
    
#     for i in _range(len(winfo)):
#         nitr = winfo[i][0]
#         tl   = winfo[i][1]
        
#         if nitr != nitr_prv: # load detail data
#             detail_data = summary_obj.load_detail(cid-1, nitr)

#             v1 = detail_data["vlfp"][1]
#             v2 = detail_data["vlfp"][2]
            
#             if filt_range is not None:
#                 v1 = hhsignal.filt(v1, pre_sos)
#                 v2 = hhsignal.filt(v2, pre_sos)
            
#             v1 = (v1 - v1.mean())/v1.std()
#             v2 = (v2 - v2.mean())/v2.std()
            
#             nitr_prv = nitr
        
#         if (tl[0] < teq[0]) or (tl[1] > detail_data["ts"][-1]+teq[1]):
#             continue
            
#         # collect voltage segments
#         nr = ((tl - detail_data["ts"][0]) * srate).astype(int)
#         nr[0] -= nadd
        
#         if nr[0] < teq[0]*srate: continue
#         if nr[1] > (detail_data["ts"][-1]+teq[1])*srate: continue
        
#         if nequal_len is not None:
#             v_sub = np.zeros((2, nequal_len))
#             nmax = min(nequal_len, nr[1]-nr[0])
#             v_sub[:, :nmax] = np.array([
#                 _norm(v1[nr[0]:nr[0]+nmax]),
#                 _norm(v2[nr[0]:nr[0]+nmax])
#             ])
            
#         else:
#             v_sub = [_norm(v1[nr[0]:nr[1]]), _norm(v2[nr[0]:nr[1]])]

#         v_seg.append(v_sub)
        
        
#     if nequal_len is not None:
#         v_seg = np.array(v_seg)
        
#     return v_seg


def load_pickle(fname):
    """ load pickle dataset"""
    with open(fname, "rb") as fp:
        return pkl.load(fp)
    

def reduce_te_2d(te_data_2d, tcut=None):
    from copy import deepcopy
    if tcut is None:
        tcut = te_data_2d["tlag"][-1]
    assert tcut > te_data_2d["tlag"][0]
    
    tlag = te_data_2d["tlag"]
    te_data = deepcopy(te_data_2d)
    
    N = int((tcut - tlag[0])/(tlag[1]-tlag[0]))+1
    N = min([N, len(tlag)])
    
    te_data["tlag"] = te_data["tlag"][:N]
    te_data["te"] = np.zeros((te_data["info"]["ntrue"], 2, N))
    te_data["te_surr"] = np.zeros((te_data["info"]["nsurr"], 2, N))

    for ntp in range(2):
        te_data["te"][:, ntp] = te_data_2d["te"][:,ntp,:N,:N].mean(axis=2) # source, receiver
        te_data["te_surr"][:, ntp] = te_data_2d["te_surr"][:,ntp,:N,:N].mean(axis=2)
        
    if "info" in te_data.keys():
        te_data["info"]["nmax_delay"] = N

    return te_data